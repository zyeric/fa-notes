# FlashAttention-4 On Blackwell: TMEM, Fully-Async MMA, 2-CTA, And Determinism

Date: 2026-07-29

Status: FA4 paper and current official CuTeDSL source study complete for the
BF16 fixed-length, head-dimension-128 mechanism; CPU-only source reasoning,
with B200 SASS/profile and repeated GPU validation deferred

Read the
[FA3 Hopper deep dive](fa3.md) first. This note assumes the
reader already understands TMA, warp specialization, online softmax,
persistent scheduling, and the backward ownership fact that a K/V-owner CTA
produces complete `dK/dV` but only a partial `dQ`.

For a graphical reading surface, open the standalone
[FA4 Blackwell visual map](../slides/fa4.html). Forward and backward live
in one document because both are responses to the same asymmetric Blackwell
scaling, but their ownership and determinism arguments remain separate. This
Markdown file is the source of truth.

## 1. Scope And Evidence Envelope

Pinned primary evidence:

- paper:
  [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric
  Hardware Scaling](https://arxiv.org/abs/2603.05451), dated 2026-03-05;
- current official source:
  [Dao-AILab/flash-attention commit
  `849f660f73b176e5ad5670e7f822c7fa9f3eaf8b`](https://github.com/Dao-AILab/flash-attention/tree/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b),
  committed on 2026-07-28;
- NVIDIA references:
  [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/),
  [tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html),
  and
  [PTX fifth-generation TensorCore instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-family-instructions).

The paper establishes the Blackwell problem, intended algorithm, roofline
model, and reported B200 performance. The pinned source establishes one
current dispatch, role allocation, memory layout, pipeline, and deterministic
protocol. They are related but not identical snapshots:

- the paper is primarily about B200/GB200 and its central 2-CTA result is
  backward;
- the current `flash_attn/cute` package calls itself FlashAttention-4 and
  includes separate SM80, SM90, SM100/SM110, and SM120 paths;
- current SM100 forward can also select 2-CTA, and supports features added or
  changed after the paper snapshot.

Do not silently project a paper statement onto every current path.

This pass fixes a simple teaching and audit configuration:

```text
B = 1, Hq = Hkv = 32, Nq = Nkv = 8192, Dqk = Dv = 128
dtype = BF16, fixed length, dense MHA, no dropout
hardware = B200-class SM100
```

Both causal and noncausal ownership are discussed, but the source dispatch
details below use ordinary dense attention unless stated otherwise.

Deferred:

- FP8/FP4 and block scaling;
- head dimensions 192, 256, and 512;
- MLA, absorbed MLA, DeepSeek-specific `(192, 128)` and `Dv=512` paths;
- paged KV, decode, inference mutation, SplitKV, varlen sorting, block sparse,
  local attention, and custom score/mask modifiers;
- SM103/B300, SM110, SM120 consumer Blackwell, and current Hopper tuning;
- exact B200 SASS, issue timing, occupancy, cache traffic, and bitwise probes.

The CPU-only evidence label is:

> **Source-backed schedule, memory-residence, ownership, and ordered-reduction
> model; performance overlap and fixed-artifact bitwise behavior still require
> B200 SASS/profile/repeat validation.**

### 1.1 Source landmarks

Public dispatch:

- [`flash_attn/cute/interface.py`](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/interface.py):
  architecture selection, tile/config choice, 2-CTA predicates, persistent
  selection, backward workspaces, and deterministic semaphores.

Forward:

- [`flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/flash_fwd_sm100.py):
  16-warp role split, tcgen05/TMEM layouts, Q/K/V pipelines, softmax,
  correction, epilogue, persistent scheduler, and 1-/2-CTA modes;
- [`softmax.py`](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/softmax.py):
  row reductions, exponential emulation, and online-softmax state;
- [`tile_scheduler.py`](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/tile_scheduler.py):
  static persistent, CLC, LPT, varlen, and backward schedulers.

Backward:

- [`flash_bwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/flash_bwd_sm100.py):
  five MMA shapes, 16-warp role split, TMEM aliasing, 2-CTA dS exchange,
  global FP32 reduce-add, and semaphore ordering;
- [`flash_bwd_preprocess.py`](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/flash_bwd_preprocess.py)
  and
  [`flash_bwd_postprocess.py`](https://github.com/Dao-AILab/flash-attention/blob/849f660f73b176e5ad5670e7f822c7fa9f3eaf8b/flash_attn/cute/flash_bwd_postprocess.py):
  preprocess statistics / accumulator initialization and final FP32-to-BF16
  conversion.

Generic GPU hierarchy remains owned by
[`gpu-hardware-notes`](https://github.com/zyeric/gpu-hardware-notes). This
note keeps only the H100-to-B200 delta and its concrete FA4 consequences.

## 2. One-Page Mental Model

The version-to-version change is:

```text
FA1 -> FA2:
  improve logical ownership and warp work partition

FA2 -> FA3:
  keep the owner graph, pipeline TMA / WGMMA / softmax on Hopper

FA3 -> FA4:
  keep the broad owner graph again,
  but redesign the pipeline and parts of the online-softmax algorithm
  because Blackwell matrix compute scales faster than SMEM and exp
```

### 2.1 The human-memory version: move, reorder, reduce, order

The most useful one-sentence comparison is:

> **FA3 teaches Hopper to overlap TMA, WGMMA, and softmax. FA4 responds to
> Blackwell making matrix compute about 2x faster without similarly scaling
> SMEM or exponential throughput: move accumulator state, deepen the overlap,
> and reduce the non-MMA work that can no longer be hidden.**

Do not remember FA4 primarily as “a faster GEMM implementation.” B200 already
provides faster Tensor Cores. FA4 changes how the kernel feeds them, where
their results live, what overlaps their execution, and how much surrounding
work remains:

```text
B200 Tensor Core ~= 2x, while SMEM / exp ~= 1x
  -> the bottleneck moves outside the matrix engine
  -> move intermediate state out of scarce RMEM
  -> reorder the pipeline around fully asynchronous MMA
  -> reduce exposed softmax, SMEM traffic, and global reductions
  -> when exact replay is requested, order the remaining many-writer reductions
```

For human recall, use four verbs:

| Verb | Concrete FA3 -> FA4 change | Why it matters |
| --- | --- | --- |
| **Move / 搬家** | WGMMA accumulator in RMEM -> tcgen05 accumulator in TMEM | frees accumulator registers and removes ordinary register-writeback dependency |
| **Reorder / 重排** | larger fully-async MMA, two forward softmax groups plus correction, backward cross-iteration overlap | keeps more independent matrix and non-matrix work in flight |
| **Reduce / 减负** | partial exp, conditional rescale, TMEM residency, 2-CTA B staging, fewer `dQ` global updates | increases effective exp throughput and removes exposed ALU/SMEM/reduction work |
| **Order / 定序** | optional semaphore order for `dQ` and GQA `dK/dV` writers | converts a safe but arrival-ordered global reduction into a fixed floating-point association |

The first three are performance responses to asymmetric scaling. The fourth is
an optional reproducibility constraint. It is related to performance because
FA4 carefully schedules the ordered writers, but it is not free and is not
enabled by the ordinary fast path.

The important negative statement is:

> **FA4 does not make the backward ownership conflict disappear.**

Default backward still has many K/V-owner work items contributing to the same
`dQ`. The fast path is therefore nondeterministic. The deterministic path
imposes a fixed semaphore order on those global reductions. Most GEMM,
softmax, and data-movement work can remain parallel; only contributors to the
same destination are ordered.

### 2.2 What “minimal deterministic overhead” does and does not mean

The paper's introduction calls the deterministic mode “minimal performance
overhead,” but its reported quantitative statement is more informative:

```text
best reported deterministic speed
  = up to 75% of nondeterministic 1-CTA backward speed
```

At a `0.75x` throughput ratio, the same work has:

```text
throughput loss = 1 - 0.75 = 25%
runtime increase = 1 / 0.75 - 1 ~= 33%
```

This does **not** mean zero overhead, does not guarantee `0.75x` for every
shape, mask, or head grouping, and is not a direct comparison against the
fastest nondeterministic 2-CTA path. The useful claim is narrower: ordering
only the conflicting global writes, then using swizzling and an SPT-style
writer order, preserves much more of the surrounding kernel parallelism than
a naive globally serialized implementation.

## 3. Why Blackwell Requires Another Redesign

### 3.1 The asymmetric scaling

For the paper's B200 model:

| Per-SM resource | H100 | B200 | Change |
| --- | ---: | ---: | ---: |
| BF16 Tensor Core | 4096 ops/clock | 8192 ops/clock | 2x |
| exponential / MUFU | 16 ops/clock | 16 ops/clock | unchanged |
| measured SMEM read | 128 B/clock | 128 B/clock | unchanged |
| register file | 64K × 32-bit | 64K × 32-bit | unchanged |
| maximum SMEM per SM | 228 KiB | 228 KiB | unchanged |
| Tensor Memory | absent | 256 KiB/SM | new |

Peak GPU figures and clocks vary by SKU. The paper's useful invariant is the
ratio:

```text
matrix throughput doubles
but exp throughput and SMEM read throughput do not
```

Thus an FA3 schedule that was well overlapped on H100 can expose three new
bottlenecks on B200:

- softmax `max / exp / sum / rescale`;
- repeatedly reading MMA operands from SMEM;
- register pressure and dependencies around large accumulators.

TMEM is the new capacity that makes a different schedule possible; it does not
make SMEM bandwidth or exponential throughput faster.

### 3.2 A small forward roofline example

For `M=N=D=128`, one K/V iteration has two GEMMs:

```text
QK^T:  (128,128) = (128,128) @ (128,128)
PV:    (128,128) = (128,128) @ (128,128)
```

The paper estimates:

| Resource | Cycles |
| --- | ---: |
| two MMA operations | 1024 |
| SMEM reads | 768 |
| 128×128 exponentials | 1024 |

At `M=256, N=D=128`, all three roughly double:

| Resource | Cycles |
| --- | ---: |
| MMA | 2048 |
| SMEM reads | 1536 |
| exponential | 2048 |

The lesson is not that one unit alone wins. It is that useful performance now
requires matrix work and softmax work to overlap almost completely.

### 3.3 What a GPU cycle means here

A cycle is one tick of the relevant GPU clock domain. At a nominal core clock
of 1.85 GHz:

```text
1 cycle = 1 / 1.85 GHz ≈ 0.54 ns
```

This conversion is only an intuition. Real clocks change with power,
temperature, SKU, and boost state, and different GPU subsystems need not share
one identical clock domain.

More importantly, distinguish latency from throughput:

```text
instruction latency:
  how many cycles pass before one instruction's result becomes usable

pipeline throughput:
  how much independent work the hardware can accept or complete per cycle
```

An MMA can have a latency of many cycles while the Tensor Core pipeline accepts
new independent MMA work before the earlier operation completes. Therefore
`8192 BF16 ops / cycle / SM` is an aggregate peak-throughput statement, not a
claim that one whole MMA instruction starts and finishes in one cycle.

The FA4 paper's roofline cycle counts are approximately:

```text
resource time = work assigned to that resource / peak resource throughput
```

For example, the forward MMA estimate divides the FLOPs from `QK^T` and `PV`
by 8192 operations per cycle per SM. These estimates answer:

> If this resource were the only limit and reached peak throughput, how many
> cycles would its assigned work occupy?

Taking the maximum across the Tensor Core, SMEM, and exponential paths gives a
lower-bound bottleneck model. It is not the measured kernel wall time because
the simplified model omits pipeline fill/drain, dependencies, instruction
issue, register/L2 traffic, barriers, occupancy, tails, and imperfect overlap.

## 4. Blackwell Primitives Before FA4

### 4.1 tcgen05 is the instruction interface; Tensor Core is the hardware

Use the layers precisely:

| Name | Layer | Role |
| --- | --- | --- |
| `tcgen05.mma` | Blackwell PTX instruction family | launch matrix work |
| `cute.gemm(...)` | CuTeDSL abstraction | lower to the selected MMA |
| TMEM | 256 KiB on-SM Tensor Memory | accumulator / optional A operand |
| fifth-generation Tensor Core | physical execution pipelines | perform matrix arithmetic |

One elected thread issues a tcgen05 MMA for a 1-CTA operation, or one elected
thread in a CTA pair issues a 2-CTA operation. That does not mean one thread
computes the matrix. It describes instruction issue ownership.

The architecture progression is:

| Path | Collective packaged by hardware | B operand | Accumulator |
| --- | --- | --- | --- |
| warp WMMA / `mma.sync` | one warp's matrix microtile | explicit per-warp register fragments | warp registers |
| Hopper WGMMA | four warps / one warpgroup | one SMEM descriptor for the collective | warpgroup registers |
| Blackwell `tcgen05` | larger 1- or 2-CTA asynchronous MMA | current and optionally peer-CTA SMEM | TMEM |

The growing primitive reduces software-visible fragment and routing work. It
does not choose FA4's logical Q/K tiles, SMEM layouts, TMEM aliasing, role
split, pipeline order, or deterministic reduction protocol. Those remain
kernel responsibilities.

### 4.2 Fully asynchronous means the accumulator does not block register writeback

Hopper WGMMA is asynchronous, but its accumulator is ultimately a distributed
register fragment. Blackwell tcgen05 writes the accumulator directly to TMEM.

```text
Hopper:
  SMEM operands -> WGMMA -> distributed register accumulator

Blackwell:
  SMEM/TMEM A + SMEM B -> tcgen05 MMA -> TMEM accumulator
```

The issuing warp can launch MMA and continue independent instruction work.
Consumers wait through `tcgen05.commit` / `mbarrier` protocols before loading
the result from TMEM. Dependency freedom, not merely the spelling "async",
creates the overlap opportunity.

A `128 x 128` FP32 accumulator occupies 64 KiB:

```text
128 * 128 * 4 bytes = 64 KiB
```

Four such live regions reach Blackwell's 256 KiB/SM TMEM capacity. On Hopper,
the corresponding WGMMA fragments consume the participating threads' general
register budget. Direct-to-TMEM accumulation is therefore what makes several
large intermediate lifetimes and the more flexible FA4 schedule plausible.
CUDA-core softmax or epilogue work still needs `tcgen05.ld` to move selected
TMEM slices into lane registers.

### 4.3 TMEM is neither register file nor shared memory

| Property | RMEM | SMEM | TMEM |
| --- | --- | --- | --- |
| ownership | thread-private | CTA-visible | warp-synchronous / MMA-oriented |
| normal use | scalar/vector state | staged operands and exchange | MMA accumulators, optional operand A |
| MMA result | Hopper destination | no | Blackwell destination |
| direct ordinary addressing | per-thread registers | load/store address | explicit tcgen05 copy/address model |
| capacity pressure relieved | no | no | accumulator register pressure |

TMEM is explicitly allocated/deallocated in 32-column, 16-KiB granules. FA4's
current source requests the maximum 512 columns and manually assigns offsets.
Aliasing is deliberate: an intermediate can reuse a TMEM region only after its
last consumer has completed.

### 4.4 2-CTA MMA

A 2-CTA operation is one cooperative MMA issued by a fixed pair of adjacent
CTAs in the same cluster:

```text
representative M=256, N=K=128:

CTA 0:
  A0 = A[0:128, :]
  local SMEM B_left  = B[:, 0:64]
  local TMEM D0      = D[0:128, :]

CTA 1:
  A1 = A[128:256, :]
  local SMEM B_right = B[:, 64:128]
  local TMEM D1      = D[128:256, :]

one cta_group::2 MMA:
  D0 = [A0 @ B_left | A0 @ B_right]
  D1 = [A1 @ B_left | A1 @ B_right]
```

The output and A ownership split is along M; B storage is split across the peer
CTAs along N. The pair holds one logical B tile as two local SMEM halves, and
the `cta_group::2` hardware path consumes the combined operand. CTA 0 does not
first copy all of CTA 1's half into its own SMEM with ordinary thread loads.

Compared with two independent 1-CTA MMAs that each need the complete B tile,
the pair removes one full duplicated B residence and roughly halves B operand
SMEM traffic across the pair. It does not halve total SMEM traffic because
other MMA operands, DSMEM exchange, and epilogue paths remain. The pair must:

- be launched together in a cluster;
- remain resident while the operation is in flight;
- agree on 2-CTA TMEM/MMA mode;
- use cluster barriers and, where needed, DSMEM.

Two CTAs remain two CUDA CTAs. They are not merged into a single 1024-thread
CTA, and neither CTA spans multiple SMs. The cluster contract co-schedules
them and permits cross-CTA cooperation.

The `tcgen05` collector buffer is a separate mechanism. The 2-CTA split is
spatial reuse across peers for one logical operand; collector `fill/use/lastuse`
annotations permit opportunistic temporal reuse of an unchanged A or B across
successive MMA instructions. Neither contract exposes the exact physical SRAM
bank or crossbar transaction count.

The generic operand-path explanation lives in
[`gpu-hardware-notes/docs/notes/cuda-kernel-patterns.md`](https://github.com/zyeric/gpu-hardware-notes/blob/main/docs/notes/cuda-kernel-patterns.md#how-2-cta-tcgen05-distributes-b).

### 4.5 CuTeDSL is part of the implementation envelope

The current path is Python source embedded in CuTeDSL:

```text
Python/CuTeDSL specialization
  -> MLIR/CUTLASS lowering
  -> PTX tcgen05 / TMA / barrier operations
  -> ptxas machine code
```

It still behaves like a compiled, specialized kernel rather than an
interpreted Python loop. Dtype, head dimensions, mask features, 1-/2-CTA,
stages, scheduler, and deterministic mode enter the compile key.

The paper reports 20–30x shorter single-kernel compile time than the compared
FA3 C++ template path. That is a build-system measurement, not a numerical
equivalence guarantee. Compiler/CuTeDSL/CUTLASS versions and generated binary
remain part of deterministic qualification.

## 5. Forward Ownership Before The Pipeline

Logically, forward still gives one work item complete ownership of output
query rows:

```text
owned Q rows
  -> visit required K/V column tiles in a defined order
  -> maintain online (m, l, O_unnormalized)
  -> final normalize
  -> write O and LSE once
```

For the teaching shape:

```text
Nq / 128 = 64 Q subtiles per head
Nkv / 128 = 64 K/V tiles per noncausal Q subtile
B * H = 32 independent head planes
```

The exact current work item can contain two ping-pong Q stages and, on eligible
noncausal inputs, a 2-CTA cluster. Therefore do not equate:

```text
one 128-row mathematical subtile == one launched CTA
```

The stable statement is that each final output row has one logical owner;
K/V is read redundantly by different output-row owners.

## 6. Forward: Where Everything Lives

For the representative SM100 kernel:

| Object | Main residence | Why |
| --- | --- | --- |
| Q, K, V inputs | HBM; staged in SMEM | TMA feeds MMA operands |
| score `S` accumulator | TMEM FP32 | direct tcgen05 destination |
| score row being transformed | lane-private RMEM FP32 | CUDA-core/MUFU softmax |
| probability `P` | TMEM in input precision | direct A operand for `P @ V` |
| running output `O` | TMEM FP32 | persistent `PV` accumulator |
| row max / sum and correction metadata | RMEM plus small TMEM/SMEM signaling state | online softmax |
| final O | HBM BF16 | public output |
| final LSE | HBM FP32 | backward input |

The current source assigns two TMEM score regions and two output regions.
`P` overlays part of the corresponding `S` storage after score consumers are
done. This is a lifetime proof, not just a capacity trick:

```text
S ready in TMEM
  -> softmax loads S into RMEM
  -> S no longer needed in that region
  -> write P back into aliased TMEM region
  -> P @ V consumes P
```

## 7. Forward: The 16-Warp Role Split

At the pinned source, the main SM100 forward CTA contains 16 warps / 512
threads:

| Warps | Role |
| --- | --- |
| 0–3 | softmax group 0 |
| 4–7 | softmax group 1 |
| 8–11 | output-correction warpgroup |
| 12 | tcgen05 MMA issuer/control |
| 13 | output epilogue when separate |
| 14 | TMA load/control |
| 15 | empty or dynamic scheduler role |

This is warp specialization. The 512 threads do not all execute every stage.
Register budgets are reassigned by role; softmax threads receive many more
registers because each thread can hold an entire 128-element score row plus
temporary state.

These warp ranges are CTA-local software identities used to select branches in
one kernel control-flow graph; they are not separate per-warp SASS binaries.
The exact mapping from those warp IDs to Blackwell's physical warp schedulers
is not a documented CUDA/PTX guarantee. A four-way round-robin mapping is a
useful performance hypothesis, but the schedule's correctness comes from its
barrier and TMEM protocols rather than that assumed mapping. See the shared
[GPU execution-model note](https://zyeric.github.io/gpu-hardware-notes/notes.html#one-kernel-instruction-stream-many-warp-contexts)
for the evidence boundary between documented static scheduler sets and an
architecture-specific warp-ID mapping.

The paper's higher-level wording groups the control work more coarsely. The
pinned source is the authority for the exact current warp IDs.

## 8. Forward: Ping-Pong Dataflow

### 8.1 The Missing FA3-To-FA4 Causal Chain

Do not compress the argument to:

```text
softmax becomes a bottleneck -> use TMEM
```

TMEM does not execute `max`, `exp`, or `sum`, and by itself does not increase
softmax throughput. The connection runs through the amount of independent MMA
work that can remain in flight while softmax is using the CUDA-core / MUFU
paths.

FA3's Hopper forward has two consumer warpgroups owning disjoint 64-row output
macrotiles. Each group alternates between a Tensor Core bundle and softmax:

```text
time ------------------------------>

consumer WG0:  PV_j + QK^T_{j+1}    softmax(S_{j+1})   PV + next QK^T
consumer WG1:  softmax(S_j)          PV_j + QK^T_{j+1} softmax(next S)
```

The `PV + next QK^T` bundle from one group is intended to occupy the Tensor
Core while the other group uses the non-MMA paths for softmax. The groups do
not hand partial output rows to each other; ping-pong changes their issue
timing while each retains its own register-resident `S`, `P`, running `O`, and
statistics. See the
[FA3 overlap-layer audit](fa3.md#6-the-two-overlap-layers) and the
[FA3 paper](https://arxiv.org/abs/2407.08608), Sections 3.1--3.2.

The FA3 paper's d128 throughput argument estimates that exponential work takes
about half as many cycles as the corresponding matmul work on H100. This makes
the other consumer group's Tensor Core bundle a plausible cover window for
one softmax section. It is an intended resource-overlap model, not proof of an
exact SASS timeline.

On B200 the same BF16/FP16 Tensor Core work takes roughly half as long, while
MUFU exponential throughput remains unchanged. The FA4 paper's simplified
`M=N=d=128` model consequently assigns about 1024 cycles both to the two MMAs
and to the exponentials. The cover window has tightened:

```text
H100 teaching model:
  one group's softmax       [========]
  other group's MMA bundle  [================]

B200 direction of change:
  one group's softmax       [========]
  other group's MMA bundle  [========]
```

If the second line finishes first, the Tensor Core needs another
dependency-ready `QK^T` or `PV` operation. If the next `P` is not ready and no
free score destination exists, it idles even though the kernel still has
softmax work. The FA4 paper also explicitly prevents the two softmax
warpgroups from overlapping their exponential critical sections, so simply
running both softmax groups at once does not double MUFU throughput.

### 8.2 What Pipeline "Depth" Means Here

There are several unrelated stage counts in this kernel. The K/V circular
SMEM pipeline depth counts TMA buffers. The relevant GEMM-softmax depth instead
asks:

> While one softmax critical section is running, how many future score tiles,
> probability tiles, and dependency-ready MMA bundles can legally remain in
> flight before a producer runs out of storage or hits a true dependency?

FA3 already has an intra-warpgroup two-stage schedule: it can retain a current
`P` for `PV` while an asynchronous WGMMA produces the next `S`. Extending that
lookahead requires another full score fragment in registers. The FA3 paper's
three-stage experiment therefore trades potential overlap for more register
pressure. This is separate from, and composes with, the two-warpgroups
ping-pong schedule above.

It is imprecise to summarize FA4 as merely having a larger numeric stage
count. The more durable change is that its pipeline is **more decoupled and
more bufferable**:

- not-yet-consumed score tiles and persistent outputs reside in TMEM rather
  than consuming the softmax workers' general registers;
- one control role can issue MMA work for both output streams instead of each
  softmax warpgroup also owning its MMA accumulator registers;
- `P` crosses from a softmax warpgroup to the Tensor Core through TMEM;
- output rescaling crosses to a separate correction warpgroup instead of
  extending the softmax warpgroup's register dependency chain.

TMEM provides the legal residence and handoff protocol for this schedule. It
does not create an unlimited reservoir of work: if steady-state softmax
throughput remains lower than the available MMA throughput, every finite
buffer eventually fills and the Tensor Core still waits.

### 8.3 FA4 Rebuilds Ping-Pong Around TMEM

FA4 maps the two independent streams to 128-row high and low Q/output tiles.
The paper-level role structure is:

```text
MMA/control       produces S_high/S_low and consumes P_high/P_low
softmax WG0       consumes one S stream and produces its P
softmax WG1       consumes the other S stream and produces its P
correction WG     consumes rescale statistics and corrects O
```

For `d=128`, two persistent FP32 output tiles consume half of the 256-KiB
TMEM. The remaining half can hold either two FP32 `S` tiles or four BF16/FP16
`P` tiles. FA4 chooses two `S` regions whose addresses are later reused by the
smaller `P` values. That choice lets the prologue issue both independent score
MMAs before either softmax result exists:

```text
prologue:
  MMA -> S_high in TMEM
  MMA -> S_low  in TMEM

steady-state handoff for either stream:
  S ready in TMEM
    -> softmax loads S into RMEM
    -> the S lifetime ends
    -> P overwrites the aliased TMEM region
    -> P @ V consumes P from TMEM
```

This is the same high-level ping-pong objective as FA3--softmax for one output
stream overlaps dependency-ready Tensor Core work for the other--but with a
different implementation. FA3 alternates two register-owning, do-both
consumer groups. FA4 has a separate MMA issuer and uses TMEM as the bounded
handoff storage between MMA, two softmax groups, and correction.

The current source-backed role timeline is schematically:

```text
time ---->

MMA/control:   S0=Q0K^T       S1=Q1K^T       P0V        P1V
softmax WG0:       load S0 -> max/exp/sum -> write P0
softmax WG1:                       load S1 -> max/exp/sum -> write P1
correction WG:     correct O_old0   correct O_old1
load warp:      TMA K/V next      TMA K/V next
```

Key dependencies are expressed with mbarrier-backed pipelines:

- TMA signals Q/K/V stage readiness to MMA;
- MMA signals that `S` is complete in TMEM;
- softmax/correction signal that `P` and the corrected output are ready;
- MMA consumes `P` for `PV`;
- MMA signals the new output accumulator to correction;
- epilogue waits for the final corrected accumulator.

The overlap is safe only because every reused TMEM/SMEM region has an explicit
producer-complete and consumer-release protocol.

The resulting causal chain is:

```text
B200 makes the FA3-style MMA cover window shorter
  -> more of softmax can become exposed
  -> fully-async MMA plus TMEM decouple and buffer the ping-pong handoffs
  -> partial exp emulation raises effective exp throughput
  -> conditional rescaling and a correction role reduce/move remaining work
```

The fully-async/TMEM step improves overlap opportunity. Partial exponential
emulation changes effective exponential throughput; conditional rescaling
reduces correction work, and the correction role moves the remaining work off
the softmax warpgroup's critical path. Neither category should be silently
credited with the other's effect.

### 8.4 Why one thread per row becomes natural

A Blackwell accumulator tile is naturally 128 rows high. Four softmax warps
contain 128 lanes, so a warpgroup can assign:

```text
lane/thread r -> score row r, 128 FP32 elements
```

Each row's `max`, exponentials, and sum are private to one thread. FA3's
inter-thread shuffle for pieces of the same row is avoided. The tradeoff is
high per-thread register demand.

## 9. Forward: Two Algorithm Changes, Not Just A New Schedule

### 9.1 Partial software emulation of `exp2`

#### 9.1.1 Baseline: how the ordinary fast exponential path works

This discussion is about the fast path used by an attention kernel, not every
possible implementation of CUDA `expf`. After a softmax worker has loaded one
FP32 score row from TMEM into lane-private registers, it first subtracts the
row maximum. Mathematically it needs:

```text
p_i = exp(s_i - m)
```

The fast GPU path rewrites the natural exponential in base 2:

```text
x_i = (s_i - m) * log2(e)
p_i = 2^x_i
```

Each lane issues `exp2`-class work for its register-resident elements. That work
is lowered to the specialized MUFU `EX2` path, and its result returns to
registers before the row sum and the BF16 `P` fragment are produced:

```text
FP32 S element in RMEM
  -> subtract row max and apply base-2 scale on FP32 ALU
  -> MUFU.EX2
  -> approximate FP32 exponential in RMEM
  -> row sum + BF16 conversion
  -> P in TMEM
  -> Tensor Core consumes P for P @ V
```

MUFU is pipelined; this is not one lane blocking the whole SM until one scalar
function finishes. The relevant limit is its aggregate throughput. B200
provides 16 exponential operations per clock per SM. A `128 x 128` score tile
contains 16,384 values, so the roofline cost is:

```text
T_exp = 128 * 128 / 16 = 1024 cycles
```

For `M=N=d=128`, the two forward MMAs `Q @ K^T` and `P @ V` together also take
about 1024 idealized Tensor Core cycles. The exp path is therefore as heavy as
all matrix work for the tile. FA4 overlaps the paths, but any unhidden part of
softmax directly delays the next `P @ V`.

#### 9.1.2 Why a software exponential can help

Blackwell doubled BF16 Tensor Core throughput while B200 kept the previous
generation's 16-op/clock MUFU throughput. It did not make a hardware
`MUFU.EX2` instruction itself easier to optimize from software. Instead, FA4
uses ordinary FMA and integer ALU capacity as a second exponential production
path that can operate alongside MUFU.

For an element assigned to the emulated path, FA4 uses range reduction:

```text
x = n + f
n = floor(x)
f = x - n, where f is in [0, 1)

2^x = 2^n * 2^f
2^f ~= p0 + p1*f + p2*f^2 + p3*f^3 + ...
```

The concrete steps described by the paper are:

1. clamp `x` to at least `-127` to avoid underflow;
2. obtain `floor(x)` with a round-down range-reduction trick;
3. compute the fractional remainder `f`;
4. evaluate the polynomial for `2^f` in Horner form using a short FMA chain;
5. construct the `2^n` factor through IEEE-754 exponent-bit manipulation and
   combine it with the polynomial result.

Conceptually the Horner part is:

```text
t = fma(f, p3, p2)
t = fma(f, t,  p1)
t = fma(f, t,  p0)
```

This takes more instructions and registers than one MUFU operation, but it uses
a different set of execution resources. It helps when MUFU is saturated while
enough FMA/ALU issue capacity remains available.

#### 9.1.3 Why only part of the row is emulated

**Partial** describes a work split across independent score elements. It does
not mean that MUFU and FMA each compute half of one exponential:

```text
one softmax row
  about 75-90% of elements -> hardware MUFU.EX2 ---------\
  about 10-25% of elements -> integer ALU + FMA poly ----+-> one P row
```

The paper uses only about 10–25% emulation because a polynomial needs several
instructions and temporary registers. Moving every exponential to this path
would shift the bottleneck to FMA issue, register traffic, latency, or spills
instead of removing it. The optimization is thus **load balancing**, not a
claim that software `exp2` is individually faster than `MUFU.EX2`. The fraction
is tuned for the tile and hardware so that the MUFU and emulation paths finish
their assigned work at roughly the same time.

**Kernel-design takeaway:** after faster matrix hardware exposes softmax as the
critical path, reducing softmax cost does not have to mean making one
exponential instruction faster. FA4 instead uses heterogeneous execution
resources more evenly: keep most exponentials on the specialized MUFU path and
move only enough independent elements to the general FMA/ALU path to shorten
the combined critical path:

```text
before: softmax time ~= all work queued through MUFU
after:  softmax time ~= max(MUFU share, FMA-emulated share) + coordination
```

For BF16 output of `P`, the paper reports that degree-3 approximation error is
dominated by BF16 quantization: its maximum FP32 relative error is about
`8.8e-5`, while BF16 quantization is about `3.9e-3`, and 99% of tested values
are within one BF16 ULP of the hardware path. That is still a numerical-policy
change:

- it may differ bitwise from hardware `EX2`;
- the emulation fraction is a tuned specialization parameter;
- B300/SM103 can select a different policy because its exp throughput changes.

Therefore deterministic means repeatable under one fixed exponential policy,
not equal to FA3 or to every FA4 build.

### 9.2 Conditional online-softmax rescaling

Ordinary online softmax updates:

```text
m_j = max(m_{j-1}, rowmax(S_j))
O_j = exp(m_{j-1}-m_j) O_{j-1} + exp(S_j-m_j) V_j
```

FA4 observes that multiplying the entire old output is necessary only when the
reference maximum changes enough. With threshold `tau`:

```text
if rowmax(S_j) - m_ref > tau:
    rescale O_old and advance m_ref
else:
    keep m_ref and accumulate exp(S_j - m_ref) V_j
```

The true row maximum and denominator are still tracked, and final
normalization corrects the chosen intermediate scale. The old probabilities
are not revisited or rewritten. `P_j` is normalized relative to the current
reference scale, consumed immediately by `P_j @ V_j`, and its contribution is
merged into the running unnormalized output.

The current source applies a warp-uniform decision when any lane requires
rescaling, avoiding divergent control within the warp.

## 10. Why Forward Is Expected To Be Deterministic

The important proof is ownership, not whether instructions are asynchronous:

1. a logical output-row owner visits K/V tiles in a defined order;
2. online row statistics and output accumulation stay inside that owner or its
   fixed 2-CTA cooperative group;
3. independent work items do not race to floating-point-add the same ordinary
   output tile;
4. barriers constrain every TMEM/SMEM handoff before reuse;
5. O and LSE are written once by their owner.

Async scheduling can change *when* independent instructions finish without
changing the dependency-ordered arithmetic graph inside one fixed compiled
kernel.

Qualification boundaries:

- SplitKV creates partial outputs and a combine kernel, so its combine order
  must be audited separately;
- dropout additionally requires identical RNG seed/offset consumption;
- different 1-/2-CTA dispatch, tile size, exp-emulation fraction, compiler, or
  architecture may produce different bits;
- source reasoning does not replace repeated B200 exact comparisons.

## 11. Backward Geometry Before Optimization

Backward computes:

```text
S^T  = K @ Q^T                  # recompute scores
dP^T = V @ dO^T
P, dS = elementwise(S, LSE, dP, D)
dV  += P^T  @ dO
dK  += dS^T @ Q
dQ_partial = dS @ K
```

The main work item fixes a K/V row tile and streams Q tiles:

```text
outer grid identity: one K/V-owner tile
inner loop:          all relevant Q tiles

complete local outputs: dK tile, dV tile
partial output:         dQ for each visited Q tile
```

For `N=8192` and `tile_n=128`, each head has 64 K/V owners. Every Q tile
therefore receives up to 64 partial `dQ` contributions. That is the
many-to-one edge.

The five GEMM partition dimensions are:

| GEMM | Output rows / non-reduction owner | Reduction dimension |
| --- | --- | --- |
| `K @ Q^T -> S^T` | K rows | D |
| `V @ dO^T -> dP^T` | V/K rows | Dv |
| `P^T @ dO -> dV` | K rows | Q rows |
| `dS^T @ Q -> dK` | K rows | Q rows |
| `dS @ K -> dQ_partial` | Q rows | K rows in this owner |

The transposed `S^T`, `dP^T`, `P^T`, and `dS^T` are chosen so the K/V-owner
layout naturally feeds complete `dK/dV`. They are logical views / TMEM and
SMEM layouts, not necessarily materialized global transposes.

## 12. Backward: TMEM Unlocks A Different Five-GEMM Pipeline

FA3 register accumulators impose strong lifetime and ordering constraints.
FA4 can keep several results in TMEM while CUDA-core work consumes another.

For the ordinary d128 path, the current source aliases regions approximately
as:

```text
TMEM region A: S <-> P
TMEM region B: dV accumulator
TMEM region C: dP <-> dS <-> dQ   (1-CTA; 2-CTA offsets differ)
TMEM region D: dK accumulator
```

`dK` and `dV` accumulate across the whole Q loop and cannot alias each other.
Short-lived `S/P` and `dP/dS/dQ` can reuse storage after their last consumer.
The paper notes that only four 128×128 FP32 accumulator tiles fit, so scheduling
is partly a graph-coloring problem over TMEM lifetimes.

The core cross-iteration idea is:

```text
iteration j:
  compute S_j, dP_j
  CUDA cores form P_j and dS_j

overlapped:
  dQ_{j-1}, dK_{j-1} MMAs from the previous iteration
```

This supplies at least two MMA operations during the softmax/elementwise
critical path, something FA3's register-resident graph could not sustain as
freely.

### 12.1 Current backward warp roles

The pinned SM100 source again uses 16 warps / 512 threads:

| Warps | Role |
| --- | --- |
| 0–3 | reduce/write partial `dQ` |
| 4–11 | elementwise compute plus `dK/dV` epilogues |
| 12 | tcgen05 MMA issue/control |
| 13 | TMA load/control |
| 14 | 2-CTA dS relay |
| 15 | empty |

This is larger than the older FA1/FA2 4- or 8-warp teaching CTA. It is still
one CTA resident on one SM. Warps take turns issuing instructions through the
SM's schedulers; 16 warps do not imply 16 physical MMA engines or 16 MMAs
completing simultaneously.

## 13. Why Backward Is SMEM-Bound Even With TMEM

For `M=N=D=128`, the paper estimates:

| Resource | 1-CTA cycles |
| --- | ---: |
| five MMA operations | 2560 |
| MMA-operand SMEM reads | 2048 |
| dS SMEM write | 256 |
| dQ FP32 SMEM write + read | 1024 |
| total SMEM | 3328 |
| exponentials | 1024 |

Five GEMMs have ten operands. TMEM can directly supply only two of those
operand paths in the analyzed schedule; the rest still consume SMEM bandwidth.
The dQ path also stages FP32 partials through SMEM for the global reduction.

Thus the new TMEM capacity exposes SMEM traffic as the next limiter. This is
the motivation for 2-CTA, not merely "more CTAs means more parallelism."

## 14. 2-CTA Backward Step By Step

### 14.1 Four GEMMs use a larger cooperative tile

The CTA pair covers a combined `M=256, N=K=128` tile for `S`, `dP`, `dV`, and
`dK`. Each CTA stages the N-half of the shared B operand and owns its M-half of
the output-row accumulator. The paired Tensor Core path supplies both B halves
to both M-row owners; this is why B can be distributed without turning the
output into a partial-K reduction.

Paper roofline:

| Resource | 1-CTA | 2-CTA |
| --- | ---: | ---: |
| MMA | 2560 | 2560 |
| MMA-operand SMEM reads | 2048 | 1536 |
| dS write / exchange | 256 / 0 | 256 / 384 |
| dQ write + read | 1024 | 512 |
| total SMEM | 3328 | 2688 |

2-CTA adds DSMEM exchange, but the saved operand and dQ traffic is larger.
The modeled SMEM excess over MMA falls from about 30% to about 5%.

### 14.2 dQ is the subtle exception

For `dQ = dS @ K`, K rows are the reduction axis. Naively splitting K across
the pair leaves each CTA with only half a reduction, not a final row.

The pair therefore exchanges half of `dS` through DSMEM:

```text
before exchange:
  CTA0: dS for its 128 K rows
  CTA1: dS for its 128 K rows

after exchange/repack:
  each CTA owns 128 Q rows
  each has the full 256-K-row reduction operand for those rows

per-CTA result:
  dQ_partial shape = (128 Q rows, D)
```

The dQ MMA shape is consequently different from the other four:

```text
other four: combined M = 256
dQ:         M = 128, reduction K = 256
```

Each CTA writes only its half of dQ rows, so the pair performs half as many
global reduce-add operations as two independent 1-CTA owners.

### 14.3 2-CTA reduces contention; it does not prove determinism

There are still many different K/V clusters along the sequence. Each cluster
produces one partial contribution to a given dQ row tile. Halving the number
of global updates improves traffic and contention, but independent clusters
can still arrive in different orders.

## 15. The Exact Nondeterministic Operation In Current Source

The current CuTeDSL fast path does not literally call the old scalar CUDA
`atomicAdd(float*, float)` once per lane. It uses a bulk asynchronous FP32
reduce-add:

```text
TMEM dQ partial
  -> tcgen05 load to lane RMEM
  -> cooperative RMEM -> SMEM staging
  -> cp.async.bulk reduce-add.f32
  -> shared global dQ_accum
```

In source this is `cpasync_reduce_bulk_add_f32`.

The semantic conflict is the same:

```text
dQ_accum[address] += partial_from_KV_owner_i
```

Several independent CTAs/clusters target the same address. The hardware makes
each reduction update safe, but does not prescribe a repeatable floating-point
arrival order. Since FP32 addition is not associative, different orders can
produce different low bits.

Use terminology carefully:

- **algorithmic/global atomic reduction:** many writers safely reduce into one
  destination;
- **specific instruction:** current SM100 source uses a bulk reduce-add path,
  while older kernels may use scalar/vector `atomicAdd`.

## 16. Deterministic Backward: Ordered Writers, Not Removed Writers

With `deterministic=True`, the host allocates:

```text
dQ_semaphore[B, Hq, ceil(Nq / tile_m), cluster_size]
```

For GQA/MQA it additionally allocates ordered semaphores for shared `dK` and
`dV` destinations.

For each dQ tile, writer `r` executes:

```text
wait until semaphore == r
perform all FP32 bulk reduce-add stages
wait for those async writes to become globally visible
release/increment semaphore to r + 1
```

The release operation includes the required memory-ordering fence. The next
writer cannot overtake it. Therefore the floating-point association becomes a
fixed sequence.

For causal/local work, equal writer order can cause bubbles because some KV
owners have much shorter loops. FA4 combines:

- descending KV-block launch order;
- ascending Q traversal from the diagonal;
- descending-Q reduction order;
- head/batch swizzling for L2 locality.

The paper describes this as shortest-processing-time-first for the ordered dQ
writes, while its general work scheduling uses longest-processing-time-first
to reduce makespan. These are not contradictory: they optimize two different
queues.

### 16.1 The Semaphore Is Global, Small, And Not A Dynamic Ticket Allocator

Ordinary SMEM is CTA-private, so a counter that orders writers on different
SMs must use a device-global address. The counter array is HBM-addressed but
small enough that its hot working set is normally mediated by the coherent
L2/atomic path. For the teaching shape with `tile_m=128` and 2-CTA cluster
size 2:

```text
B=1, H=32, Nq=8192
counter bytes = 1 * 32 * 64 Q tiles * 2 * 4 B = 16 KiB
FP32 dQ accumulator                               = 128 MiB
```

The synchronization cost is therefore more about dependency latency, polling,
release fences, and backpressure than about counter capacity.

Writer `r` already knows its prescribed logical rank. It does not perform a
fetch-add race to acquire an arbitrary ticket. One elected lane polls an
acquire load until the counter equals `r`; after the bulk `dQ` reduction has
become globally visible, a release-scoped int32 global increment exposes
`r+1`. The current helper protocol corresponds to:

```text
wait_eq:
  while ld_acquire(global_counter) != r:
    poll

arrive_inc:
  red.release.gpu.global.add.s32(global_counter, 1)
```

Two atomic meanings remain distinct:

- the `dQ` data path does not call the legacy lane-wise
  `atomicAdd(float*, float)`, but its bulk reduce-add still has safe global
  reduction semantics;
- the semaphore release is itself an ordered integer global reduction, not a
  floating-point gradient contribution.

The deterministic path keeps the efficient bulk reduction and orders its
invocations. Removing scalar `atomicAdd` syntax is not the determinism proof.

### 16.2 Four-Q/Four-K Causal Toy Schedule

Use equal tile counts so the diagonal and the writer wavefront are visible.
Let `Qi` attend `K0..Ki`:

| Q destination | Legal K/V contributors |
| --- | --- |
| `Q0` | `K0` |
| `Q1` | `K0,K1` |
| `Q2` | `K0,K1,K2` |
| `Q3` | `K0,K1,K2,K3` |

A K/V-owner CTA has the transposed view:

| CTA | Ascending Q traversal from its diagonal | Relative work |
| --- | --- | ---: |
| `CTA(K3)` | `Q3` | 1 |
| `CTA(K2)` | `Q2 -> Q3` | 2 |
| `CTA(K1)` | `Q1 -> Q2 -> Q3` | 3 |
| `CTA(K0)` | `Q0 -> Q1 -> Q2 -> Q3` | 4 |

A naive fixed order `K0 -> K1 -> K2 -> K3` for every `dQ` destination makes
the shortest `CTA(K3)` compute `dQ[Q3]` immediately and then wait for the
longest `CTA(K0)` to reach `Q3`. That is correct but creates head-of-line
blocking.

The FA4 causal idea launches K/V owners in descending order and assigns each
destination a matching descending-K ticket chain:

```text
dQ[Q0]: K0
dQ[Q1]: K1 -> K0
dQ[Q2]: K2 -> K1 -> K0
dQ[Q3]: K3 -> K2 -> K1 -> K0
```

The resulting conceptual wavefront is:

```text
wave 0:  K3/Q3   K2/Q2   K1/Q1   K0/Q0
wave 1:           K2/Q3   K1/Q2   K0/Q1
wave 2:                    K1/Q3   K0/Q2
wave 3:                             K0/Q3
```

Different Q destinations have different counters, so entries across one wave
are not one global serial queue. For each column, however, the ticket chain is
fixed. Every CTA's first write is rank 0 for a different diagonal Q tile, and
later writes arrive in the same diagonal wavefront. This is the useful SPT
intuition behind descending K launch, ascending Q traversal, and descending-Q
reduction priority.

This toy omits partial diagonal tiles, 2-CTA pairing, TMEM stages, multiple
heads, and exact issue timing. It explains the dependency alignment, not a
cycle-accurate source trace. The later
[DASH paper](https://arxiv.org/abs/2601.21824) formalizes the same broader
lesson for deterministic attention: a valid writer order can still be slow
when it is misaligned with partial-ready time.

### 16.3 Cost

Deterministic mode pays for:

- semaphore polling/wait stalls;
- device-wide visibility / release fences;
- less freedom to overlap global reductions;
- possible head-of-line blocking under imbalance.

The paper reports up to 75% of the nondeterministic 1-CTA backward speed for
its deterministic kernel. Thus "minimal overhead" should not be interpreted as
zero overhead or as one fixed percentage for all shapes. At a `0.75x`
throughput ratio this is 25% lower throughput, or about 33% longer runtime for
the same work. The comparison is against the nondeterministic 1-CTA path, not a
proof of parity with the fastest nondeterministic 2-CTA path. It is
substantially better than a naive ordered schedule, but still shape- and
mask-dependent.

## 17. Scheduling: LPT, Persistent CTA, And Determinism Are Separate Axes

Three concepts must remain separate:

1. **persistent CTA:** a resident CTA repeatedly asks for logical work tiles,
   reducing grid-tail and launch/scheduling overhead;
2. **LPT work order:** longer causal/varlen tiles are assigned earlier to
   reduce the final straggler tail;
3. **deterministic writer order:** contributors to one output address acquire a
   semaphore in a fixed arithmetic order.

Persistent execution alone neither creates nor removes an arithmetic race.
LPT can change which logical tile runs first without changing ownership.
Deterministic mode constrains only the many-to-one combine edges that require a
fixed order.

Current-source caveat:

- ordinary eligible noncausal SM100 forward selects a persistent scheduler;
- causal/local or special features select other schedulers;
- the current backward source sets `is_persistent=False` from public dispatch,
  even though the class retains a persistent parameter;
- deterministic backward selects a dedicated LPT/head-swizzled scheduler.

## 18. Paper FA4 Versus Current `flash-attn-4`

| Question | Paper claim | Pinned current source |
| --- | --- | --- |
| target | B200/GB200 | Hopper and several Blackwell families |
| language | CuTeDSL Python | CuTeDSL package with arch-specific files |
| d128 forward tile | two 128-row ping-pong Q tiles | q-stage and optional 2-CTA dispatch |
| forward 2-CTA | not the central paper result | enabled for an eligible SM100 subset |
| backward 2-CTA | central SMEM/dQ optimization | default for d128 when supported/not disabled |
| deterministic dQ | ordered semaphore reductions | `wait_eq` + bulk reduce-add + release increment |
| GQA dK/dV | identified as another global reduction | separate deterministic K/V semaphores |
| B300 exp | mentioned as doubled MUFU | source disables emulation for SM103-family path |

The package name describes an evolving implementation family, not one immutable
paper kernel. Every production verdict must resolve the actual specialization.

## 19. Determinism Scorecard

| Component | Default | `deterministic=True` | Reason |
| --- | --- | --- | --- |
| ordinary forward O/LSE | expected deterministic | same ownership | one logical output owner |
| 2-CTA forward | expected deterministic | same | fixed cooperative group, disjoint final rows |
| SplitKV forward | separate audit | separate audit | partial-output combine |
| backward dQ | nondeterministic | ordered | inter-CTA bulk FP32 reduce-add |
| backward dK/dV, MHA | locally owned | locally owned | one K/V owner per head |
| backward dK/dV, GQA/MQA | nondeterministic combine possible | ordered semaphores | many Q heads share one KV head |
| dropout | RNG-dependent | RNG-dependent | seed/offset consumption is part of replay |

The useful verdict is:

> **For the pinned SM100 CuTeDSL family, ordinary forward has a visible
> single-owner/fixed-cooperation structure. Default backward is
> nondeterministic at global many-writer reductions. Supported deterministic
> backward serializes those reductions in a defined semaphore order. Exact
> repeatability remains scoped to one dispatch, compiler artifact, device, and
> RNG state.**

## 20. Kernel-Design Guidance Generalized From FA4

When studying a new kernel, ask in this order:

1. **What scaled asymmetrically?** Faster matrix engines may expose SMEM,
   special functions, register traffic, or synchronization.
2. **Where does each intermediate physically live?** HBM, SMEM, TMEM, and
   RMEM have different visibility and producer/consumer rules.
3. **Who owns the final output?** Separate logical work identity from CUDA CTA
   identity and physical SM residence.
4. **Which dimension is partitioned?** Never say "split the tile" without
   naming output, reduction, batch/head, or sequence dimension.
5. **Which lifetimes can alias?** Memory reuse is valid only after the last
   consumer, enforced by a visible barrier protocol.
6. **Can non-matmul work overlap at least enough matrix work?** A faster Tensor
   Core can require more independent MMAs, not merely a deeper load pipeline.
7. **Did an optimization reduce traffic or only relocate it?** 2-CTA saves
   SMEM B reads but adds DSMEM exchange.
8. **Are atomic updates fewer or gone?** Fewer updates improve performance but
   do not establish deterministic order.
9. **What changes numerical policy?** Polynomial exp, conditional rescaling,
   tile order, and compiler reassociation can be repeatable yet cross-build
   unequal.
10. **What is the support/dispatch envelope?** A public flag is not proof that
    every specialized path implements it.

## 21. Smallest Future B200 Validation

### 21.1 Resolve the artifact

Record:

- FlashAttention commit and `flash-attn-4` package version;
- public entry point and resolved `flash_fwd_sm100.py` /
  `flash_bwd_sm100.py` path;
- exact B200 SKU, SM count, compute capability, driver, CUDA, CUTLASS, CuTeDSL,
  PyTorch, and Docker digest;
- dtype, shapes, causal/GQA/varlen/dropout features;
- `tile_m`, `tile_n`, Q stages, 1-/2-CTA, cluster shape, scheduler,
  exp-emulation config, and dynamic SMEM;
- compiled cubin/SASS hash and compile-cache key.

### 21.2 Verify the claimed pipeline

Use SASS/Nsight Compute to confirm:

- `tcgen05.mma` issue and TMEM accumulator traffic;
- TMA loads overlapping MMA;
- MUFU and FMA exp paths;
- at least two MMA operations overlapping softmax;
- 1- versus 2-CTA selection and DSMEM traffic;
- SMEM footprint, TMEM columns, registers, spills, active clusters, and stalls;
- count of global dQ bulk reduce-add operations.

### 21.3 Verify exact replay

For at least 100 repeats:

- hash and `torch.equal` forward O/LSE;
- hash and `torch.equal` backward dQ/dK/dV;
- compare `deterministic=False` and `True`;
- include enough K/V tiles to create contention;
- cover noncausal, causal, MHA, and GQA separately;
- restore exact RNG state for dropout;
- repeat warm/cold, fresh process, and clean compile cache.

Use op-guard/tensor hashing to localize the first mismatch:

```text
forward:
  score/softmax stage -> O/LSE owner -> epilogue

backward:
  dQ_accum first -> semaphore order / reduce-add
  then dK/dV -> local owner or GQA combine
```

## 22. Stop Line And Open Questions

This pass is sufficient when the reader can explain:

- why B200 needs more than an FA3 instruction port;
- tcgen05 versus physical Tensor Core;
- why TMEM enables larger asynchronous schedules;
- why forward uses two softmax groups plus a correction group;
- polynomial exp and conditional rescaling;
- the five backward GEMMs and their partition dimensions;
- why 2-CTA saves SMEM but needs DSMEM for dQ;
- why half as many global reductions is still nondeterministic;
- how semaphore order fixes dQ and GQA dK/dV;
- why current `flash-attn-4` is broader than the paper.

Open questions:

- Which exact 1-/2-CTA specialization wins for representative target training
  shapes on B200 and B300?
- Does the production Docker preserve the source-visible exp emulation and
  conditional-rescale policy after compilation?
- What are the measured TMEM, SMEM, and issue-pipeline bottlenecks for causal
  versus noncausal d128?
- How much does 2-CTA improve backward after accounting for DSMEM and cluster
  occupancy?
- What is deterministic overhead for long causal MHA and GQA, and where do
  semaphore stalls occur?
- Are current d192/d256/d512 and MLA paths behaviorally equivalent to the d128
  teaching path, or do they introduce separate ownership/reduction protocols?
- Can op-guard automatically capture dispatch, compiled artifact identity,
  semaphore mode, and the first mismatching gradient tile?
