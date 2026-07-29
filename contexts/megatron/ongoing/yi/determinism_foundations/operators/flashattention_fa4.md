# FlashAttention-4 On Blackwell: TMEM, Fully-Async MMA, 2-CTA, And Determinism

Date: 2026-07-29

Status: FA4 paper and current official CuTeDSL source study complete for the
BF16 fixed-length, head-dimension-128 mechanism; CPU-only source reasoning,
with B200 SASS/profile and repeated GPU validation deferred

Read the
[FA3 Hopper deep dive](flashattention_fa3.md) first. This note assumes the
reader already understands TMA, warp specialization, online softmax,
persistent scheduling, and the backward ownership fact that a K/V-owner CTA
produces complete `dK/dV` but only a partial `dQ`.

For a graphical reading surface, open the standalone
[FA4 Blackwell visual map](flashattention_fa4.html). Forward and backward live
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

FA4 has four central moves:

1. **fully asynchronous tcgen05 MMA plus TMEM:** accumulators no longer occupy
   the issuing threads' registers;
2. **larger tiles and deeper overlap:** keep at least two matrix operations in
   flight while CUDA-core/MUFU work handles another tile;
3. **less exposed softmax work:** share exponential work across MUFU and FMA
   paths, and skip most unnecessary output rescaling;
4. **2-CTA backward:** pair adjacent CTAs, reduce duplicated SMEM operand
   traffic, reshape the `dQ` reduction through DSMEM, and halve global `dQ`
   reduce-add operations.

The important negative statement is:

> **FA4 does not make the backward ownership conflict disappear.**

Default backward still has many K/V-owner work items contributing to the same
`dQ`. The fast path is therefore nondeterministic. The deterministic path
imposes a fixed semaphore order on those global reductions.

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
CTA 0: stages its share of operand B, owns half of accumulator M rows
CTA 1: stages its share of operand B, owns the other half
                         |
               one cta_group::2 MMA
                         |
               combined M = 256 tile
```

Compared with two independent 1-CTA MMAs, the pair can avoid redundantly
staging the full B operand. The pair must:

- be launched together in a cluster;
- remain resident while the operation is in flight;
- agree on 2-CTA TMEM/MMA mode;
- use cluster barriers and, where needed, DSMEM.

Two CTAs remain two CUDA CTAs. They are not merged into a single 1024-thread
CTA, and neither CTA spans multiple SMs. The cluster contract co-schedules
them and permits cross-CTA cooperation.

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

The paper's higher-level wording groups the control work more coarsely. The
pinned source is the authority for the exact current warp IDs.

## 8. Forward: Ping-Pong Dataflow

The two score/output stages alternate:

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

### 8.1 Why one thread per row becomes natural

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

Hardware `MUFU.EX2` remains scarce relative to Tensor Core throughput. FA4
evaluates a small fraction of exponentials with a degree-3-or-higher polynomial
on FMA/ALU paths while the rest use MUFU:

```text
x = floor(x) + frac(x)
2^x = 2^floor(x) * polynomial(frac(x))
```

The paper uses only about 10–25% emulation because doing all values this way
would add register pressure, register traffic, and latency. For BF16 output of
`P`, the paper reports that degree-3 approximation error is dominated by BF16
quantization, but that is still a numerical-policy change:

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
`dK`. Each CTA stages only half of the shared B operand and owns its half of
the output-row accumulator.

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

### 16.1 Cost

Deterministic mode pays for:

- semaphore polling/wait stalls;
- device-wide visibility / release fences;
- less freedom to overlap global reductions;
- possible head-of-line blocking under imbalance.

The paper reports up to 75% of the nondeterministic 1-CTA backward speed for
its deterministic kernel. Thus "minimal overhead" should not be interpreted as
zero overhead or as one fixed percentage for all shapes. It is substantially
better than a naive ordered schedule, but still shape- and mask-dependent.

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
