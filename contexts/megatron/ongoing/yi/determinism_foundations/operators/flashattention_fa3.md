# FlashAttention-3 On Hopper: Hardware, Two-Level Overlap, Forward, And Backward

Date: 2026-07-29

Status: FA3 paper plus early official SM90 source study complete for the
FP16/BF16 mechanism; CPU-only source reasoning, with SASS/profile and repeated
GPU validation deferred

Read the
[FA2 forward delta](flashattention_fa2_forward.md) and
[FA2 backward audit](flashattention_fa2_backward.md) first. This note assumes
the reader already understands tiled online softmax, FA2's one-Q-block forward
ownership, and the backward fact that one K/V-owner CTA produces complete
`dK/dV` but only a partial `dQ`.

For a graphical reading surface, open the standalone
[FA3 Hopper deep-dive visual map](flashattention_fa3.html). Forward and
backward live in one document because they share the same Hopper primitives,
but their ownership and determinism proofs remain separate. This Markdown file
is the source of truth.

## 1. Scope And Evidence Envelope

Pinned primary evidence:

- paper:
  [FlashAttention-3: Fast and Accurate Attention with Asynchrony and
  Low-precision](https://tridao.me/publications/flash3/flash3.pdf),
  dated 2024-07-12;
- early official source after the initial backward release:
  [Dao-AILab/flash-attention commit
  `3669b25206d5938e3cc74a5f7860e31c38af8204`](https://github.com/Dao-AILab/flash-attention/tree/3669b25206d5938e3cc74a5f7860e31c38af8204),
  committed on 2024-08-05;
- NVIDIA architecture and ISA references:
  [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html),
  [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
  and the
  [PTX WGMMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma).

The paper establishes the intended algorithm and performance mechanisms. The
pinned source establishes one realized dispatch and synchronization protocol.
The current `main` branch has evolved substantially since this snapshot, so
tile sizes, schedulers, supported modes, and deterministic protocols must be
re-resolved before using this note as a current-runtime audit.

This pass focuses on:

- Hopper H100 / `sm_90a`;
- FP16/BF16 forward and backward;
- dense MHA, with causal behavior used for the representative forward tile;
- `B=1, H=32, N=8192, D=128`;
- TMA, WGMMA, warp specialization, register reallocation, barriers, and the
  two levels of overlap;
- output ownership and determinism consequences.

This pass does not deeply derive:

- FP8 block quantization, incoherent processing, or the FP8 V transpose;
- current FA3/FA4 dispatch on `main`;
- inference KV-cache mutation, paged attention, SplitKV, or decode scheduling;
- cluster-level performance beyond the source-visible multicast/barrier role;
- measured H100 issue timing, occupancy, cache traffic, or MFU.

The CPU-only evidence label is:

> **Source-backed schedule and ownership model; performance overlap and
> fixed-artifact bitwise behavior still require H100 SASS/profile/repeat
> validation.**

### 1.1 Source landmarks

Forward:

- [`flash_fwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/flash_fwd_launch_template.h):
  tile, warp, stage, cluster, scheduler, and grid dispatch;
- [`flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/flash_fwd_kernel.h):
  producer/consumer roles, `setmaxnreg`, persistent tile loop, mainloop, and
  epilogue;
- [`mainloop_fwd_sm90_tma_gmma_ws.hpp`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp):
  TMA producer, circular pipelines, WGMMA/softmax schedule, and pingpong
  barriers;
- [`kernel_traits.h`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/kernel_traits.h):
  WGMMA atoms, SMEM layouts, stages, and shared-storage unions;
- [`tile_scheduler.hpp`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/tile_scheduler.hpp):
  single-tile, static-persistent, and dynamic-persistent scheduling.

Backward:

- [`flash_bwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/flash_bwd_launch_template.h):
  preprocess, main, and postprocess sequence plus d64/d96/d128 tile dispatch;
- [`flash_bwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/flash_bwd_kernel.h):
  producer, dQ writer, two consumer warpgroups, and register allocation;
- [`mainloop_bwd_sm90_tma_gmma_ws.hpp`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp):
  Q/dO pipelines, five GEMMs, TMA reduce-add, and deterministic semaphore;
- [`epilogue_bwd_sm90_tma.hpp`](https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/epilogue_bwd_sm90_tma.hpp):
  dK/dV register-to-SMEM conversion and TMA stores.

Generic grid/CTA/SM and memory-hierarchy background remains owned by
[`gpu-hardware-notes`](https://github.com/zyeric/gpu-hardware-notes). This
note keeps only the Hopper delta and the concrete FA3 instance.

## 2. One-Page Mental Model

The simplest correct summary is:

```text
FA1 -> FA2:
  change who owns a complete output tile

FA2 -> FA3:
  mostly keep that owner graph
  but redesign how one tile is moved and executed on Hopper
```

Forward still logically assigns one Q tile to the work item that produces the
corresponding complete O tile. Backward still logically assigns one K/V tile
to the work item that produces complete `dK/dV` and partial `dQ`.

FA3 adds two nested overlap layers:

```text
Layer A -- movement versus main computation:

  producer warp(group):  TMA HBM -> SMEM
                                ||
  consumer warpgroups:   WGMMA + online softmax

Layer B -- inside the consumer computation:

  Tensor Core path:      asynchronous QK / PV WGMMA
                                ||
  non-Tensor-Core path:  max, exp, sum, rescale, convert
                          on CUDA-core / multi-function paths
```

Layer B has two realizations in forward:

1. **inter-warpgroup pingpong:** one consumer warpgroup executes softmax while
   another has GEMMs in flight;
2. **intra-warpgroup two-stage pipeline:** the same warpgroup keeps current and
   next score fragments so `PV` for one tile can overlap softmax preparation
   for another.

Backward uses the same TMA/WGMMA/role-specialization foundation, but five GEMMs
and the global many-writer `dQ` combine make its lower pipeline and
determinism boundary different.

## 3. Why Hopper Changes The Optimization Target

### 3.1 A100 versus H100: compute grows faster than HBM bandwidth

Representative peak specifications:

| Resource | A100 80 GB SXM | H100 SXM5 | Approximate change |
| --- | ---: | ---: | ---: |
| dense BF16/FP16 Tensor Core | 312 TFLOP/s | 989 TFLOP/s | 3.17x |
| HBM bandwidth | 2.039 TB/s | 3.35 TB/s | 1.64x |
| L2 capacity | 40 MiB | 50 MiB | 1.25x |
| maximum SMEM per SM | 164 KiB | 228 KiB | 1.39x |
| register file per SM | 64K 32-bit registers | 64K 32-bit registers | unchanged capacity |

The exact comparison depends on SKU, clock, dtype, and whether sparsity is
counted. The important imbalance is robust:

```text
Tensor Core throughput grows much faster
than HBM, SMEM capacity, register capacity, and exp throughput.
```

FA2 already avoids materializing the quadratic score and probability matrices,
so the remaining H100 problem is not merely "read fewer HBM bytes." As the
matmul portion accelerates, previously secondary costs become visible:

- issuing and addressing data movement;
- SMEM stage reuse and synchronization;
- `max`, `exp`, `sum`, rescaling, type conversion, and masking;
- dependencies from `QK^T -> softmax -> PV`;
- register pressure from keeping more in-flight state.

The FA3 paper reports only about 35% H100 utilization for FA2 versus 80-90%
for optimized GEMM and reaches up to 75% with FA3 FP16. That is evidence for a
large scheduling/utilization gap, not proof that HBM alone was the bottleneck.

### 3.2 WGMMA is related to the new Tensor Core generation, but is not a Tensor Core

Use the names precisely:

| Name | Layer | Scope |
| --- | --- | --- |
| `nvcuda::wmma` | CUDA C++ API | one warp |
| `mma.sync` / HMMA lowering | PTX/SASS-level warp MMA path | one warp |
| `wgmma.mma_async` | Hopper PTX instruction family | one warpgroup |
| Tensor Core | physical matrix-compute unit/pipeline | implementation hardware |

Hopper's fourth-generation Tensor Cores are the physical matrix engines.
WGMMA is the new `sm_90a` architectural interface that lets four contiguous
warps cooperate on an asynchronous matrix multiply and efficiently feed those
engines from SMEM or, for supported operands, registers.

It is reasonable to say:

> **The faster Hopper Tensor Core generation created both the need and the
> opportunity for WGMMA's larger asynchronous execution model.**

It is not safe to say:

> "one warpgroup maps one-to-one onto four specific Tensor Cores."

NVIDIA documents the collective instruction semantics, operand layouts, and
completion protocol, but not a portable one-to-one physical Tensor Core
mapping that kernel reasoning should depend on.

### 3.3 What A100 already had, and what Hopper adds

| Mechanism | A100 / SM80 | H100 / SM90a |
| --- | --- | --- |
| Tensor Cores | third generation | fourth generation |
| matrix collective | warp-level MMA | warp-level MMA plus warpgroup WGMMA |
| async global-to-shared | thread-issued `cp.async` | TMA bulk tensor transfer |
| address generation | participating threads generate copy addresses | TMA descriptor and hardware traversal |
| producer/consumer barriers | split arrive/wait supported | async transaction barriers improved |
| warp specialization | software pattern is possible | software pattern plus matching TMA/WGMMA support |
| dynamic registers between roles | no `setmaxnreg` redistribution | warpgroup `setmaxnreg` |

Warp specialization is not a Hopper-only idea. On A100, one can assign some
warps to `cp.async` and others to MMA. Hopper makes this composition much more
effective because a small producer role can launch TMA, consumer warpgroups can
launch asynchronous WGMMA, and registers can be shifted from producers to
consumers.

## 4. Hopper Primitives Before FA3

### 4.1 Warpgroup

A Hopper warpgroup is four contiguous warps:

```text
warpgroup g
  = warp ranks 4g, 4g+1, 4g+2, 4g+3
  = 128 threads
```

It is an ISA-visible collective scope for WGMMA and `setmaxnreg`. An A100
kernel may logically group four warps, but it cannot issue Hopper's
warpgroup-wide WGMMA.

All participating threads must execute the same aligned WGMMA instruction.
The collective constraint is a correctness rule, not a statement that all 128
threads independently calculate a complete matrix product.

### 4.2 WGMMA lifecycle

The conceptual instruction protocol is:

```text
make prior register / SMEM writes visible
    -> wgmma.fence and async-proxy fence as required
issue one or more wgmma.mma_async
    -> wgmma.commit_group
perform independent instructions
    -> wgmma.wait_group N before consuming/reusing dependent state
```

Important distinction:

- `mma_async` means completion is decoupled from issue;
- it does not remove data dependencies;
- `wait_group` still blocks when the required result is not ready;
- scoreboard, async-proxy fences, barriers, and buffer lifetime rules all
  remain part of correctness.

The first WGMMA operand may be sourced from registers or SMEM depending on the
instruction (`RS` versus `SS` in FA3 notation); the other operand uses a SMEM
descriptor. FP32 accumulators remain distributed across private registers of
the 128 participating threads.

#### 4.2.1 WGMMA saves per-warp B staging, not automatically HBM bytes

For the same B tile reused across four different M-row slices, a well-tiled
warp-MMA kernel can already copy B from HBM into CTA SMEM once. The repeated
work is the SMEM-to-register fragment load performed separately by each warp:

```text
warp-scoped MMA:
  B in CTA SMEM
    -> warp 0 B registers -> A0 @ B
    -> warp 1 B registers -> A1 @ B
    -> warp 2 B registers -> A2 @ B
    -> warp 3 B registers -> A3 @ B

WGMMA:
  B in CTA SMEM
    -> one B descriptor for the four-warp collective
    -> [A0; A1; A2; A3] @ B
```

WGMMA therefore removes the software-visible per-warp B fragment loads and B
register replication. It does **not** prove that HBM-to-SMEM bytes shrink, and
PTX does not specify that each B element causes exactly one physical SMEM SRAM
read. Internal broadcasts, bank accesses, and replays remain undocumented.

This is also the right abstraction boundary for the architecture change:
Hopper packages a larger matrix microkernel, but FA3 still has to design the Q
tile, WGMMA macrotile ownership, SMEM layout, TMA stages, ping-pong schedule,
barriers, register budgets, and every reduction outside WGMMA. The primitive
reduces lane/warp micro-orchestration; it does not replace kernel scheduling.

The workload-independent comparison is maintained in
[`gpu-hardware-notes/docs/notes/cuda-kernel-patterns.md`](https://github.com/zyeric/gpu-hardware-notes/blob/main/docs/notes/cuda-kernel-patterns.md#which-b-reads-does-wgmma-save).

### 4.3 TMA lifecycle

TMA is a dedicated Hopper tensor-copy engine:

```text
host constructs tensor map / descriptor
one elected GPU thread issues a bulk tensor copy
TMA generates multidimensional addresses and moves HBM <-> SMEM
transaction barrier records expected and completed bytes
consumer waits before reading the stage
```

Compared with SM80 `cp.async`, TMA reduces per-element address arithmetic and
the number of participating load instructions. "One thread issues TMA" does
not mean one CUDA thread carries the payload through its registers; the copy
engine moves the data.

### 4.4 Circular SMEM stages and transaction barriers

For an `s`-stage K/V buffer, stage `j % s` follows:

```text
EMPTY for generation j
  -> producer_acquire
  -> TMA issue + expected-byte count
  -> FULL when transaction completes
  -> consumer_wait
  -> WGMMA reads the stage
  -> consumer_release
  -> EMPTY for generation j+s
```

The phase/generation is essential. A bare "ready bit" could confuse an old
completion with the next reuse of the same physical SMEM slot.

### 4.5 `setmaxnreg`

Role specialization creates asymmetric register needs:

```text
TMA producer:
  descriptors, counters, pointers
  little matrix accumulator state

WGMMA consumers:
  O, S, dK, dV, and other FP32 fragments
  much larger register demand
```

Hopper lets the producer warpgroup decrease its register allowance and
consumer warpgroups increase theirs from the CTA's fixed register pool. This
redistributes a CTA's admitted registers; it does not dynamically increase SM
occupancy or the physical register-file capacity.

In the pinned d128 forward path:

```text
12-warps CTA:
  producer warpgroup: 24 registers/thread target
  two consumer groups: 240 registers/thread target
```

The values are implementation-specific and should not be treated as H100
constants.

## 5. Three Different Meanings Of "Tile"

FA3 becomes confusing if all three objects are called only "the tile":

| Tile level | Example | Owner / purpose |
| --- | --- | --- |
| logical attention tile | `Q_i -> O_i`, plus one `K_j,V_j` step | defines semantic output ownership |
| CTA tile | `B_M x B_N x D` specialization | defines one work item and resource footprint |
| WGMMA macrotile | commonly 64 rows along M per atom | defines one consumer warpgroup's collective GEMM fragment |
| pipeline stage | one physical K/V SMEM slot or current/next score fragment | defines overlap and reuse lifetime |

For `B_M=128`, the source uses:

```text
AtomLayoutMNK = (B_M / 64, 1, 1) = (2,1,1)

consumer warpgroup 1 -> one 64-row WGMMA M macrotile
consumer warpgroup 2 -> the other 64-row WGMMA M macrotile
```

Their O fragments are disjoint rows. Pingpong scheduling changes issue timing,
not the numeric ownership relation.

## 6. The Two Overlap Layers

### 6.1 Layer A: TMA movement versus consumer computation

Space partition the CTA:

```text
producer role                         consumer roles
-------------                         --------------
acquire stage                         wait for stage FULL
issue TMA K_j/V_j   ||                QK_j WGMMA
prefetch next tile  ||                softmax + PV_j WGMMA
wait only if ring full                release consumed stage
```

This hides HBM-to-SMEM movement and address-generation latency behind useful
computation, subject to:

- enough circular stages;
- enough independent work;
- no producer outrunning all free slots;
- barriers and expected-byte counts being correct;
- register and SMEM footprint still admitting a useful CTA configuration.

### 6.2 Layer B: Tensor Core work versus non-Tensor-Core work

One attention step has a true dependency:

```text
S_j = Q K_j^T
  -> max / exp / sum -> P_j
  -> P_j V_j
```

No scheduler can make `P_j V_j` start before `P_j` exists. FA3 pipelines
different tiles or different consumer warpgroups:

```text
Tensor Core async path:   QK_j ............. PV_{j-1}
non-Tensor-Core path:           softmax(S_{j-1}) .....
```

The non-Tensor-Core side is not only "CUDA cores." `max`, additions,
multiplications, shuffles, masks, and conversions use ordinary execution
paths; exponential is emitted through the multi-function/special-function
path. The durable category is:

```text
Tensor Core WGMMA
versus
CUDA-core / multi-function / shuffle / conversion work
```

### 6.3 Why two consumer warpgroups help

The paper's pingpong schedule intentionally orders WGMMA issue:

```text
time ---->

consumer WG1:  GEMMs(j)   softmax(j)  GEMMs(j+1) ...
consumer WG2:  softmax(k) GEMMs(k+1)  softmax(k+1) ...
```

When WG1 reaches low-throughput softmax work, WG2 should have WGMMA work in
flight, and then the roles swap. Named `bar.sync` barriers influence the warp
scheduler's issue order; they do not transfer partial O values between the
warpgroups.

The paper reports a d128/N8192 forward improvement from about 570 TFLOP/s to
620-640 TFLOP/s from this schedule, while warning that the realized timeline
is not as clean as the diagram.

## 7. Forward Deep Dive

### 7.1 Mathematical and ownership contract

For one logical Q tile:

$$
S_{ij}=Q_iK_j^\top,\qquad
\widetilde P_{ij}=\exp(S_{ij}-m_i),
$$

$$
\widetilde O_i\leftarrow
\alpha_i\widetilde O_i+\widetilde P_{ij}V_j,\qquad
\ell_i\leftarrow\alpha_i\ell_i+\operatorname{rowsum}(\widetilde P_{ij}),
$$

where:

$$
m_i^{new}=\max(m_i^{old},\operatorname{rowmax}(S_{ij})),\qquad
\alpha_i=\exp(m_i^{old}-m_i^{new}).
$$

After the last K/V tile:

$$
O_i=\widetilde O_i/\ell_i,\qquad LSE_i=m_i+\log\ell_i.
$$

The FA2 ownership idea remains:

| Output/state | Contributors | Numeric owner | Combine |
| --- | --- | --- | --- |
| one `O_i` row block | all interacting K/V tiles | one logical Q-tile work item | fixed local online update |
| `m_i,l_i` | all interacting score tiles | same work item | fixed local max/sum order |
| final O and LSE | one work item | same work item | disjoint HBM store |

No other Q-tile work item numerically adds into `O_i`.

### 7.2 Representative d128 specialization

For `B=1,H=32,N=8192,D=128`, causal BF16/FP16 in the pinned source:

```text
B_M = 128 Q rows
B_N = 128 K/V rows
D   = 128
CTA = 12 warps = 384 threads
SMEM stages = 2
cluster M = 1 for causal
```

There are:

```text
logical Q/O tiles = 1 * 32 * ceil(8192 / 128) = 2048
K/V tiles seen by a late full-causal Q block = up to 64
```

Noncausal d128 uses `B_N=176` in this early source and may use a two-CTA
cluster when the Q-tile count is even. These are dispatch facts, not universal
FA3 tile constants.

### 7.3 Logical work item is not always one physical CTA lifetime

Persistent CTA scheduling is not a Hopper invention and is not one of FA3's
new numeric ownership rules. It is an implementation pattern that keeps a
fixed population of CTA workers resident for the duration of one kernel
launch. The worker loops over several logical `Q_i -> O_i` work items instead
of exiting after one.

#### 7.3.1 The ordinary-grid counterfactual has no wave barrier

A non-persistent launch does not wait for every CTA in an analytical "wave" to
finish:

```text
SM 0: CTA 0 finishes -> resources released -> pending CTA 132 can be admitted
SM 1: CTA 1 may still be running
SM 2: CTA 2 finishes -> resources released -> pending CTA 133 can be admitted
```

CUDA admits a pending CTA whenever one SM can reserve its complete register,
SMEM, thread/warp, barrier, and cluster resources. If resources permit,
several CTAs may already be resident on one SM. A wave count such as
`2048 / 132 = 15.5` is a concurrency model, not a grid-wide synchronization
protocol.

Consequently, persistence does not uniquely make the scheduler
work-conserving. The normal hardware block scheduler already refills available
CTA slots, and both designs still have an underfilled final tail when fewer
logical tiles remain than available worker slots.

#### 7.3.2 What the pinned FA3 forward source changes

For fixed-length forward, the pinned source selects persistent schedulers:

```text
noncausal:
  gridDim.x = number of SMs
  CTA c handles logical tile ids c, c + gridDim.x, ...

causal:
  gridDim.x = number of SMs
  an integer atomic work queue assigns the next logical Q tile
```

For `B=1,H=32,N=8192,D=128`, there are 2048 logical Q/O tiles but an H100
SXM-class device has about 132 persistent workers:

```text
ordinary conceptual grid:
  2048 CTA identities, each completes one logical tile and exits

pinned persistent grid:
  about 132 CTA identities, each loops over about 15-16 logical tiles
```

This is still one host kernel launch in both cases. It is therefore misleading
to say that persistence removes "one kernel launch per CTA." The more precise
potential savings are:

- fewer CTA exit/admission transitions for the same logical tile count;
- amortized producer/consumer role setup, register redistribution, barrier and
  pipeline initialization, and prologue/tail work;
- the ability to preserve the CTA-level TMA/SMEM pipeline machinery across
  logical work-item boundaries instead of rebuilding it from scratch;
- explicit programmer control over static stride, dynamic queue, tile order,
  and future cost-aware scheduling policy.

The dynamic queue can help with causal tiles whose K/V-loop lengths differ,
but the counterfactual is already a work-conserving hardware scheduler. The
incremental benefit comes from customized ordering and from avoiding a full
CTA/pipeline handoff, not from removing a nonexistent barrier between waves.
Because the task granularity remains one Q tile, persistence also does not
eliminate the fundamental final tail.

#### 7.3.3 Why this pattern fits this FA3 specialization

The d128 path gives its producer threads a low register target and its two
consumer warpgroups a high target. The nominal targets nearly consume H100's
64K-register SM budget for one 12-warp CTA. Exact residency still requires
compiled metadata and an occupancy/profile check, but the design is plausibly
close to one large CTA per SM.

That makes CTA handoff and pipeline rebuild more exposed than in a smaller
kernel that can keep several CTAs resident. Persistence can be more valuable
for short or causal work where fixed setup and imbalance are a larger fraction
of runtime. For long tiles, its contribution may be secondary to TMA/WGMMA
overlap and GEMM/softmax interleaving. The paper does not provide an isolated
persistent-scheduler ablation, so this note does not assign it a standalone
speedup.

Persistence does **not**:

- reduce the mathematical Q/K/V bytes required by each logical tile;
- increase the number of independent logical outputs;
- allow one CTA to migrate across several SMs;
- guarantee more resident CTAs or remove the last underfilled tail;
- change `Q_i -> complete O_i` ownership.

The dynamic scheduler's `atomicAdd` chooses an integer work identity, not a
floating-point contributor order. Different CTA-to-tile assignment does not by
itself make forward numerically nondeterministic because logical O tiles are
disjoint. The pinned mode boundary also matters: variable-length forward uses
the single-tile scheduler, and the pinned backward main path does not use this
forward persistent loop.

### 7.4 CTA role map

The 12-warps forward CTA is:

```text
warpgroup 0 -- producer role, 4 resident warps
  warp 0: elected thread issues Q/K/V TMA operations
  other producer warps: little or no mainloop payload work in this path

warpgroup 1 -- consumer
  WGMMA macrotile for one 64-row Q/O slice
  online softmax and O accumulator

warpgroup 2 -- consumer
  WGMMA macrotile for the other 64-row Q/O slice
  online softmax and O accumulator
```

Warp count is not a physical Tensor Core count. More resident warps provide
independent instruction streams and state; the SM's schedulers issue their
instructions to the available execution pipelines over time.

### 7.5 Forward state residence and lifetime

| State | Residence | Lifetime | Consumer |
| --- | --- | --- | --- |
| Q tile | SMEM | whole logical Q work item | QK WGMMA |
| staged K/V tiles | circular SMEM | one stage generation | QK/PV WGMMA |
| current/next S | distributed FP32 accumulator registers | pipeline iteration | softmax |
| converted P fragment | consumer registers | until corresponding PV issue | PV WGMMA |
| running unnormalized O | distributed FP32 registers | all K/V iterations | online update |
| running max/sum | consumer registers | all K/V iterations | softmax/finalize |
| final O staging | SMEM, reusing V storage in the pinned layout | epilogue | TMA store |
| final O and LSE | HBM | kernel output | backward/model |

The shared-storage union:

```text
smem_v
  overlays
smem_o
```

does not mean V can be overwritten while a WGMMA still reads it. The epilogue
waits for the mainloop's V-stage use to finish before repurposing the physical
SMEM region.

### 7.6 Layer A forward state machine

Producer:

```text
load Q_i once with TMA

for K/V tile j:
    wait until stage j%s is EMPTY
    issue TMA K_j -> smem_k[stage]
    issue TMA V_j -> smem_v[stage]
    transaction barrier eventually marks FULL
```

Consumers:

```text
wait Q_i

for K/V tile j:
    wait K stage FULL
    issue QK_j WGMMA
    wait V stage FULL when PV needs it
    issue P_j V_j WGMMA
    release K/V stage only after last async reader is complete
```

The producer can run ahead until all stages are occupied. Circular buffering
turns memory latency into a capacity/backpressure problem.

### 7.7 Layer B forward: current and next score fragments

A literal single-tile schedule is serial:

```text
QK_j -> wait -> softmax_j -> PV_j -> wait
```

FA3's two-stage consumer pipeline has a prologue, steady state, and epilogue:

```text
prologue:
  compute and wait for S_cur = QK_0
  softmax S_cur -> P_cur

steady iteration j:
  issue S_next = QK_j; commit, do not immediately wait
  issue O += P_cur V_{j-1}; commit, do not immediately wait
  wait until S_next is safe to consume
  softmax S_next -> P_next
  wait before consuming/rescaling the O accumulator
  rotate next -> current

epilogue:
  issue the last P V
  wait, finalize O and LSE, store
```

The extra score accumulator is the key resource cost:

```text
more overlap
  -> current + next FP32 score fragments live together
  -> more registers
  -> pressure against tile size, spills, and occupancy
```

The paper notes that compiler reordering can improve or disrupt the intended
sequence. A source-level `commit` is not a measured overlap percentage; SASS
and Nsight Compute are needed for the realized schedule.

### 7.8 Why forward remains expected deterministic

The async schedule changes physical completion time but preserves the numeric
owner graph:

- each O row has one logical owner;
- consumer warpgroups own disjoint row fragments;
- K/V stages carry read-only inputs;
- named barriers and wait groups protect lifetime, not a racing numeric
  reduction;
- persistent scheduling changes which CTA executes a tile, not how many CTAs
  add into that tile.

Therefore fixed-artifact repeated forward is expected to be deterministic,
subject to:

- identical input and dropout RNG state;
- identical compiled specialization, tile sizes, masks, and scheduler mode;
- correct barrier protocol and no undefined memory behavior;
- empirical confirmation because PTX leaves WGMMA internal accumulation order
  unspecified.

"Unspecified accumulation order" is a portability boundary, not evidence that
one fixed instruction randomly changes order between runs. It means source
alone cannot promise identical bits across compiler versions, instruction
shapes, or architectures.

## 8. Backward Deep Dive

### 8.1 Five-GEMM dataflow

Let `G=dO` and:

$$
D_i=\operatorname{rowsum}(G_i\odot O_i).
$$

For Q tile `i` and K/V tile `j`:

$$
S_{ij}=Q_iK_j^\top,\qquad
P_{ij}=\exp(S_{ij}-LSE_i),
$$

$$
dP_{ij}=G_iV_j^\top,\qquad
dS_{ij}=P_{ij}\odot(dP_{ij}-D_i),
$$

$$
dQ_i^{(j)}=dS_{ij}K_j,
$$

$$
dV_j\mathrel{+}=P_{ij}^\top G_i,\qquad
dK_j\mathrel{+}=dS_{ij}^\top Q_i.
$$

The GEMM-axis ledger is:

| GEMM | Output | Reduction axis | Result ownership |
| --- | --- | --- | --- |
| `Q_i K_j^T` | score rows x key columns | D | current tile |
| `G_i V_j^T` | dP rows x key columns | D | current tile |
| `P_ij^T G_i` | dV key rows x D | Q rows | partial running dV for owner j |
| `dS_ij^T Q_i` | dK key rows x D | Q rows | partial running dK for owner j |
| `dS_ij K_j` | dQ Q rows x D | key rows | partial dQ from owner j |

Backward cannot give all three gradients unique owners with one loop
orientation. FA3 keeps the FA2 K/V-owner orientation because it exposes many
K/V work items and completes `dK/dV` locally.

### 8.2 Three-kernel protocol

For pinned d128:

```text
1. preprocess over Q tiles
     D = rowsum(dO * O)
     transform/store LSE state used by the main kernel
     clear FP32 dQ accumulator and deterministic semaphore

2. main over K/V tiles
     CTA j owns K_j,V_j and final dK_j,dV_j
     loops all interacting Q_i,dO_i tiles
     emits one dQ_i^(j) partial per iteration

3. postprocess over Q tiles
     scale FP32 dQ accumulator
     convert/store BF16/FP16 dQ
```

For `B=1,H=32,N=8192,D=128`, with `B_M=64,B_N=128`:

```text
Q blocks = 8192 / 64  = 128
K blocks = 8192 / 128 = 64

preprocess work items = 128 * 32 = 4096
main K/V work items   =  64 * 32 = 2048
postprocess work items= 128 * 32 = 4096

FP32 dQ accumulator  = 1 * 32 * 8192 * 128 * 4 B = 128 MiB
```

The early backward source uses one main CTA per K/V work item rather than the
forward persistent scheduler.

### 8.3 Backward CTA role map

The d128 main CTA also has 12 warps:

```text
producer warpgroup 0 -- register-deallocated
  warp 0: TMA load K/V once; pipeline Q, dO, LSE, D
  warp 1: dedicated dQ writer
  warps 2-3: no main payload role in this early path

consumer warpgroup 1
  WGMMA and pointwise work for one output partition

consumer warpgroup 2
  WGMMA and pointwise work for the other output partition
```

The consumers keep FP32 `dK_j,dV_j` register accumulators across all Q tiles.
The dQ writer consumes CTA-private SMEM partials and publishes them to the
global FP32 accumulation buffer.

### 8.4 Layer A backward: load versus five-GEMM computation

The producer fixes `K_j,V_j` and pipelines future Q-side inputs:

```text
load K_j,V_j once with TMA

stage 0: consumers process Q_i,dO_i,LSE_i,D_i
stage 1: producer TMA-loads Q_{i+1},dO_{i+1},LSE_{i+1},D_{i+1}
```

The two-stage Q/dO pipelines are protected independently. Consumers release a
stage only after all asynchronous WGMMA readers of its Q/dO contents have
completed.

### 8.5 Layer B backward is constrained by a longer dependency graph

One source-visible steady step is approximately:

```text
issue S  = Q K^T
issue dP = dO V^T
wait enough to form P from S and LSE
wait enough to form dS = P * (dP - D)

issue dV += P^T dO
issue dK += dS^T Q
issue dQ_local = dS K

publish dQ_local to the writer warp
```

The async WGMMA groups allow independent GEMMs and pointwise work to overlap
where register dependencies permit. However, unlike forward Algorithm 2, the
paper does not present one equally simple universal two-stage GEMM-softmax
pipeline for backward. The exact realized overlap among `S`, `dP`, pointwise
`P/dS`, `dK/dV`, and `dQ` is specialization- and compiler-dependent.

The safe conclusion is:

> Backward clearly uses Layer A warp-specialized TMA/compute overlap and
> asynchronous WGMMA issue; its Layer B schedule is more constrained and must
> be verified from the selected source/SASS rather than inherited from the
> forward diagram.

### 8.6 Why a dedicated dQ writer exists

Every K/V CTA contributes to the same Q rows:

```text
CTA j consumer groups:
  compute dQ_i^(j) in registers
  -> write one complete CTA partial to smem_dqacc
  -> signal dQFull

CTA j dQ-writer warp:
  wait dQFull
  -> issue TMA reduce-add from SMEM into global FP32 dq_accum[i]
  -> wait before SMEM reuse
  -> signal dQEmpty
```

This is role specialization around a contested global operation. It lets the
two consumer warpgroups start their next computation while the writer manages
publication, until backpressure makes the shared dQ slot unavailable.

The dedicated writer changes critical-path exposure. It does **not** change the
mathematical contributor graph and therefore does not by itself make dQ
deterministic.

### 8.7 Default versus deterministic dQ

Default:

```text
many independent K/V CTAs
  -> TMA reduce-add partial dQ into the same global FP32 words
  -> arrival/combine order can vary
  -> floating-point association can vary
```

Pinned deterministic path:

```text
for each dQ tile publication:
  wait until dq_semaphore says this K/V contributor is next
  issue and complete TMA reduce-add
  increment semaphore to release the prescribed next contributor
```

Thus:

| Gradient | Normal path | Deterministic path |
| --- | --- | --- |
| dK/dV | one K/V-tile CTA, fixed local Q traversal | same owner principle |
| dQ | many CTA reduce-add arrivals race | same partials, semaphore-ordered publication |

Deterministic mode can be slower because:

- contributors to the same dQ destination are serialized in a prescribed
  order;
- the dQ writer may wait for its turn;
- if the dQ SMEM slot remains full, consumer progress eventually sees
  backpressure;
- semaphore traffic and waiting add overhead.

It need not eliminate all overlap: loading, WGMMA, and publication of
independent tiles can still proceed where dependencies allow.

## 9. FA2 Versus FA3

### 9.1 Forward

| Axis | FA2 SM80 | FA3 SM90 |
| --- | --- | --- |
| logical owner | one Q block owns complete O block | mostly unchanged |
| global-to-shared | thread-issued `cp.async` | TMA descriptor-driven copy |
| matrix instruction | warp-level MMA | async warpgroup WGMMA |
| role map | warps relatively symmetric | producer plus consumer warpgroups |
| register policy | kernel-wide fixed allocation | producer gives registers to consumers |
| K/V buffering | software-pipelined stages | TMA circular SMEM stages |
| softmax/GEMM schedule | more synchronous dependency chain | pingpong plus intra-WG pipeline |
| physical scheduling | ordinary CTA grid in pinned FA2 | persistent CTA schedulers in pinned FA3 |
| numeric ownership | disjoint O rows | still disjoint O rows |

### 9.2 Backward

| Axis | FA2 SM80 | FA3 SM90 |
| --- | --- | --- |
| logical owner | K/V tile owns dK/dV, emits partial dQ | unchanged at high level |
| input movement | `cp.async`-style pipeline | TMA Q/dO pipeline |
| matrix instruction | warp MMA | WGMMA |
| dQ publication | thread/lane atomic adds in legacy path | dedicated writer plus TMA reduce-add |
| deterministic mechanism | later FA2 split workspace/fixed reduction | pinned FA3 semaphore-ordered reduce-add |
| local state | dK/dV accumulators plus temporary layouts | larger WGMMA register fragments plus staged SMEM |

The strongest summary is:

```text
FA2 fixed ownership and unnecessary partial communication.
FA3 keeps that ownership and attacks exposed latency:
  movement issue,
  Tensor Core feed,
  softmax throughput,
  compiler scheduling,
  and contested backward publication.
```

## 10. Determinism Audit

| Boundary | Owner/contributors | Ordering mechanism | Scoped verdict |
| --- | --- | --- | --- |
| forward O/LSE | one logical Q work item | local waits and fixed K/V traversal | expected fixed-artifact repeatability |
| forward persistent scheduler | one assigned CTA per disjoint tile | static stride or atomic work allocation | scheduling atomic is not numeric combine |
| WGMMA internal accumulation | one warpgroup fragment | instruction-defined but ISA order unspecified | pin binary; do not promise cross-artifact bits |
| backward dK/dV | one K/V work item | local Q traversal | expected fixed-artifact repeatability |
| default backward dQ | many K/V CTAs | racing TMA reduce-add arrival | nondeterministic association risk |
| deterministic backward dQ | many K/V CTAs | semaphore-prescribed publication | expected repeatability under supported fixed dispatch |
| dropout | RNG consumers | seed/offset/replay protocol | deterministic only with identical RNG state |

Asynchrony is not synonymous with nondeterminism:

```text
async + disjoint outputs + correct waits
  can be deterministic

sync-looking code + many racing floating-point writers
  can be nondeterministic
```

The proof question remains: **who numerically contributes to the same address,
and what fixes their association order?**

## 11. Performance And Resource Ledgers

### 11.1 Compute

Retained:

- the same exact attention GEMMs and online-softmax arithmetic;
- backward's five mathematical GEMMs.

Changed:

- warp-level MMA becomes WGMMA;
- low-throughput softmax work is hidden behind WGMMA where dependencies allow;
- address and copy instruction work moves toward TMA;
- more score/gradient fragments may coexist for pipeline depth.

### 11.2 Data

Retained:

- no quadratic S/P HBM materialization;
- final forward O/LSE and backward gradients in HBM.

Changed:

- TMA moves tensor tiles directly between HBM and SMEM;
- circular buffers keep several physical generations live;
- forward may reuse V SMEM for O epilogue staging;
- backward uses a global FP32 dQ accumulator and TMA reduce-add.

### 11.3 Concurrency

Gained:

- separate producer and consumer instruction streams;
- two consumer warpgroups;
- persistent forward work distribution;
- independent dQ writer progress.

Counter-pressure:

- 12-warp/384-thread CTA;
- high consumer register counts;
- circular-stage SMEM;
- causal imbalance;
- synchronization and compiler-schedule sensitivity.

### 11.4 Protocol/liveness

Forward requires:

- every acquired K/V stage eventually becomes full or is correctly skipped;
- every consumer eventually releases every used stage;
- Q and O reuse barriers advance phases consistently;
- persistent producer and consumer scheduler views agree on the next tile.

Backward additionally requires:

- Q/dO stage releases;
- dQFull/dQEmpty handoff;
- every deterministic writer eventually obtains the semaphore;
- no CTA exits while another role still depends on its shared state.

## 12. Reusable Kernel-Design Guidance

FA3 adds several lessons beyond FA2:

1. **Architecture upgrades change the balance, not only the peak.** Faster
   matrix engines can make softmax, address generation, synchronization, and
   register movement first-order costs.
2. **Specialize roles when hardware paths are independent.** A producer role
   should issue movement without carrying consumer accumulator pressure.
3. **Async needs a lifetime proof.** Every commit/wait must be paired with the
   buffer or register generation it protects.
4. **Pipeline across iterations to break local dependencies.** A true
   `QK_j -> softmax_j -> PV_j` dependency can coexist with overlap across
   `j-1`, `j`, and `j+1`.
5. **More pipeline stages consume resources.** Track current/next fragments,
   circular SMEM slots, registers, spills, and CTA admission together.
6. **A helper writer hides latency but does not alter contributors.** Re-audit
   many-to-one numeric order separately from execution overlap.
7. **Do not infer physical units from software agents.** Warpgroups, warps,
   and Tensor Cores are not a portable one-to-one map.
8. **Inspect compiler output for schedule claims.** Source order plus async
   APIs defines legal dependencies; SASS/profile establishes realized
   interleaving.

## 13. Smallest Future H100 Validation

### 13.1 Resolve the exact kernel

For every probe, record:

- FlashAttention commit and public entry point;
- architecture (`sm_90a`) and H100 SKU/SM count;
- dtype, head dimension, causal/varlen/GQA/dropout modes;
- `B_M`, `B_N`, stages, warps, cluster size, scheduler;
- compiler, CUDA/CUTLASS versions, binary hash, and dynamic SMEM;
- deterministic flag acceptance and workspace/semaphore allocation.

### 13.2 Verify Layer A

Use Nsight Compute/SASS to check:

- TMA request and barrier activity;
- consumer WGMMA issue while future TMA transactions are in flight;
- stage stalls from producer-full or consumer-empty conditions;
- SMEM footprint, register count, spills, and active CTAs.

### 13.3 Verify Layer B

Check:

- WGMMA commit/wait groups;
- interleaving of HGMMA/WGMMA with `MUFU.EX2`, FP32 arithmetic, shuffles, and
  conversions;
- the pingpong barrier's effect through an ablation if the source exposes one;
- actual Tensor Core active cycles and exposed softmax stalls.

### 13.4 Verify determinism

Repeat at least 100 times and exact-compare/hash:

- forward O and LSE;
- backward dQ, dK, and dV;
- `deterministic=False` versus `True`;
- causal/noncausal, varlen, MHA/GQA, and dropout with restored RNG state.

The first useful localization order is:

```text
forward mismatch:
  O/LSE tile -> resolved specialization -> RNG -> barrier/memory bug

backward mismatch:
  dQ first -> deterministic dispatch/semaphore/reduce-add
  dK/dV next -> local traversal, GQA combine, or memory bug
```

## 14. Stop Line And Open Questions

This pass is sufficient when the reader can explain:

- why Hopper changes the attention bottleneck;
- WMMA/MMA versus WGMMA versus physical Tensor Core;
- why TMA is more than an A100 `cp.async` spelling change;
- producer/consumer register asymmetry;
- the circular-stage generation protocol;
- the two levels of overlap;
- forward persistent logical ownership;
- backward's separate dQ writer and deterministic semaphore;
- why async forward can be deterministic while default backward dQ is not.

Open questions:

- What exact forward and backward specialization does the target training
  Docker dispatch for representative model shapes?
- Does current FA3 preserve the early source's pingpong and intra-WG choices,
  or select newer cooperative/persistent variants?
- What are the net SMEM/register footprints and active-CTA limits for the
  selected d128 and large-head-dimension paths?
- How much of FA3's measured speedup comes from TMA/warp specialization,
  pingpong, intra-WG overlap, persistent scheduling, and tile changes under one
  controlled build?
- Does deterministic backward preserve exact bits across cold/warm runs and
  fresh processes on the pinned H100 Docker?
- Which GQA and varlen paths add another many-to-one combine beyond the dQ
  semaphore described here?
