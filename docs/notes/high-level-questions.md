# FlashAttention High-Level Questions — Draft Source Map

Date: 2026-08-06

Status: first teaching draft; source-backed conceptual answers, with physical
tile constants restricted to representative audited paths. This page is the
source map for the standalone
[high-level questions slides](../slides/high-level-questions.html).

## Scope And Reading Contract

This guide is for a reader who wants a stable mental model before reading the
FA1--FA4 kernel audits. It deliberately mixes three levels, but labels them:

- CUDA programming-model facts from NVIDIA documentation;
- architecture facts from NVIDIA public material;
- FlashAttention algorithm, source, and paper observations from the pinned
  generation notes in this repository.

Generic GPU background remains owned by
[gpu-hardware-notes](https://zyeric.github.io/gpu-hardware-notes/). The short
GPU sections here exist only to make the Attention schedule readable; they do
not replace the longer hardware notes.

The representative teaching path is ordinary training attention, BF16/FP16,
head dimension 128, on A100/H100/B200-class devices. A physical tile or
determinism mechanism is never intended as a universal dispatch statement.

Primary visual sources used by the slides:

- [NVIDIA Hopper architecture in depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/);
- [CUDA Programming Guide: programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html);
- [NVCC compilation trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#the-cuda-compilation-trajectory);
- [FlashAttention](https://arxiv.org/abs/2205.14135),
  [FlashAttention-2](https://arxiv.org/abs/2307.08691),
  [FlashAttention-3](https://arxiv.org/abs/2407.08608), and
  [FlashAttention-4](https://arxiv.org/abs/2603.05451);
- [DASH deterministic-attention scheduling](https://arxiv.org/abs/2601.21824);
- [NVIDIA Rubin architecture overview](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/).

## 1. GPU Hardware Should Be Remembered At Two Scales

At full-chip scale, a GPU contains GPCs, many SMs, a shared L2, memory
controllers, and HBM interfaces. A kernel launch creates a grid of CTAs; the
GPU admits pending CTAs to SMs as resources become available.

At one-SM scale, the useful Attention picture contains:

- resident warp state and registers;
- warp schedulers, dispatch units, and scoreboards;
- CUDA-core, load/store, special-function, and Tensor Core pipelines;
- CTA-private shared memory / L1;
- architecture-specific asynchronous engines such as Hopper TMA;
- on Blackwell, TMEM for Tensor Core operands and accumulators.

The NVIDIA GH100 diagrams are good orientation images, but they are not a
cycle-accurate wiring contract. In particular, a CUDA thread cannot choose an
SMSP, a physical Tensor Core, or a stable CTA-to-SM mapping.

## 2. The Mathematical Baseline: Forward And Backward

For one head, with scale `alpha = 1 / sqrt(d)`:

```text
forward:
  S  = alpha Q K^T
  P  = softmax_row(S)
  O  = P V

backward, given G = dO:
  dV = P^T G
  dP = G V^T
  D  = rowsum(G * O)
  dS = P * (dP - D)
  dQ = alpha dS K
  dK = alpha dS^T Q
```

The implementation does not need to materialize the `N x N` matrices `S`,
`P`, `dP`, or `dS` in HBM. FlashAttention tiles them, forward carries online
softmax state, and backward reconstructs `P` from `Q`, `K`, and saved `LSE`.
The schedule may change; this mathematical contract does not.

## 3. What The Kernel Launch Parameters Mean

A conventional CUDA launch has the conceptual shape:

```cpp
kernel<<<gridDim, blockDim, dynamic_smem_bytes, stream>>>(arguments...);
```

- `gridDim`: number of CTA coordinates requested, not the number of SMs;
- `blockDim`: threads per CTA; all of one ordinary CTA reside on one SM;
- `dynamic_smem_bytes`: extra CTA-private shared-memory allocation;
- `stream`: ordering domain for the launch relative to other work;
- kernel arguments: tensor pointers, sizes, strides, masks, RNG state, and
  dispatch metadata.

`gridDim` and `blockDim` are three-dimensional integer shapes (`dim3`). A
single integer is shorthand for `(x, 1, 1)`; inside the kernel the components
are visible as `.x`, `.y`, and `.z`.

For the representative FA2 forward path with `B=1`, `H=32`, `N_q=8192`,
`B_M=128`, and four warps per CTA:

```text
gridDim  = (ceil(8192 / 128), 1, 32) = (64, 1, 32)
blockDim = (4 * 32, 1, 1)            = (128, 1, 1)
total launched CTAs = 64 * 1 * 32 = 2048
```

`blockIdx.x=m` owns Q/O rows `[128m, 128(m+1))`; `blockIdx.z` selects the
head. This still does not say which SM runs CTA `(m, 0, h)`.

Inside the kernel, `blockIdx`, `threadIdx`, `blockDim`, and `gridDim` identify
logical work. The hardware separately decides when and where each CTA runs.
Registers, shared memory, resident warps, and architectural block limits bound
how many CTAs can coexist on an SM.

### 3.1 Occupancy And CTA Task Assignment

Occupancy is the ratio of active warps on an SM to the architectural maximum
active warps. A closely related question is how many CTAs can be resident at
once. The limit is the minimum imposed by:

- threads and warps per CTA;
- registers per thread/CTA, including allocation granularity;
- static plus dynamic shared memory per CTA;
- architectural resident-CTA and cluster limits;
- architecture-specific resources such as cooperative-cluster placement.

A deliberately simple arithmetic example: suppose an SM can host at most 64
warps, a CTA contains 128 threads = 4 warps, and registers/SMEM admit four such
CTAs. Then 16 warps are resident and theoretical occupancy is `16 / 64 = 25%`.
If an SMEM-heavy specialization admits only one CTA, the same warp count gives
`4 / 64 = 6.25%`. These numbers describe capacity, not how often a pipeline is
busy.

These are admission capacity limits, not utilization measurements. A warp can
be resident but stalled; a one-heavy-CTA-per-SM kernel can still run efficiently
if it exposes enough independent instruction/pipeline work. Conversely, high
occupancy does not repair uncoalesced traffic, shared-memory bank conflicts, an
unbalanced grid, or a serialized reduction chain. NVIDIA's Best Practices Guide
explicitly notes that higher occupancy does not always mean higher performance.

When mapping Attention tasks to CTAs, check:

1. **grid supply:** are there enough independent CTA/work items for all SMs and
   enough waves to absorb tails?
2. **tile reuse versus residency:** a larger tile can reuse more data and
   amortize overhead, but consumes more registers/SMEM and yields fewer CTAs;
3. **owner boundary:** prefer one CTA/cluster completing a final output tile;
   splitting a reduction axis creates partials and a combine obligation;
4. **latency hiding:** decide whether multiple resident CTAs or one deeply
   pipelined heavy CTA provides the needed ready warps/instructions;
5. **load balance:** causal, varlen, sparse, and persistent schedules need tile
   ordering or work stealing so one long owner does not define the tail;
6. **legal communication:** ordinary CTAs cannot assume co-scheduling or direct
   SMEM sharing; cross-CTA dependencies require a cluster/cooperative protocol
   or a global-memory synchronization/combine design;
7. **deterministic readiness:** a prescribed writer rank should align with when
   each CTA can actually publish, or semaphore bubbles will dominate.

Use compiler resource output, the CUDA occupancy API or Nsight Compute
occupancy calculator for the theoretical limit, then use profiler counters to
check achieved active warps and actual stall/utilization behavior.

## 4. What Warp Scheduler And Dispatch Mean

A resident warp has a program counter, active mask, registers, and dependency
state. At an issue opportunity, a warp scheduler selects a warp whose next
instruction is eligible: its operands are ready and the target path can accept
work. The dispatch path sends that issued instruction toward the corresponding
execution pipeline.

The scheduler sees ready warp instructions, not an Attention tile or a Python
loop. The dispatch unit does not itself perform the matrix multiplication.
Tensor Core, CUDA-core, load/store, and MUFU pipelines perform the work, often
over many cycles after issue. Scoreboard dependencies prevent a consumer from
using a result before it is ready, while other warps or independent
instructions can make progress.

The important specialization rule is:

> **A software warp role is not ownership of a physical compute unit.**

An SM is physically partitioned into subpartitions with scheduler/dispatch and
execution resources, but CUDA exposes no stable `warp -> SMSP -> Tensor Core`
binding. Resident warps are time-multiplexed over the execution paths available
to their SM partition. A producer warp issuing TMA control does not own the TMA;
a consumer warpgroup issuing WGMMA does not reserve a Tensor Core; a softmax
warp does not own MUFU or CUDA cores. Warp specialization only constrains which
warps issue which instruction classes and establishes producer/consumer
protocols.

This is why the useful pipeline picture has three separate rows:

```text
software roles:       producer warp | MMA warpgroups | softmax / writer warps
issue arbitration:    ready instruction selected by scheduler / scoreboard
physical execution:   TMA | Tensor Core | CUDA-core / MUFU | load-store paths
```

Overlap comes from independent instructions from different roles being in
flight on different pipelines. It does not require, and should not be explained
as, permanently binding one specialized warp to one engine.

## 5. How Kernel Source Becomes SASS

For a conventional CUDA C++ path:

```text
CUDA C++ / templates
  -> device compilation and PTX
  -> ptxas
  -> architecture-specific cubin containing SASS
  -> fatbinary embedded in a host object or library
  -> runtime selects a compatible image at launch
```

PTX is a virtual ISA. SASS is the machine instruction stream executed by a
specific NVIDIA architecture. `cuobjdump --dump-sass` and `nvdisasm` inspect
the latter. The selected binary also fixes register allocation and static
resource metadata.

The current FA4 CuTeDSL path adds a different front end:

```text
Python/CuTeDSL specialization
  -> MLIR/CUTLASS lowering
  -> PTX tcgen05 / TMA / barrier operations
  -> ptxas machine code
```

It is still a compiled specialized kernel, not an interpreted Python loop.

FA3 Appendix B.2 is a useful real SASS sample: `MUFU.EX2`, `F2FP`, and `FADD`
instructions appear between `HGMMA` instructions. This is evidence that the
compiler produced the intended overlap; source ordering alone would not prove
the final schedule.

A small source-to-artifact mnemonic is:

```text
source intent     PTX-level operation       Hopper SASS family
exp2(x)           ex2.approx.f32            MUFU.EX2
wgmma.mma_async   warpgroup async MMA       HGMMA...
wait group        wgmma.wait_group          WARPGROUP.DEPBAR...
```

FA3 Appendix B.2 shows an `HGMMA` sequence, then `MUFU.EX2`, conversion and
`FADD`, followed by more `HGMMA`. It is a simplified excerpt from the paper,
not a locally measured disassembly.

## 6. Why FA1/FA2 Backward Can Be Nondeterministic

The high-parallelism FA1 backward gives separate K/V-tile CTAs complete
ownership of `dK` and `dV`, but each CTA produces only a partial `dQ`. With
sequence-K parallelism, independent CTAs use FP32 atomic additions into the
same global `dQ` accumulator.

The update is race-safe but its floating-point association order follows CTA
arrival. Because floating-point addition is not associative, repeated launches
can produce different bits.

The FA2 sequence-K main kernel makes the owner boundary concrete:

```text
CTA J fixes K_J,V_J and loops over Q tiles I:
  acc_dV_J += P_IJ^T dO_I
  acc_dK_J += dS_IJ^T Q_I
  atomicAdd(&dq_accum[I, element], dS_IJ K_J[element])

one CTA J -> final dK_J and dV_J
many CTAs J -> partial contributions to the same dQ_I
```

CUDA C++ `old = atomicAdd(address, value)` performs one atomic read-modify-write:
it reads `old`, stores `old + value`, and returns `old`. FA1/FA2 discard the
return value. Atomicity prevents a lost update to that scalar address; it does
not prescribe the order in which independent CTA contributions reach it.

FA1 v1.0.9's deterministic public mode sets `num_splits=1`: one CTA per
batch/head visits K/V tiles in a fixed order. The cost is lost sequence-K CTA
parallelism. If `batch * heads` is already large, the penalty can be small; if
it is not enough to fill the GPU, the cost can be large. No universal FA1
percentage is claimed.

See [FA1 foundations](fa1-foundations.md) and the
[current determinism audit](current-implementation-and-determinism.md).

## 7. What FA2 Changes In Tile Scheduling

The important forward delta is ownership and loop orientation:

```text
FA1 audited source:
  keep K/V tile j on chip
  visit several smaller Q tiles i
  hand partial Q/O state through global memory between j steps

FA2:
  one CTA owns one larger Q_i -> O_i tile
  keep Q and online (m,l,U) state with that owner
  stream all valid K/V tiles j
```

FA2 also changes warp partitioning from split-K-style partial output slices to
split-Q ownership: warps own disjoint query/output rows, avoiding a shared
memory partial-output combine. It may duplicate K/V reads across Q owners, but
gains sequence-parallel CTAs, keeps output state on chip longer, and removes
communication and non-MMA work.

For **forward** `O_tile = P_tile V_tile`:

```text
P_tile [B_M, B_N] x V_tile [B_N, d] -> O_tile [B_M, d]

split B_N: split the reduction dimension -> overlapping partial O -> sum
split B_M: split an output dimension    -> disjoint O row slices -> concatenate
```

This is the mathematical difference behind the FA1/FA2 warp-partition figures.
It should not be confused with the **backward CTA orientation**: the high-
parallelism backward path fixes a K/V tile so it can own final `dK_J,dV_J`, and
therefore emits only a partial `dQ_I`.

See [FA2 forward](fa2-forward.md).

## 8. A100, Hopper, And Blackwell Change The Kernel Contract

| Product / architecture | Tensor Core interface | Movement / residence feature | Attention consequence |
| --- | --- | --- | --- |
| A100 / Ampere | 3rd-generation Tensor Cores, warp `mma.sync` | `cp.async` global-to-shared staging | software pipelines inside a CTA; register accumulators |
| H100 / Hopper | 4th-generation Tensor Cores, warpgroup `wgmma` | TMA, transaction barriers, clusters / DSMEM | warp specialization and two-level load/GEMM/softmax overlap |
| B200 / Blackwell | 5th-generation Tensor Cores, fully asynchronous `tcgen05` | 256 KiB TMEM per SM and 1-/2-CTA MMA modes | larger tiles, accumulator handoff through TMEM, deeper role separation |

Instruction name, physical Tensor Core generation, result residence, and
programming abstraction are different concepts. TMA moves tensors; WGMMA or
tcgen05 requests matrix work; Tensor Cores execute it; registers/TMEM hold
results depending on the generation.

## 9. How To Interpret A Cycle And FA4's Roofline

A clock cycle is one period of the SM clock. It is not synonymous with one
source line, one instruction latency, or one completed matrix multiply.

Keep four quantities separate:

- latency: cycles until one result becomes usable;
- initiation interval / throughput: how often a pipeline accepts or completes
  independent work;
- aggregate per-SM throughput: work per cycle across the relevant pipelines;
- wall time: cycles divided by clock rate, plus effects omitted by the model.

FA4 constructs a simplified throughput roofline by counting required work or
traffic for one tile and dividing by the measured or specified per-SM rate:

```text
cycles(resource) ~= work(resource) / peak_throughput(resource)
tile lower bound  ~= max(MMA cycles, SMEM cycles, EX2 cycles)
```

For forward `M=N=d=128`, the FA4 paper estimates 1024 MMA cycles, 768 SMEM-read
cycles, and 1024 exponential cycles. For backward it estimates 2560 MMA cycles
and 3328 total SMEM cycles in the 1-CTA path. These are paper models, not an
Nsight measurement and not a sum when the paths overlap perfectly.

## 10. Why FA3 Pipelines, And Why FA3 Is Not Enough On Blackwell

FA3 responds to Hopper asynchrony in a specific order:

1. **producer/consumer pipeline:** a producer warpgroup uses TMA while
   consumer warpgroups compute;
2. **inter-warpgroup ping-pong:** two consumer warpgroups alternate; when WG1
   executes low-throughput softmax, WG2 has WGMMA work in flight, then roles
   swap. Named barriers influence issue order but do not bind either warpgroup
   to a Tensor Core;
3. **intra-warpgroup inter-iteration pipeline:** one consumer keeps current and
   next score fragments, overlapping `PV(j-1)` with `QK(j)` and softmax work.

Its two-stage pipeline keeps an extra score fragment in registers. This pays
register capacity to hide non-MMA work.

Blackwell doubles the analyzed BF16 Tensor Core throughput per SM while SMEM
read and exponential throughput do not scale in the same way. The FA3
register-resident schedule therefore exposes SMEM and softmax more strongly.
FA4 rebuilds the handoff around fully asynchronous MMA and TMEM, uses two
softmax groups plus a correction group, and changes the algorithm with partial
software exponential and conditional rescaling. It is not a simple WGMMA to
tcgen05 spelling change.

## 11. Representative FA2--FA4 Tile Specs

Physical tile constants are specialization-specific. For the pinned ordinary
d128 teaching paths:

| Generation | Representative score tile | Logical owner | Physical schedule |
| --- | --- | --- | --- |
| FA2 SM80 | `128 x 64` | one CTA owns one 128-row Q/O tile | four split-Q warps scan K/V tiles |
| FA3 SM90 causal | `128 x 128` | one logical Q/O tile; persistent CTA may execute multiple work items | TMA producer + two 64-row WGMMA consumer groups |
| FA4 SM100 family | `128 x 128` mathematical subtile | one owner per final row | commonly two independent Q streams; eligible paths may add a 2-CTA physical group |

Do not collapse mathematical subtile, logical work item, launched CTA, CTA
cluster, or persistent CTA lifetime into one meaning of “tile.” The canonical
seven-field description is maintained in [tile spec](tile-spec.md).

## 12. Why Backward Needs More On-Chip State

Forward has two GEMMs per attention cell: `QK^T` and `PV`. Backward recomputes
the score/probability path and performs five GEMMs: `S`, `dP`, `dV`, `dK`, and
`dQ`, plus the elementwise `P` and `dS` transforms.

Backward must keep or stage more simultaneous operands and intermediates, keep
`dK/dV` accumulators across the Q loop, and publish partial `dQ`. This makes
lifetime planning and layout conversion harder even when a later version uses
less total SMEM in one particular specialization.

Representative audited d128 FA1/FA2 allocations are 152 KiB and 144 KiB. FA2
removes a 64-KiB eight-partial `dQ` buffer but grows Q/dO and P/dS staging; the
net decrease is only 8 KiB. On FA4, TMEM stores several accumulator lifetimes,
yet the paper still models backward as SMEM-bound because eight of ten GEMM
operand paths plus `dS` and partial-`dQ` traffic use SMEM.

See [FA2 backward](fa2-backward.md) and [FA4](fa4.md).

## 13. How Deterministic Backward Evolves, And What DASH Adds

| Boundary | Fast combine | Deterministic mechanism | Main price |
| --- | --- | --- | --- |
| FA1 v1.0.9 | scalar FP32 atomics from sequence-K CTAs | single serial K/V owner | lost sequence parallelism |
| initial FA2 v2.0.0 | scalar FP32 atomics | no public deterministic switch | not selectable in that release |
| later legacy FA2 | one shared FP32 `dQ` accumulator | private split images + fixed split-index reduction | global workspace and traffic |
| early FA3 | TMA bulk reduce-add | semaphore-prescribed writers | waits and pipeline backpressure |
| FA4 SM100 family | async bulk FP32 reduce-add | acquire, reduce, completion, release in a fixed order | waits, fences, head-of-line risk |

The semaphore fixes floating-point association order; it does not automatically
choose a good readiness order. DASH models deterministic backward as a DAG
scheduling problem. Its descending-Q and shift schedules align compute
readiness with ordered publication and shorten the critical path. On H800, the
paper reports up to 37.9% throughput loss for its FA3-derived deterministic
baseline and up to 1.28x recovery from improved scheduling. These are DASH
paper measurements, not universal FA3 numbers.

## 14. Rubin: Opportunities And Challenges For Attention

Public Rubin material announces 22 TB/s HBM4, faster `EX2`, doubled Tensor Core
K-dimension processing per clock, enhanced TMA descriptor handling,
fine-grained dependent-kernel triggering, counted writes, and faster scale-up
fabric.

Potential Attention consequences:

- re-search native versus software-assisted exponential;
- re-search tile sizes, pipeline depth, and the K-loop;
- lower address/descriptor overhead for paged or segmented K/V movement;
- make split-stage, decode, or distributed tile dependencies more composable;
- improve KV-cache and distributed-attention traffic with HBM4 and NVLink 6.

Challenges and non-claims:

- faster HBM is not a proportional training-attention speedup because FA
  already avoids the quadratic HBM intermediate;
- faster `EX2` does not remove max/sum, rescaling, conversion, and handoff;
- counted completion does not prescribe floating-point reduction order;
- activation compression can change dense-softmax semantics;
- no public Rubin FlashAttention artifact, SASS, profile, or complete on-chip
  resource contract is pinned here.

See the explicitly scoped [Rubin projection](rubin-attention-projection.md).

## 15. What A Larger Head Dimension Changes

Separate the two widths first:

```text
S = Q K^T    -> d_qk is the reduction axis
O = P V      -> d_v  is the output-column axis
```

For a fixed `B_M x B_N` score tile:

```text
QK work       ~= 2 * B_M * B_N * d_qk
PV work       ~= 2 * B_M * B_N * d_v
softmax work  ~= O(B_M * B_N)
```

If both widths grow from 128 to 512, matrix work grows roughly fourfold while
the number of scores consumed by softmax is unchanged. This creates an
opportunity to amortize scheduling and pointwise softmax work and to sustain
Tensor Core pipelines more easily. MQA/GQA can add a stronger opportunity by
reusing one K/V tile across several query heads.

The same change creates sharper resource pressure:

- Q/K/V movement and total FLOPs grow;
- a wider output accumulator consumes more registers or TMEM;
- larger operand stages consume SMEM and can reduce CTA residency;
- spills and tile-boundary effects become performance cliffs;
- publishing a probability tile to multiple PV consumers adds SMEM traffic
  and synchronization;
- decode may remain KV-cache-bandwidth-bound despite more compute per score.

`d_qk` and `d_v` suggest different schedule responses. A larger `d_qk` creates
more reduction steps for QK. A larger `d_v` creates independent output columns;
splitting those columns gives disjoint outputs and avoids adding replicated
complete-width partial `O` tensors. One public SM90 large-`d_v` design direction
therefore computes QK/softmax once, publishes `P`, and gives separate PV
consumers disjoint output-column ranges.

The hardware-friendly summary is not “larger head dimension is good.” It is:

> More regular Tensor Core work is welcome only when operand reuse, accumulator
> residence, output ownership, and movement keep the compute pipeline fed.

Current hardware prefers Attention that:

- exposes large regular matrix tiles and enough independent owners;
- reuses K/V or P without multi-writer numeric partials;
- keeps transient score/probability state on chip;
- overlaps movement, MMA, and softmax with explicit dependency handoffs;
- avoids wide live accumulators, spills, irregular masks, tiny ragged tiles,
  and unordered global reductions.

This preference is conditional. Training forward, training backward, prefill,
and decode can stress different resources. See the narrow
[large-head-dimension inference note](large-head-dimension-inference.md).

## 16. What Makes A Kernel Hard, How To Analyze It, And How To Debug It

FA1--FA4 suggest that kernel difficulty is rarely one difficult formula. The
hard part is satisfying several contracts at once:

1. **semantic contract:** exact output, masking, RNG, precision, and backward;
2. **ownership contract:** which worker completes each output and which values
   are still partial;
3. **residence/lifetime contract:** where every live value sits, when storage
   can be reused, and what must cross a CTA boundary;
4. **schedule contract:** dependency order, overlap, synchronization, and
   liveness under imbalance;
5. **resource contract:** registers, SMEM, TMEM, warps, occupancy, bandwidth,
   instruction throughput, and launch overhead;
6. **artifact contract:** dispatch, compiler version, generated SASS, and the
   exact GPU architecture.

The reusable analysis loop is:

```text
1. derive the tile-local math and reduction axes
2. assign final-output owners; list every multi-writer edge
3. write a residence + lifetime ledger
4. count work and bytes per tile for each physical resource
5. divide by per-SM throughput to predict the exposed bottleneck
6. propose overlap only across truly independent dependencies
7. compile and inspect resource usage + SASS
8. profile the selected kernel and compare counters to the model
9. run correctness, race, boundary, and repeated-bit tests
10. change one variable and repeat
```

Useful tools, each answering a different question:

| Question | Tool / method |
| --- | --- |
| Which kernel actually launched, and where are CPU/launch/memcpy gaps? | [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) |
| Is the kernel compute-, memory-, latency-, or occupancy-limited? Which pipelines stall? | [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html), roofline, Scheduler Stats, Warp State, Memory Workload |
| What did the compiler really emit? What are registers and static resources? | `ptxas -v`, [cuobjdump / nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) |
| Are addresses, initialization, shared-memory accesses, or barriers invalid? | [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html): `memcheck`, `initcheck`, `racecheck`, `synccheck` |
| Is the result mathematically and numerically correct? | small CPU/PyTorch reference, adversarial masks/tails, FP64 oracle where useful, finite-difference or autograd checks |
| Is repeated execution bitwise stable? | fixed artifact/RNG/stream, repeated hashes or `torch.equal`, then perturb launch order to expose multi-writer assumptions |
| Where does a deadlock or first bad tile occur? | tiny shapes, debug barriers/counters, assertions, selective device `printf`, `cuda-gdb`; remove instrumentation before profiling |

No one profiler view proves the schedule. Source establishes intended
dependencies, SASS establishes the compiled instruction order, counters and
timelines establish realized behavior, and correctness/replay tests establish
observable results. All four are needed for a strong kernel claim.

## Memorization Card

```text
GPU:  grid creates CTAs; SMs host CTAs; schedulers issue ready warps;
      execution pipelines do the work.

Compile: source -> PTX -> ptxas -> SASS; launch selects a specialization;
         hardware executes SASS, not CUDA C++ or Python.

FA1: tile the attention plane.
FA2: make output ownership natural.
FA3: pipeline the Hopper owner.
FA4: decouple and deepen for asymmetric Blackwell scaling.

Backward: five GEMMs + multi-writer dQ.
Determinism: fix the combine order; scheduling decides how expensive that is.
Rubin: re-measure the balance before naming the next algorithm.

Large d: more matrix work per score, but wider operands and accumulators.

Kernel method: owner -> lifetime -> resource ledger -> SASS/profile -> tests.
```
