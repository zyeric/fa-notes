# FlashAttention 1 To 4: The Top-Down Evolution

Date: 2026-08-02

Status: synthesis of the pinned FA1–FA4 notes; source-backed where it describes
the audited implementations, with GPU performance validation still pending

## 1. The Stable Contract

Every generation still computes exact attention:

$$
O = \operatorname{softmax}(QK^\top)V.
$$

The central algorithmic invariant is also stable: do not materialize the full
score or probability matrix in HBM. Process score tiles, maintain an online
softmax state, and combine each probability tile with the corresponding value
tile while the useful state is on chip.

The generations differ in how this dataflow is mapped onto the available GPU:

- who owns one complete output or gradient tile;
- which values remain live in registers, shared memory, or TMEM;
- how global-to-shared movement overlaps matrix and non-matrix computation;
- how large a Tensor Core collective the hardware exposes;
- how partial gradients are combined when ownership is inherently many-to-one.

## 2. One Table To Remember

| Generation | Primary problem | Most important response | Human-memory version |
| --- | --- | --- | --- |
| FA1 | materialized attention performs too much HBM IO | IO-aware tiling plus online softmax | **avoid the quadratic HBM intermediate** |
| FA2 | FA1 leaves parallelism and warp efficiency on the table | Q-block CTA ownership, better warp partition, less partial-state communication and non-MMA work | **make ownership natural** |
| FA3 | Hopper Tensor Cores outpace the old synchronous movement/compute schedule | TMA, WGMMA, warp specialization, persistent scheduling, and two-level overlap | **build an asynchronous factory** |
| FA4 | Blackwell matrix compute scales faster than SMEM and exponential throughput | TMEM, fully asynchronous tcgen05, 1-/2-CTA cooperation, deeper pipelines, and softmax algorithm changes | **feed and consume a faster factory** |

This is not four unrelated kernels. It is one mathematical dataflow repeatedly
remapped as the bottleneck and hardware contract change.

## 3. FA1: Change The IO Complexity

Ordinary attention commonly materializes an $N \times N$ score/probability
matrix in HBM. FA1 instead moves Q/K/V tiles through on-chip memory and keeps,
for each query row, a running maximum, normalization sum, and output
accumulator.

The conceptual win is not “softmax is faster.” It is that the score matrix no
longer makes a full HBM round trip. The price is a more complicated tiled
online-softmax update and an implementation whose performance depends on tile
size, memory lifetime reuse, shared-memory layout, and warp cooperation.

The audited historical forward schedule also exposes work-partition costs:
warps own score-column slices and produce partial output accumulators that must
be combined. FA1 therefore gives the vocabulary needed to recognize why FA2's
partition is cleaner.

## 4. FA2: Change Ownership Before Adding New Hardware Tricks

FA2's most reusable lesson is that a mathematically valid tile decomposition is
not automatically a good worker decomposition.

In forward, one CTA owns one Q-row block and traverses all required K/V blocks.
Its warps split Q rows rather than producing several reductions over the same
output rows. This keeps the running output state on chip, removes much of the
partial-O communication, reduces non-matmul scaling/reduction work, and creates
more CTAs when batch times head count alone is too small.

FA2 therefore improves performance without changing the attention equation or
requiring a new Tensor Core generation. Its top-down rule is:

> Partition workers along an output dimension when possible; splitting a
> reduction dimension creates partial ownership and a combine protocol.

Backward cannot erase every partial in the same way. A K/V-tile owner can
produce complete `dK` and `dV`, but it sees only part of each `dQ`. More K/V
owners improve parallelism and make `dQ` a many-writer destination. The
deterministic path must preserve partials and combine them in a fixed order, or
otherwise serialize the writers.

## 5. FA3: Exploit Hopper's Asynchronous Execution Contract

FA3 broadly keeps FA2's logical owner graph. The major change is how one tile
is executed on Hopper.

Hopper provides:

- TMA for descriptor-driven asynchronous tensor movement;
- WGMMA for asynchronous warpgroup matrix multiplication from shared-memory
  descriptors;
- warp specialization and register reallocation so producer and consumer
  warps can hold different responsibilities and register budgets;
- barriers and pipeline stages that make producer/consumer ownership explicit.

FA3 builds two overlap levels:

```text
TMA movement              || main computation
WGMMA Tensor Core work    || CUDA-core softmax / correction
```

Persistent scheduling lets a resident CTA request multiple logical tiles over
its lifetime. It reduces repeated CTA setup and gives the kernel finer control
over work order and tail balance; it does not imply that ordinary CTA launches
have a global wave barrier.

The important distinction is that WGMMA packages a larger cooperative matrix
operation and removes explicit per-warp operand-fragment staging, while the
kernel still owns tile choice, layouts, pipeline stages, barriers, epilogue,
and reduction semantics.

## 6. FA4: Respond To Asymmetric Blackwell Scaling

Blackwell makes matrix multiplication faster without scaling every supporting
resource by the same factor. Shared-memory bandwidth, exponential throughput,
and the CUDA-core work around MMA become more visible in the roofline.

FA4 continues FA3's pipeline-first direction but changes more than the MMA
spelling:

- tcgen05 exposes a fully asynchronous matrix collective;
- TMEM holds large Tensor Core accumulators outside the general register file;
- 2-CTA MMA can distribute a logical operand tile across paired CTAs and reduce
  duplicated shared-memory staging/traffic for the audited shapes;
- deeper lifetime aliasing and role specialization overlap tile generations;
- partial exponential emulation and conditional rescaling reduce the exposed
  softmax cost when exact full-rate exponentiation would bottleneck the
  pipeline.

TMEM, WGMMA, and 2-CTA MMA save different things:

- WGMMA removes explicit per-warp B fragments and their register replication;
- TMEM reduces general-register pressure from accumulators/intermediates;
- 2-CTA tcgen05 can remove duplicated per-CTA B staging and reduce some global
  partial-gradient updates.

None of these statements means that each B element is physically read exactly
once, or that HBM traffic necessarily falls. The memory level and cooperation
scope must be named.

## 7. What Changes In Forward

The durable forward proof shape is:

```text
one logical Q-row output tile
  -> one prescribed owner/cooperative owner group
  -> fixed traversal of contributing K/V tiles
  -> fixed local online-softmax combine
  -> one final O/LSE write
```

FA2 improves that ownership over the audited FA1 schedule. FA3 and FA4 mostly
retain the complete-output owner graph while changing the local movement and
compute pipeline. Fixed-artifact forward repeatability is therefore plausible
when RNG state is replayed, but it does not imply bitwise equality across tile
plans, compiler artifacts, architectures, or implementations.

## 8. What Changes In Backward

The gradient dependency graph is stable:

```text
dV = P^T dO
dP = dO V^T
dS = P * (dP - D)
dQ = dS K
dK = dS^T Q
```

A K/V-tile owner naturally produces complete `dK/dV` but partial `dQ`.
Parallelism across K/V tiles therefore creates a global many-to-one combine.
The concrete mechanism evolves:

- legacy CUDA paths may use contending FP32 atomic adds by default;
- a deterministic legacy path stores split partials and reduces them in a
  fixed split order;
- newer Hopper/Blackwell paths may use bulk reduce-add for the fast path and
  semaphores to impose a prescribed writer order in supported deterministic
  modes.

FA4 2-CTA cooperation can reduce traffic and the number of global `dQ`
updates, but fewer writers are not the same as one writer. Determinism still
requires a fixed combine order.

## 9. The Hardware-Primitive Ladder

```text
Ampere warp-scoped MMA
  -> Hopper warpgroup WGMMA + register accumulators
  -> Blackwell tcgen05 + TMEM accumulators
  -> Blackwell two-CTA/two-SM cooperative tcgen05
```

The ladder expands the operation and reuse scope as Tensor Core throughput
grows. It should not be interpreted as “more participating threads means the
same proportional speedup.” Larger collectives introduce co-scheduling,
barriers, tail behavior, distributed shared-memory traffic, and load-balance
costs.

## 10. Reusable Kernel-Design Guidance

Across the four generations:

1. Start with ownership. Decide whether a split creates disjoint outputs or
   numerical partials before discussing performance.
2. Keep mutable running state with one owner when possible; repeated global
   handoffs are both traffic and ordering boundaries.
3. Separate HBM bytes, SMEM operand traffic, register staging, and accumulator
   residence. “Read once” without a memory level is not actionable.
4. Use larger matrix collectives to amortize staging and issue work, but audit
   the new cooperation and tail costs.
5. Remove exposed non-MMA work as matrix throughput grows; a Tensor-Core-fast
   kernel can be limited by softmax, address calculation, barriers, or SMEM.
6. Treat a deterministic option as a protocol: name the partial buffer,
   semaphore/order, workspace, and supported dispatch.
7. Pin source, shape, dtype, architecture, compiler artifact, and runtime state
   before making a bitwise claim.

## 11. Continue Reading

- Rebuild the full physical model in [FA1 foundations](fa1-foundations.md).
- Study the ownership improvement in [FA2 forward](fa2-forward.md) and
  [FA2 backward](fa2-backward.md).
- Study Hopper's pipeline in [FA3](fa3.md).
- Study asymmetric Blackwell scaling in [FA4](fa4.md).
- Apply the ownership model to the current code in
  [current implementation and determinism](current-implementation-and-determinism.md).
