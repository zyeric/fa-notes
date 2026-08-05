# FlashAttention 1 To 4: A Tile And Computation Mental Model

Date: 2026-08-05

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

This note deliberately separates three things that are often collapsed in a
verbal explanation:

1. **mathematical tile:** a rectangle such as $Q_i$, $K_j$, or $S_{ij}$;
2. **logical owner:** the work item responsible for a complete output tile;
3. **physical schedule:** CTA/cluster shape, warp roles, memory residence, and
   pipeline.

FA1 to FA4 changes all three at different times. A larger tile does not by
itself mean a different owner graph, and a persistent CTA is not a larger
mathematical tile.

## 2. Draw This Tile Grid First

Before naming a FlashAttention generation, draw the attention plane:

```text
                         K/V sequence: j ->
                    K0/V0      K1/V1      K2/V2
                  +----------+----------+----------+
Q sequence:  Q0   |   S00    |   S01    |   S02    |  -> O0
             Q1   |   S10    |   S11    |   S12    |  -> O1
             Q2   |   S20    |   S21    |   S22    |  -> O2
                  +----------+----------+----------+
                         one cell = Sij = Qi Kj^T
```

Use the same symbols for every generation:

| Tile | Logical shape | Meaning |
| --- | --- | --- |
| $Q_i$ | $B_M\times D$ | one block of query rows |
| $K_j,V_j$ | $B_N\times D$ | one block of key/value rows |
| $S_{ij},P_{ij}$ | $B_M\times B_N$ | one transient score/probability cell |
| $O_i$ | $B_M\times D$ | output rows corresponding to $Q_i$ |

There are three axes to track:

- $D$ is the reduction axis of $Q_iK_j^\top$;
- $j$ / the K/V sequence is the reduction axis of the complete forward
  output $O_i$;
- $i$ / the Q sequence is the reduction axis of complete backward $dK_j$ and
  $dV_j$.

That last distinction predicts the natural forward and backward loop
orientations before any CUDA detail appears.

### 2.1 Forward: fix a row tile, scan cells horizontally

For one fixed $Q_i$, the complete forward result needs every valid $j$:

```text
fix i / own O_i
  initialize row state (m_i, l_i, U_i)

  for each valid K/V tile j:
      S_ij = Q_i K_j^T
      update row max m_i
      P_tilde_ij = exp(S_ij - m_i_new)
      rescale old l_i and U_i
      l_i += rowsum(P_tilde_ij)
      U_i += P_tilde_ij V_j

  O_i   = U_i / l_i
  LSE_i = m_i + log(l_i)
```

The state $(m_i,l_i,U_i)$ summarizes every cell already visited in that row.
The full $S$ or $P$ plane never needs to exist in HBM. This is the common
algorithm underneath FA1, FA2, FA3, and FA4 forward.

The strongest default ownership rule is therefore:

> Split workers along $i$ when possible, and let one owner keep the complete
> $j$ reduction for $O_i$.

### 2.2 Backward: fix a column tile, scan cells vertically

Let $G_i=dO_i$ and $D_i=\operatorname{rowsum}(G_i\odot O_i)$. Backward
revisits one attention cell and performs five tiled matrix products:

```text
S_ij  = Q_i K_j^T                 # recompute score
dP_ij = G_i V_j^T
P_ij  = exp(S_ij - LSE_i)
dS_ij = P_ij * (dP_ij - D_i)

dV_j += P_ij^T G_i
dK_j += dS_ij^T Q_i
dQ_i_from_j = dS_ij K_j
```

The natural high-parallelism work item fixes $j$ and scans $i$:

```text
fix j / own dK_j and dV_j
  for each interacting Q tile i:
      recompute P_ij and dS_ij
      accumulate complete local dK_j and dV_j
      publish one partial dQ_i_from_j
```

This orientation makes `dK_j` and `dV_j` single-owner outputs, but every
`dQ_i` receives one contribution from every interacting $j$ owner. Swapping
the loop would make `dQ` complete and `dK/dV` partial. One orientation cannot
make all three gradients single-owner.

The backward memory rule is:

> **Fix $j$, keep $dK_j/dV_j$; `dQ` is the unavoidable cross-$j$ combine.**

This contributor graph remains recognizable from FA1 through FA4. Atomic
add, bulk reduce-add, a partial workspace, and an ordered semaphore are
different physical protocols for the same many-to-one edge.

## 3. The Four Forward Tile Schedules

### 3.1 One comparison frame

The table uses representative pinned paths, not universal constants. Dispatch
can change with head dimension, dtype, dropout, mask, sequence length,
architecture, and source revision.

| Generation / pinned path | Representative score tile | Logical output owner | High-level loop and live state | Physical idea to remember |
| --- | --- | --- | --- | --- |
| FA1 v1.0.9 CUDA, long-sequence d128 | $16\times128$ | CTA split $r$ owns several interleaved 16-row $Q_i$ tiles | $j$ outer; visit all $i=r,r+R,\ldots$ inside; each $O_i$/LSE state is handed through HBM between $j$ steps | keep one K/V tile, revisit many small Q tiles; four warps split score columns / the PV reduction |
| FA2 v2.0.0 SM80, d128 no dropout | $128\times64$ | one CTA owns one 128-row $Q_i\rightarrow O_i$ block | fix $i$; scan valid $j$ in reverse; keep running $m,l,U$ in registers until epilogue | split-Q: four warps own disjoint Q-row/output slices |
| FA3 pinned early SM90, causal d128 | $128\times128$ | one logical $Q_i\rightarrow O_i$ work item; a persistent CTA may execute several such items over its lifetime | same fix-$i$/scan-$j$ online update as FA2 | TMA stages K/V while two consumer warpgroups use WGMMA and softmax; each owns 64 output rows |
| FA4 pinned SM100 d128 family | $128\times128$ mathematical subtiles, commonly two Q stages in the work item | each final row still has one logical owner; eligible paths may use a 2-CTA cooperative group | two independent 128-row streams scan the K/V tiles and keep online state | tcgen05 writes S/P/O through TMEM; separate MMA, softmax, and correction roles deepen the handoff pipeline |

Two cautions make the table safe to teach:

- FA2's often-drawn $128\times128$ example is the representative d64 path;
  the audited d128/no-dropout A100-class path uses $128\times64$.
- In FA4, “two 128-row Q stages” and “2-CTA” are independent concepts. A
  mathematical 128-row subtile, one logical work item, one CTA, and one CTA
  cluster are not interchangeable nouns.

### 3.2 The loop-nest memory trick

```text
FA1 audited source:
  for K/V tile j:                    # keep K_j,V_j on chip
      for several small Q tiles i:   # load/update/store state_i

FA2:
  fix one larger Q tile i            # keep Q_i and state_i on chip
      for K/V tile j:                # stream K_j,V_j

FA3:
  same logical loop as FA2
  + asynchronous movement/MMA/softmax factory

FA4:
  same broad owner graph
  + two buffered Q streams, TMEM handoffs, fully-async MMA
  + optional 2-CTA cooperation for eligible physical tiles
```

The shortest correct mnemonic is:

```text
FA1: tile the attention plane
FA2: turn the loop so output state stays with its owner
FA3: pipeline the owner
FA4: decouple and deepen the owner pipeline
```

FA1's paper pseudocode and its audited v1.0.9 CUDA schedule are not identical.
The comparison above intentionally describes the source schedule; the paper's
abstract K/V-outer loop is the origin of the tiled online-softmax algorithm,
not evidence that every later kernel uses the same CTA mapping.

### 3.3 A five-minute teaching route

1. Draw the $(i,j)$ attention grid and say “one cell is transient
   $S_{ij}/P_{ij}$.”
2. Move horizontally across one row: that is forward, and $(m,l,U)$ is the
   row's compressed history.
3. Move vertically down one column: that is the natural backward K/V owner;
   it finishes `dK/dV` but emits partial `dQ`.
4. Explain FA1 to FA2 as the important ownership/loop-nest change.
5. Keep the same owner graph and explain FA3/FA4 as progressively more
   asynchronous physical factories, not new attention equations.
6. End with the audit question: “which output is complete here, and which is
   only a partial?”

If the audience remembers only one drawing, it should be the grid. Forward is
row ownership; backward's high-parallelism path is column ownership; FA2 fixes
the forward mapping; FA3 and FA4 mainly redesign how each owner runs.

## 4. One Table To Remember

| Generation | Primary problem | Most important response | Human-memory version |
| --- | --- | --- | --- |
| FA1 | materialized attention performs too much HBM IO | IO-aware tiling plus online softmax | **avoid the quadratic HBM intermediate** |
| FA2 | FA1 leaves parallelism and warp efficiency on the table | Q-block CTA ownership, better warp partition, less partial-state communication and non-MMA work | **make ownership natural** |
| FA3 | Hopper Tensor Cores outpace the old synchronous movement/compute schedule | TMA, WGMMA, warp specialization, persistent scheduling, and two-level overlap | **build an asynchronous factory** |
| FA4 | Blackwell matrix compute scales faster than SMEM and exponential throughput | TMEM, fully asynchronous tcgen05, 1-/2-CTA cooperation, deeper pipelines, and softmax algorithm changes | **feed and consume a faster factory** |

This is not four unrelated kernels. It is one mathematical dataflow repeatedly
remapped as the bottleneck and hardware contract change.

## 5. FA1: Change The IO Complexity

Ordinary attention commonly materializes an $N \times N$ score/probability
matrix in HBM. FA1 instead moves Q/K/V tiles through on-chip memory and keeps,
for each query row, a running maximum, normalization sum, and output
accumulator.

The conceptual win is not “softmax is faster.” It is that the score matrix no
longer makes a full HBM round trip. The price is a more complicated tiled
online-softmax update and an implementation whose performance depends on tile
size, memory lifetime reuse, shared-memory layout, and warp cooperation.

In the audited v1.0.9 forward, a CTA keeps one K/V tile on chip while walking
several small Q tiles. It can reuse K/V within that CTA, but the running state
for all of those Q tiles cannot stay resident simultaneously, so partial O and
normalization state are handed through global buffers across K/V steps.

Its warps own score-column slices and produce partial output accumulators that
must be combined inside the CTA. FA1 therefore gives the vocabulary needed to
recognize why FA2's partition is cleaner.

## 6. FA2: Change Ownership Before Adding New Hardware Tricks

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

## 7. FA3: Exploit Hopper's Asynchronous Execution Contract

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

## 8. FA4: Respond To Asymmetric Blackwell Scaling

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

## 9. What Changes In Forward

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

## 10. What Changes In Backward

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

## 11. The Hardware-Primitive Ladder

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

## 12. Reusable Kernel-Design Guidance

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

## 13. Continue Reading

- Rebuild the full physical model in [FA1 foundations](fa1-foundations.md).
- Study the ownership improvement in [FA2 forward](fa2-forward.md) and
  [FA2 backward](fa2-backward.md).
- Study Hopper's pipeline in [FA3](fa3.md).
- Study asymmetric Blackwell scaling in [FA4](fa4.md).
- Apply the ownership model to the current code in
  [current implementation and determinism](current-implementation-and-determinism.md).
