# FlashAttention-2 Forward: The Delta From FA1

Date: 2026-07-29

Status: FA2 paper plus official v2.0.0 SM80 CUDA forward source study
complete; CPU-only reasoning, with GPU validation deferred and backward kept
in its separate companion audit

Read the
[FA1 one-page checkpoint](flashattention_fa1_checkpoint.md)
first. This note treats FA2 as a delta study: it does not repeat ordinary
attention, safe softmax, CUDA hierarchy, or the full FA1 derivation.

For a graphical reading surface, open the standalone
[FA2 forward delta visual map](flashattention_fa2_forward.html). It keeps the
same pinned source scope but reorganizes the conclusions around why FA2 is
faster than the audited FA1 path: Q-block CTA ownership, on-chip running state,
split-Q warp ownership, SM80 `cp.async`, causal work balance, costs, and the
forward determinism proof. This Markdown file remains the source of truth.

## 1. Scope And Evidence

Pinned evidence:

- paper:
  [FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning, arXiv:2307.08691v1](https://arxiv.org/abs/2307.08691v1);
- initial official FA2 release:
  [tag v2.0.0, commit `4f285b354796fb17df8636485b9a04df3ebbb7dc`](https://github.com/Dao-AILab/flash-attention/tree/4f285b354796fb17df8636485b9a04df3ebbb7dc),
  committed on 2023-07-17 with subject `FlashAttention-2 release`.

The source pass focuses on:

- the CUDA forward path under `csrc/flash_attn/src/`;
- SM80-style FP16/BF16 execution;
- ordinary dense self/cross attention, with causal and varlen mechanics
  included only where they change ownership or traversal;
- the representative no-dropout `d=64`, `B_M=128`, `B_N=128`, four-warp
  specialization before describing dispatch variation.

It does not cover:

- FA2 backward;
- later v2.x source evolution or the current repository `main`;
- FA3's Hopper TMA/WGMMA/warp-specialized implementation;
- inference KV-cache mutation or paged attention;
- a measured A100/H100 profile.

Paper and source claims remain separate. The paper identifies three design
goals: reduce expensive non-matmul work, add sequence parallelism, and improve
warp work partitioning. The pinned source shows one concrete realization,
including `cp.async`, reverse K/V traversal, selected tiles, and barriers.

### Source landmarks

- [`flash_fwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/flash_fwd_launch_template.h):
  one Q-row-block CTA per `blockIdx.x`, tile/warp dispatch, dynamic shared
  memory, and occupancy query;
- [`kernel_traits.h`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/kernel_traits.h):
  warp-along-M `TiledMma`, Q/K/V shared layouts, shared-size formula, and
  SM80 `cp.async` copy atom;
- [`flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/flash_fwd_kernel.h):
  Q-row ownership, reverse K/V loop, online state, QK/PV work, epilogue, O and
  LSE stores;
- [`softmax.h`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/softmax.h):
  per-thread and four-lane-group max/sum reductions;
- [`tests/test_flash_attn.py`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/tests/test_flash_attn.py):
  repeated exact-output race test for a limited d64/FP16/no-dropout matrix.

## 2. Executive Delta

FA2 keeps the mathematical contract:

$$
O=\operatorname{softmax}(QK^\top)V.
$$

The important forward changes are implementation changes:

```text
FA1 v1.0.9:
  one CTA owns several small Q tiles
  K/V tile outer loop -> Q tiles inner loop
  running O/LSE can round-trip through HBM
  warps split K/V columns -> partial O -> fixed shared-memory sum
  ordinary global-load -> register -> shared staging

FA2 v2.0.0:
  one CTA owns one larger Q-row block
  Q-row block outer ownership -> complete K/V loop inside the CTA
  running acc_o/max/sum remain in registers until the epilogue
  warps split Q rows -> disjoint O rows -> concatenate
  SM80 cp.async global -> shared staging
```

The biggest conceptual change is not one isolated trick. Loop order, CTA
granularity, state residence, warp ownership, and copy mechanism change
together.

The performance argument should therefore be read as a causal chain rather
than four independent slogans:

| Change | Immediate physical effect | Expected performance effect | Cost / caveat |
| --- | --- | --- | --- |
| one CTA per Q block | many more sequence-axis workers | more waves and better SM utilization when `B*H` is small | every Q CTA logically reloads its K/V tiles |
| keep one Q block and its running state on chip | remove repeated mutable Q/O/LSE handoffs | less global state traffic and a shorter dependency path | larger register/shared footprint per CTA |
| defer output normalization | remove repeated full-output scale/divide work | fewer expensive FP32 non-matmul instructions | row max, exponential, sum, and old-numerator rescale remain |
| split warp work by Q/output rows | disjoint O ownership | remove shared partial-O stores, barriers, reloads, and FP32 sums | K/V must be visible to all consumer warps |
| SM80 `cp.async` pipeline | direct global-to-shared payload movement with explicit staging | less payload-register pressure and more movement/compute overlap | correctness now depends on fence/wait/barrier placement |

The paper's headline speedup is not evidence that any one row of this table
contributed a fixed percentage. That attribution requires a controlled
ablation or profile.

## 3. FA2 Does Not Stop Producing Or Saving O

Three states are easy to collapse:

| State | Meaning | FA2 treatment |
|---|---|---|
| `acc_o` / $\widetilde O$ | FP32 unnormalized running numerator inside the kernel | retained in registers across K/V tiles |
| final O | normalized forward output consumed by later model layers | divide once in epilogue, convert, and write to HBM |
| LSE | row normalization state needed to reconstruct P in backward | compute from final max/sum and write FP32 to HBM |

For one Q row, after a new score tile:

$$
m_{\rm new}=\max(m_{\rm old},m_{\rm tile}),
$$

$$
\alpha=e^{m_{\rm old}-m_{\rm new}},
$$

$$
\ell_{\rm new}
=\alpha\ell_{\rm old}
+\sum_{j\in\mathrm{tile}}e^{S_j-m_{\rm new}},
$$

$$
\widetilde O_{\rm new}
=\alpha\widetilde O_{\rm old}
+\sum_{j\in\mathrm{tile}}e^{S_j-m_{\rm new}}V_j.
$$

Only after the final K/V tile:

$$
O=\widetilde O/\ell,\qquad
\operatorname{LSE}=m+\log\ell.
$$

The implementation still rescales the old numerator and exponential sum when
the running maximum increases. What it avoids is normalizing the running
output by the new sum on every iteration. In the pinned source,
`softmax_rescale_o` multiplies `scores_sum` and `acc_o` by `alpha`, while the
epilogue performs the only `1 / scores_sum` output scaling.

The final O is still available to backward. It is the semantic forward output,
not a quadratic auxiliary tensor. FA2 avoids saving full P, not final O.

The paper presents saving LSE rather than separate max and sum as a FA2
improvement. The later FA1 v1.0.9 source used in our historical audit already
writes LSE, so this particular paper-level delta is not a clean difference
between the two pinned code releases.

## 4. CTA Ownership: Swap The Loop Nest

### 4.1 Original FA1 paper versus later FA1 source

The original FA1 paper-level implementation is described retrospectively in
the FA2 paper as one CTA per attention head:

```text
paper FA1 grid = batch * heads
```

That CTA keeps a K/V tile outside and walks the Q tiles inside. This is the
baseline behind the FA2 paper's claim that the first FlashAttention version
exposes only batch/head parallelism.

The pinned FA1 v1.0.9 source is already a later implementation evolution. It
adds a configurable query `num_splits=R`:

```text
FA1 v1.0.9 grid = batch * heads * R
```

Split `r` owns disjoint Q tiles `r,r+R,...`, so this is limited query-sequence
parallelism. It is not yet the full FA2 plan: each split CTA still owns several
small Q tiles, keeps K/V outside, and hands running Q/output state through
global memory between K/V iterations.

For `B=1,H=32,N=8192`, an illustrative comparison is:

```text
original FA1 paper:       32 CTAs
FA1 v1.0.9 with R=4:     128 CTAs
FA2 with B_M=128:       2048 CTAs
```

`R=4` is illustrative rather than a universal dispatch result.

### 4.2 FA1 v1.0.9

FA1's historical source launches a configurable number `R` of CTAs per
`(batch, head)`. CTA `r` owns Q tiles `r, r+R, ...`, but uses K/V as its outer
loop:

```text
CTA r:
  for K/V tile J:
      keep K_J and V_J on chip
      for every Q tile I owned by r:
          load Q_I and prior running state
          update O_I / LSE_I
          write running state
```

This reuses one K/V tile across several Q tiles but constrains the CTA count
and sends running output state through HBM between K/V steps.

### 4.3 FA2 v2.0.0

FA2 launches:

```text
grid = (
    ceil(seqlen_q / B_M),
    batch,
    heads
)
```

`blockIdx.x=m_block` therefore owns exactly one
`[m_block*B_M, (m_block+1)*B_M)` Q/O row block:

```text
CTA m:
  load Q_m once from global memory
  initialize acc_o, row_max, row_sum

  for every valid K/V tile J:
      load K_J and V_J
      S_mJ = Q_m K_J^T
      update row_max, row_sum, acc_o

  normalize once
  store final O_m and LSE_m
```

The complete reduction over K/V positions happens inside this one CTA. Its
FP32 `acc_o`, `scores_max`, and `scores_sum` remain in registers until the
epilogue. There is no global running-partial-O buffer in this path.

For `N_q=N_k=256`, `B_M=B_N=128`, non-causal attention:

```text
CTA m=0 owns Q[  0:128] -> O[  0:128]
  visits K/V[128:256], then K/V[0:128]

CTA m=1 owns Q[128:256] -> O[128:256]
  visits K/V[128:256], then K/V[0:128]
```

K/V is reread by the two CTAs, potentially with L2 reuse, but no CTA publishes
a partial result for another CTA to finish.

### 4.4 Why more duplicated K/V can still win

The trade is:

| FA1 K/V-outer CTA | FA2 Q-block-owner CTA |
|---|---|
| reuse a K/V tile across multiple owned Q tiles | different Q CTAs reload the same K/V |
| fewer CTA work units per head | one CTA per Q-row block increases parallelism |
| running O/LSE may move through HBM repeatedly | running state remains on chip and final O/LSE store once |
| small Q tile and repeated state handoff | larger Q tile and more register/shared state |

FA2 deliberately spends some repeated read traffic to obtain more independent
CTAs and eliminate the more expensive running-output round trips and
underutilization. The right comparison is total traffic plus occupancy, not
"which version loads K/V fewer times" in isolation.

The FA2 paper describes sequence-length parallelism as a new contribution,
while the pinned FA1 v1.0.9 source already has a query `num_splits` mechanism.
These claims are compatible because the granularity is different:

```text
FA1 v1.0.9:
  R CTAs per (batch, head)
  each CTA still owns several Q tiles and uses the K/V-outer loop

FA2 v2.0.0:
  one CTA for every Q-row block
  each CTA keeps that row block's complete K/V reduction on chip
```

This is another reason to label paper-to-paper and source-to-source
comparisons separately.

### 4.5 "Reads less" needs an address-space-qualified ledger

FA2 does not simply reduce every input read. It changes which operand or
mutable state is revisited:

```text
paper-style FA1:
  K/V tile reused by one head CTA
  Q and running O/normalization state revisited for later K/V tiles

FA2:
  Q block and running state retained by one CTA
  K/V tiles logically reloaded by every Q-block CTA
```

Ignoring caches, scalar row statistics, tails, and different tile sizes, let
`T` be both the number of Q blocks and K/V blocks. A deliberately simplified
major-traffic sketch is:

```text
paper-style FA1 reads:
  K,V once                    ~ 2 N D
  Q across K/V steps          ~ T N D
  running O across steps      ~ T N D
                                ----------
                              ~ (2T + 2) N D

paper-style FA1 writes:
  running O across steps      ~ T N D

FA2 reads:
  Q once                      ~ N D
  K,V for every Q CTA         ~ 2 T N D
                                ----------
                              ~ (2T + 1) N D

FA2 writes:
  final O once                ~ N D
```

This is intuition, not an exact byte model. It shows why pure read volume may
be similar while total read-plus-write traffic is more clearly improved: FA2
replaces repeated mutable-output round trips with repeated read-only K/V
loads. For FA1 v1.0.9 `R>1`, K/V is also logically reloaded by every split CTA,
so the paper baseline and source baseline differ again.

Finally, a CUDA global-load instruction is not synonymous with a physical HBM
transaction. Repeated K/V global loads from nearby CTAs may be served partly
by L2. A performance claim should therefore keep three quantities separate:

```text
logical global-load/store instructions
L2 hit/miss and cache-sector traffic
physical HBM bytes
```

## 5. Warp Ownership: Split-K Becomes Split-Q

The representative FA2 specialization is:

```text
d = 64
B_M = 128 Q rows
B_N = 128 K/V rows
CTA = 4 warps = 128 threads
```

Its `TiledMma` places the warp layout on the GEMM M dimension:

```text
Layout<Shape<kNWarps, 1, 1>>
```

For a four-warp, 128-row CTA, each warp logically owns 32 Q/output rows,
implemented as two 16-row MMA groups:

```text
warp 0 -> Q/O row groups 0 and 4
warp 1 -> Q/O row groups 1 and 5
warp 2 -> Q/O row groups 2 and 6
warp 3 -> Q/O row groups 3 and 7
```

The groups are interleaved in the physical MMA layout; "32 rows per warp" does
not mean that every lane owns a complete contiguous row.

For every owned row, the same warp/lane group traverses all K/V tiles:

```text
warp w:
  Q_rows_w K_J^T
  softmax for Q_rows_w
  P_rows_w,J V_J
  accumulate complete O_rows_w
```

This changes the combine obligation:

```text
FA1 split-K:
  warp 0..3 each produce partial values for the same O elements
  -> shared-memory partial-O reduction

FA2 split-Q:
  warp 0..3 produce disjoint O rows
  -> concatenate stores, no inter-warp floating-point O reduction
```

Softmax still reduces each score row across its K/V columns. In the source,
that local reduction uses per-thread accumulation plus fixed four-lane
all-reduces inside the warp layout. "No inter-warp O reduction" does not mean
"no reductions anywhere."

The CTA still uses barriers because all warps consume shared Q/K/V tiles and
because the epilogue transits O through shared memory for coalesced stores.
Avoiding an inter-warp numeric sum does not make the warps independent CTAs.

The general reason split-Q is a more natural forward partition is that it
follows complete output ownership. In:

$$
PV:\quad [B_M,B_N][B_N,D]\rightarrow[B_M,D],
$$

`B_N` is a reduction dimension. Splitting it among warps creates overlapping
partial O values. `B_M` is an output dimension. Splitting it gives disjoint O
rows that can be concatenated. This is an owner-computes design, not merely a
different label on the same four warps.

FA2 can use this mapping because its larger Q tile provides enough independent
Q-row MMA work. The audited FA1 `B_r=16` tile had only one natural 16-row M
group, so splitting score columns supplied useful warp parallelism at the
cost of a later partial-O sum.

## 6. Memory Residence And The SM80 Pipeline

For the ordinary trait, shared memory contains distinct Q, K, and V regions:

$$
S_{\rm fwd}
=2(B_MD+2B_ND)
$$

bytes for FP16/BF16. For `B_M=B_N=128`, `D=64`:

```text
Q: 128 * 64 * 2 B = 16 KiB
K: 128 * 64 * 2 B = 16 KiB
V: 128 * 64 * 2 B = 16 KiB
total                         48 KiB
```

The pinned source selects an SM80 `cp.async` copy atom for Q/K/V on
architecture 8.0 or newer. Unlike the audited FA1 v1.0.9 ordinary-load path,
the global-to-shared transfer need not use ordinary payload registers as an
intermediate software staging area.

State residence for the representative path:

| State | Residence and lifetime |
|---|---|
| Q block | global -> shared by `cp.async`; consumed repeatedly by QK |
| current K block | global -> shared by `cp.async`; consumed by QK |
| current V block | global -> shared by `cp.async`; consumed by PV through a transposed shared view |
| score and P fragments | distributed registers |
| `scores_max`, `scores_sum` | FP32 registers for the complete CTA K/V traversal |
| `acc_o` | distributed FP32 accumulator registers for the complete traversal |
| epilogue O | registers -> shared -> global, converted to FP16/BF16 |
| LSE | FP32 register calculation -> global |

The no-dropout d64 loop is approximately:

```text
prologue:
  cp.async Q
  cp.async last K tile

for K/V tile J in reverse order:
  wait until K_J is safe to consume
  start cp.async V_J
  compute Q K_J^T
  wait until V_J is safe to consume
  start cp.async K_(J-1)
  update online max/sum and rescale acc_o
  compute P_J V_J into acc_o

epilogue:
  normalize acc_o once
  store O and LSE
```

`cp_async_fence`, `cp_async_wait`, and CTA barriers define safe shared-buffer
generations. Source comments explicitly note that moving one fence outside
its conditional creates a race. These synchronization operations protect data
availability; they do not introduce another floating-point contributor.

This is still not FA3:

```text
FA2 v2.0.0:
  all CTA warps participate in movement and compute
  cp.async copy + warp-level SM80 MMA

FA3:
  TMA + asynchronous WGMMA
  producer/consumer warpgroup specialization
```

## 7. Dispatch, Causal Work, And Source Boundaries

The tile and warp count are not universal. Examples from v2.0.0 include:

| Compiled head dimension / condition | `B_M x B_N` | Warps |
|---|---:|---:|
| d64, no dropout | `128 x 128` | 4 |
| d64, dropout | `128 x 64` | 4 |
| d128, A100-class branch, no dropout | `128 x 64` | 4 |
| d128, dropout | `128 x 32` | 4 |
| d192, no dropout | `128 x 64` | 8 |
| d256, A100 resource branch | `128 x 64` | 8 |

Architecture properties, head dimension, dropout, causal mode, and available
shared memory participate in dispatch. The representative four-warp d64
explanation must not be copied onto every specialization.

The forward source iterates K/V tiles in reverse order. It starts with the last
valid tile because that is where tail and causal masking are needed, and the
source comments also cite a small register simplification. This order is fixed
for one specialization but differs from an increasing-order implementation,
so it changes floating-point association.

For causal attention:

```text
n_block_max =
  min(total K/V blocks,
      ceil((m_block + 1) * B_M / B_N))
```

Future K/V blocks are therefore not loaded. Early Q-row CTAs perform less work
than later CTAs. With `B_M=B_N=128`, CTA 0 visits only K/V block 0, CTA 1
visits blocks 1 and 0, and so on. The larger number of Q-row CTAs supplies
coarse-grained work for the GPU scheduler, but the triangular workload remains
imbalanced at the CTA level.

Varlen changes base offsets and valid sequence lengths through `BlockInfo`.
It does not change the basic owner: `blockIdx.x` still names a Q-row block, and
that CTA still owns the corresponding final O/LSE region.

## 8. Forward Determinism Audit

Ownership and combine ledger:

| Value | Contributors | Owner/combine | Order |
|---|---|---|---|
| final O row | all valid K/V tiles | one Q-row CTA retains complete `acc_o` | fixed reverse K/V loop |
| one QK score | head-dimension chunks | one dependent MMA accumulator chain | fixed compiled chain |
| row max/sum | K/V columns in current and prior tiles | fixed warp-local tree plus sequential online merge | fixed for one specialization |
| final LSE row | final max and sum | same Q-row CTA | one epilogue |
| distinct Q-row blocks | none across each other | disjoint CTAs and output addresses | CTA completion order irrelevant |

The forward kernel contains no atomic numeric combine for O or LSE. Its
split-Q warp layout also removes FA1's inter-warp partial-O sum. `cp.async`
changes movement and overlap, not output ownership.

The pinned test suite contains a race-condition test that runs the same
forward 200 times and uses `torch.equal` for O and LSE over:

- FP16;
- `d=64`;
- causal and non-causal;
- no dropout;
- sequence lengths 128 through 2048.

This is useful maintainer evidence, not our own GPU validation and not a full
dispatch matrix.

The scoped source verdict is:

> The pinned FA2 v2.0.0 forward has a single CTA owner for every O/LSE row and
> fixed local contributor orders, so fixed-artifact repeatability is expected.
> Dropout additionally requires identical Philox state. Different FA1/FA2
> loop orders, tiles, MMA layouts, or binaries need not produce the same bits.

## 9. FA1 v1.0.9 Versus FA2 v2.0.0

This table is the main reusable result of the delta pass:

| Audit axis | FA1 v1.0.9 forward | FA2 v2.0.0 forward | Why it matters |
|---|---|---|---|
| mathematical output | exact dense attention | same | optimization does not change model semantics |
| online numerator | paper-style normalized running O; historical source stores running state between K/V steps | unnormalized FP32 `acc_o`, normalized once in epilogue | fewer non-matmul normalization operations |
| saved backward state | final O and LSE in the audited source | final O and LSE | FA2 does not eliminate O |
| CTA grid per `(batch, head)` | configurable `R=num_splits` | `ceil(N_q/B_M)` | FA2 exposes more Q-sequence parallelism |
| CTA Q ownership | one CTA owns several interleaved 16-row Q tiles | one CTA owns one 64/128-row Q block | changes work granularity and state footprint |
| loop nest | K/V tile outer, owned Q tiles inner | one Q block fixed, K/V tiles inner | determines which operand/state can remain on chip |
| Q global reads | Q tile revisited on later K/V steps | Q block loaded once per CTA | FA2 trades larger Q state for fewer Q/running-state round trips |
| K/V reuse | one CTA reuses current K/V across several Q tiles | every Q CTA loads its required K/V tiles | more replicated K/V traffic, potentially served by cache |
| running O/max/sum residence | HBM state between K/V steps when revisiting Q tiles | registers for complete K/V traversal | removes global partial-state handoff |
| representative Q tile | 16 rows | 128 rows for d64 | increases register/shared footprint and compute per CTA |
| representative warp split | four warps split K/V columns | four warps split Q rows | reduction-axis split becomes output-axis split |
| current-tile O | four FP32 warp partials for same elements | disjoint warp-owned O rows | fixed shared reduction becomes concatenation |
| Q/K/V global-to-shared copy | ordinary load through registers, then shared store | SM80 `cp.async` | reduces payload-register staging and enables explicit async overlap |
| shared Q/K/V structure | Q buffers plus lifetime-reused K/V region | distinct Q, K, and V regions in ordinary trait | different capacity and pipeline tradeoff |
| K/V traversal | fixed historical order | fixed reverse order | both repeat locally; cross-version bits may differ |
| causal pruning | skip invalid future work while preserving owner | per-Q-block `n_block_max`, reverse boundary-first loop | triangular CTA work remains imbalanced |
| forward partial atomic | none | none | both retain one complete CTA owner per O row |
| fixed-source verdict | expected repeatable with RNG replay | expected repeatable with RNG replay | same verdict, different proof details and arithmetic order |

### What the comparison teaches beyond attention

1. A loop interchange can change global traffic, state lifetime, grid
   parallelism, and numerical association simultaneously.
2. "Split K/V" is incomplete: in QK it may name output score columns, while
   in PV the same axis becomes a reduction axis.
3. Dividing an output axis gives disjoint results; dividing a reduction axis
   creates a combine obligation.
4. A faster kernel may deliberately duplicate read-only input traffic to avoid
   mutable partial-state traffic and expose more workers.
5. Determinism can survive a major performance rewrite when complete-output
   ownership and all local combine orders remain fixed.

### A reusable implementation-design scorecard

For every candidate tile and worker mapping, fill this table before calling it
"more natural" or "faster":

| Question | Preferred direction | Required counter-check |
| --- | --- | --- |
| Which output elements does one warp/CTA own? | complete, disjoint output regions | tails, masking, and whether another stage reuses the same axis as a reduction |
| Is the split axis output or reduction for each fused GEMM? | split output axes when enough independent work exists | splitting reduction may still win if it greatly improves reuse or concurrency |
| Which mutable state crosses a loop iteration? | keep one owner's accumulator on chip | register/shared capacity, spilling, and occupancy |
| Which read-only inputs become duplicated? | duplicate cacheable inputs before spilling mutable partial state | L2/HBM traffic must be measured rather than assumed |
| Which numeric combines remain? | shortest fixed local tree and no unordered cross-CTA writers | atomics, workspace reduction, and deterministic mode |
| Which synchronizations move numeric data versus only protect reuse? | remove cross-worker value exchange first | barriers for copy completion and buffer generations still remain |
| What is the full resource ledger? | compute, bytes, concurrency, and protocol all improve on the critical path | a storage saving is not automatically a traffic or latency saving |
| What validates the performance story? | pinned ablation/profile plus resolved specialization | theoretical peak or source shape alone is insufficient |

This scorecard is also maintained in the cross-project
[`kernel_communication_research.md`](../../../../../kernel_communication_research.md)
workflow so future kernel studies ask the same questions.

## 10. Stop Line And Next Questions

This pass is complete for the pinned FA2 forward. Do not infer the following
without a separate audit:

- FA2 backward ownership, atomics, split workspaces, or deterministic flags;
- identical behavior in later v2.x releases;
- TMA/WGMMA/warp-specialized behavior on Hopper;
- batch invariance or inference KV-cache behavior;
- cross-version bitwise equality.

The pinned v2.0.0 backward source audit now lives in
[FA2 backward](flashattention_fa2_backward.md), with a companion
[backward visual map](flashattention_fa2_backward.html). It is kept separate
because its sequence-K atomic ownership and three-kernel protocol require a
different determinism proof.

The next useful sequence is:

1. review the forward and backward visual maps together;
2. validate both pinned paths on A100 when GPU access is available;
3. add Hopper's TMA/WGMMA/warpgroup foundation before FA3.
