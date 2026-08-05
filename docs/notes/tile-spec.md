# FlashAttention 1 To 4: High-Level Tile Spec

Date: 2026-08-05

Status: compact teaching specification derived from the pinned FA1--FA4 notes;
logical contracts are stable within the stated scope, while physical tile
constants describe representative audited paths rather than universal
dispatch

## 1. What This Spec Is

This is a **descriptive semantic spec** for remembering and explaining tiled
FlashAttention. It is not a CUDA ABI, a launch configuration, or a promise
that every dtype, shape, mask, and source revision selects the same numeric
tile size.

Read it in three layers:

```text
L0 -- mathematical tile
      What rectangle is being computed?

L1 -- ownership and reduction
      Who owns the complete output, and which axis is scanned?

L2 -- physical lowering
      Which representative CTA/cluster, tile size, memory, and MMA pipeline
      realize L0 and L1 in one pinned implementation?
```

The layers must remain separate. In particular:

```text
mathematical subtile != logical work item != launched CTA != CTA cluster
```

## 2. Canonical Tile Vocabulary

For one batch/head plane:

| Name | Shape | Role |
| --- | --- | --- |
| `Q_i` | $B_M\times D$ | query-row tile `i` |
| `K_j`, `V_j` | $B_N\times D$ | key/value-row tile `j` |
| `S_ij`, `P_ij` | $B_M\times B_N$ | transient attention cell |
| `O_i` | $B_M\times D$ | final output corresponding to `Q_i` |
| `m_i`, `l_i` | $B_M$ | online-softmax row maximum and sum |
| `U_i` | $B_M\times D$ | unnormalized running output numerator |

The attention plane is:

```text
                         K/V tile j ->
                  +---------+---------+---------+
Q tile i      Q0  |   S00   |   S01   |   S02   | -> O0
              Q1  |   S10   |   S11   |   S12   | -> O1
              Q2  |   S20   |   S21   |   S22   | -> O2
                  +---------+---------+---------+
```

The three important reduction axes are:

| Result | Reduction axis |
| --- | --- |
| one `S_ij = Q_i K_j^T` cell | head dimension `D` |
| complete forward `O_i` | all interacting K/V tiles `j` |
| complete backward `dK_j`, `dV_j` | all interacting Q tiles `i` |

## 3. L1 Semantic Spec

### 3.1 Forward

```yaml
work_item:       one logical Q_i -> O_i owner
fixed_axis:      i
scanned_axis:    all valid j
transient_cell:  S_ij, P_ij
persistent:      m_i, l_i, U_i
final_output:    O_i = U_i / l_i, plus LSE_i
global_combine:  none for ordinary non-SplitKV forward
```

Per K/V cell:

```text
S_ij = Q_i K_j^T
m_new = max(m_old, rowmax(S_ij))
P_tilde_ij = exp(S_ij - m_new)
l_new = exp(m_old - m_new) l_old + rowsum(P_tilde_ij)
U_new = exp(m_old - m_new) U_old + P_tilde_ij V_j
```

Memory sentence:

> **Fix `i`, scan `j`, keep `(m,l,U)` with the logical `O_i` owner.**

FA1's audited historical schedule implements this logical ownership with one
CTA split responsible for several small Q tiles and a global state handoff;
FA2 makes the logical-to-physical mapping much more direct. FA3 and FA4 retain
the broad owner graph and change its physical pipeline.

### 3.2 Backward

```yaml
work_item:       one logical K_j/V_j owner
fixed_axis:      j
scanned_axis:    all interacting i
transient_cell:  S_ij, P_ij, dP_ij, dS_ij
persistent:      dK_j, dV_j accumulators
final_output:    complete dK_j, dV_j
partial_output:  dQ_i_from_j
global_combine:  sum dQ_i_from_j over all interacting j owners
```

Per attention cell:

```text
S_ij  = Q_i K_j^T
P_ij  = exp(S_ij - LSE_i)
dP_ij = dO_i V_j^T
dS_ij = P_ij * (dP_ij - rowsum(dO_i * O_i))

dV_j += P_ij^T dO_i
dK_j += dS_ij^T Q_i
dQ_i_from_j = dS_ij K_j
```

Memory sentence:

> **Fix `j`, scan `i`, keep `dK_j/dV_j`; publish partial `dQ`.**

No loop orientation makes `dQ`, `dK`, and `dV` all single-owner. A Q-owner
would finish `dQ` but publish partial `dK/dV`; the audited high-parallelism
paths choose K/V ownership instead.

## 4. L2 Generation Lowering Matrix

The following forward rows are representative source-pinned paths. They are
chosen to make the schedule difference concrete, not to define every dispatch.

| Generation | Representative `S` tile | Physical executor versus logical owner | Loop/state lowering | Primitive and warp idea |
| --- | --- | --- | --- | --- |
| FA1 v1.0.9, long-sequence d128 | `16 x 128` | CTA split `r` owns several interleaved small `Q_i` tiles | K/V tile `j` outside; visit owned `i` inside; partial O/LSE state passes through HBM between `j` steps | four warps partition score columns and produce PV reduction partials |
| FA2 v2.0.0 SM80, d128 no dropout | `128 x 64` | one CTA directly owns one `Q_i -> O_i` block | fix `i`; scan `j` in reverse; keep `(m,l,U)` in registers | `cp.async`, warp MMA, split-Q warps with disjoint output rows |
| FA3 pinned SM90, causal d128 | `128 x 128` | one logical owner per Q/O tile; one persistent CTA may execute several logical tiles sequentially | FA2-style fix-`i` loop with circular K/V stages | TMA producer plus two WGMMA consumer warpgroups, 64 output rows each |
| FA4 pinned SM100 d128 family | `128 x 128` mathematical subtile | one logical owner per final row; work item may contain two Q stages and eligible paths may use a 2-CTA group | two Q streams retain independent online state while sharing the K/V progression | fully-async tcgen05, TMEM S/P/O handoff, separate softmax and correction roles |

Backward keeps a more stable logical geometry:

| Generation | K/V-owner loop | `dQ` publication/combine in the pinned path |
| --- | --- | --- |
| FA1 v1.0.9 | one body fixes `j` and scans small Q tiles; serial-K mode may let one CTA invoke the bodies in order | sequence-K mode uses contending FP32 atomic adds; public deterministic mode disables sequence-K parallelism |
| FA2 v2.0.0 | one CTA fixes `j`, scans Q tiles in descending order, and owns final `dK_j/dV_j` | FP32 atomic accumulation; the pinned public API has no deterministic switch |
| FA3 pinned early SM90 | one K/V work item scans Q tiles; consumers retain `dK_j/dV_j` | TMA reduce-add by default; supported deterministic path orders publishers with a semaphore |
| FA4 pinned SM100 | one K/V owner or cooperative owner group scans Q tiles; 2-CTA can reduce update count | bulk FP32 reduce-add by default; deterministic mode prescribes writer order with semaphores |

## 5. Four Filled Forward Specs

### FA1 -- tile the plane

```yaml
logical_owner:   one CTA split owns several small, disjoint Q_i tiles
loop_order:      j outside, owned i inside
kept_live:       current K_j/V_j; current cell fragments
handed_off:      partial O_i and normalization state between j steps
warp_partition:  K/V columns and PV reduction slices
reason:          avoid materializing the N x N attention matrix in HBM
```

### FA2 -- make output ownership direct

```yaml
logical_owner:   one CTA owns one larger Q_i -> O_i block
loop_order:      fix i, scan all valid j
kept_live:       Q_i, m_i, l_i, U_i through the whole K/V traversal
streamed:        K_j/V_j
warp_partition:  disjoint Q/output rows, not overlapping PV reduction slices
reason:          expose more Q-sequence CTAs and reduce partial-state work
```

### FA3 -- pipeline the owner

```yaml
logical_owner:   broadly the FA2 Q_i -> O_i owner
worker_lifetime: a persistent CTA may pull several logical tiles sequentially
kept_live:       online state and WGMMA accumulators in consumer registers
streamed:        K/V through TMA-managed circular SMEM stages
roles:           producer movement || consumer WGMMA/softmax
reason:          overlap movement and non-MMA work with faster Hopper MMA
```

### FA4 -- decouple and deepen the owner pipeline

```yaml
logical_owner:   one final owner/cooperative owner group per output row
work_item:       commonly two independent 128-row Q streams in the d128 family
kept_live:       S, P, and running O in lifetime-aliased TMEM regions
handoff:         MMA -> TMEM S -> softmax -> TMEM P -> MMA
roles:           MMA/control, two softmax groups, output correction, TMA load
cooperation:     optional 2-CTA physical lowering on eligible dispatches
reason:          keep Blackwell MMA fed as SMEM and exponential work become exposed
```

## 6. The Memorization Card

```text
Grid:
  one cell = (Q_i, K_j, V_j)

Forward:
  fix i, scan j, keep (m,l,U), finish O_i

Backward:
  fix j, scan i, keep dK_j/dV_j, publish partial dQ_i

Generations:
  FA1 = tile
  FA2 = own
  FA3 = pipeline
  FA4 = decouple + deepen
```

When explaining any concrete kernel, fill these seven fields:

```yaml
scope:          source revision, architecture, dtype, shape, mode
math_tile:      B_M x B_N x D
logical_owner:  which final tile is complete here?
scan_axis:      which contributor tiles are visited, and in what order?
live_state:     what survives across scan iterations, and where?
partials:       which values still need another owner or global combine?
physical_map:   CTA/cluster, warp roles, movement/MMA primitives
```

If one of the fields is missing, “the tile size is 128” is not yet a useful
kernel explanation.

## 7. Evidence And Scope Boundary

The representative rows above inherit the evidence envelopes of:

- [FA1 foundations](fa1-foundations.md), pinned to v1.0.9 for the audited CUDA
  forward/backward schedules;
- [FA2 forward](fa2-forward.md) and [FA2 backward](fa2-backward.md), pinned to
  v2.0.0 SM80;
- [FA3](fa3.md), pinned to the early official SM90 source snapshot documented
  there;
- [FA4](fa4.md), pinned to the SM100 CuTeDSL snapshot and d128 teaching scope
  documented there.

This spec excludes SplitKV/decode, paged attention, FlashMLA, model-specific
large-head-dimension paths, linear attention, and inference scheduling. Those
paths can change the owner graph and need their own filled spec rather than an
unqualified extension of this one.
