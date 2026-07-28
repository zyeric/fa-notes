# FlashAttention-2 Backward: Ownership, Pipeline, And Determinism

Date: 2026-07-28

Status: official v2.0.0 SM80 CUDA backward source study complete for the
standard sequence-K path; CPU-only reasoning, with GPU validation deferred

Read the [FA1 backward visual map](flashattention_fa1_backward.html) and the
[FA2 forward delta](flashattention_fa2_forward.md) first. For a graphical
reading surface, open the
[FA2 backward visual map](flashattention_fa2_backward.html). This Markdown
file is the source of truth.

## 1. Scope And Evidence

Pinned implementation:

- [Dao-AILab/flash-attention v2.0.0, commit
  `4f285b354796fb17df8636485b9a04df3ebbb7dc`](https://github.com/Dao-AILab/flash-attention/tree/4f285b354796fb17df8636485b9a04df3ebbb7dc).

Primary source landmarks:

- [`flash_bwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/flash_bwd_launch_template.h):
  the preprocess, sequence-K main, and dQ-conversion launches, plus tile
  dispatch;
- [`flash_bwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/flash_bwd_kernel.h):
  one-K/V-tile CTA body, five GEMMs, shared transposes, `atomicAdd`, and
  epilogues;
- [`kernel_traits.h`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/kernel_traits.h):
  MMA warp layouts, `cp.async`, shared layouts, and shared-size formulas;
- [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/flash_api.cpp):
  FP32 `dq_accum`, MQA/GQA expanded gradients, and final group reduction;
- [`flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/flash_attn/flash_attn_interface.py):
  the public v2.0.0 backward surface, which has no deterministic argument;
- [`tests/test_flash_attn.py`](https://github.com/Dao-AILab/flash-attention/blob/4f285b354796fb17df8636485b9a04df3ebbb7dc/tests/test_flash_attn.py):
  a limited 200-repeat exact-output race test.

The main derivation uses dense MHA, no dropout, equal Q/K length, SM80/A100,
and:

```text
B=1, H=32, N=8192, D=128
B_M=64, B_N=128
CTA=8 warps=256 threads
```

Dropout, causal, varlen, and MQA/GQA are added only where they change the
protocol. Later FA2 releases and their deterministic workspace mode are
outside this pinned-version verdict.

## 2. Math And One-Tile Dataflow

Let `G=dO` and:

$$
D_i=\sum_d G_{id}O_{id}.
$$

For one Q tile $I$ and K/V tile $J$:

$$
S_{IJ}=Q_IK_J^\top,\quad
P_{IJ}=\exp(S_{IJ}-LSE_I),
$$

$$
dP_{IJ}=G_IV_J^\top,\quad
dS_{IJ}=P_{IJ}\odot(dP_{IJ}-D_I),
$$

$$
dQ_I^{(J)}=dS_{IJ}K_J,\quad
dV_J\mathrel{+}=P_{IJ}^\top G_I,\quad
dK_J\mathrel{+}=dS_{IJ}^\top Q_I.
$$

The physical tile graph is:

```text
Q_I,K_J -> S[I,J] -> P[I,J]
                         |-> original P + dP - D -> dS[I,J] -> dQ_from_J
                         |                         \-> shared -> dS^T -> dK
                         \-> shared -> P^T -> dV

G_I,V_J -> dP[I,J] -----/
```

`P^T` and `dS^T` are current-tile shared-memory layout conversions, not
sequence-sized global tensors. `P` and `dS` begin as score-oriented register
fragments, are stored through swizzled shared layouts, and are reloaded under
the operand mapping required by the transposed GEMMs.

## 3. Three-Kernel Sequence-K Protocol

The v2.0.0 wrapper sets `loop=true`, and `run_flash_bwd` selects the
sequence-K path:

```text
1. preprocess grid over Q tiles:
     D_i = dot(G_i, O_i)
     zero global FP32 dq_accum

2. main grid over K/V tiles:
     CTA J fixes K_J,V_J
     loops Q tiles I in descending order
     owns final dK_J,dV_J
     atomically contributes dQ_I_from_J to dq_accum

3. conversion grid over Q tiles:
     FP32 dq_accum -> requested FP16/BF16 dQ
```

For the representative shape:

```text
Q blocks = 8192 / 64  = 128
K blocks = 8192 / 128 = 64

preprocess grid = 128 * 1 * 32 = 4096 CTAs
main grid       =  64 * 1 * 32 = 2048 CTAs
convert grid    = 128 * 1 * 32 = 4096 CTAs
```

The FP32 workspaces are:

```text
dq_accum = 1 * 32 * 8192 * 128 * 4 B = 128 MiB
softmax_d = 1 * 32 * 8192 * 4 B       =   1 MiB
```

This is the central speed/cost trade: the main work has many independent K/V
tile CTAs, but it pays preprocessing, conversion, global workspace traffic,
and atomic accumulation.

## 4. Main CTA Ownership And Order

One main CTA has fixed `(J,b,h)`:

```text
load fixed K_J,V_J
initialize FP32 acc_dK_J,acc_dV_J

for I = last valid Q tile down to first valid Q tile:
    load Q_I,G_I,LSE_I,D_I
    recompute P_IJ and form dS_IJ
    atomicAdd(dq_accum[I], dS_IJ K_J)
    acc_dV_J += P_IJ^T G_I
    acc_dK_J += dS_IJ^T Q_I

convert/store final dK_J,dV_J once
```

Thus:

- each `dK_J` and `dV_J` MHA output tile has one CTA owner;
- each `dQ_I` has one contributor per interacting K/V tile CTA;
- `dK/dV` accumulate Q tiles in a fixed descending source order inside one
  CTA;
- `dQ` contribution arrival order across independent `J` CTAs is not fixed.

Causal mode raises the first Q tile visited by a K/V CTA. A late K/V tile
interacts only with sufficiently late Q rows. The grid still contains all K/V
tiles, but active work and the number of dQ contributors vary by row.

## 5. Five GEMMs And Warp Partition

For the representative d128 traits, the eight-warps layouts are:

| Product | Tile shapes | Reduction | Warp layout over output | Combine consequence |
| --- | --- | --- | --- | --- |
| $S=QK^\top$ | `[64,128][128,128] -> [64,128]` | `D=128` | `2 x 4` over Q rows and K columns | disjoint score subtiles |
| $dP=GV^\top$ | `[64,128][128,128] -> [64,128]` | `D=128` | same `2 x 4` output layout | disjoint dP subtiles |
| $dQ^{(J)}=dS K$ | `[64,128][128,128] -> [64,128]` | `B_N=128` | `2 x 4` over Q rows and D columns | disjoint pieces of this CTA's dQ partial |
| $dV=P^\top G$ | `[128,64][64,128] -> [128,128]` | `B_M=64` | `4 x 2` over K rows and D columns | disjoint final dV pieces |
| $dK=dS^\top Q$ | `[128,64][64,128] -> [128,128]` | `B_M=64` | same `4 x 2` output layout | disjoint final dK pieces |

This is an important delta from the audited FA1 backward. FA1 split the `dQ`
reduction dimension among eight warps, created eight local `[B_r,D]`
partials, and combined them through shared memory. FA2's representative dQ
MMA partitions output dimensions, so one CTA's `dQ_from_J` does not need that
eight-way local numeric reduction. The cross-CTA `J` contributions still
need the global atomic combine.

Each table row describes a logical warp-level decomposition. A warp issues
multiple instruction-sized MMA operations, and lane/register fragments are
specialization-specific.

## 6. State Residence, Shared Memory, And Pipeline

For `D=128,B_M=64,B_N=128`, the source-selected A100 path requests:

```text
Q double buffers + dO tile: 3 * 64 * 128 * 2 B = 48 KiB
K and V tiles:              2 * 128 * 128 * 2 B = 64 KiB
dS tile:                        64 * 128 * 2 B = 16 KiB
P tile:                         64 * 128 * 2 B = 16 KiB
total                                                144 KiB
```

The buffers support:

- SM80 `cp.async` global-to-shared movement for Q/K/V;
- double-buffered Q while fixed K/V and `dK/dV` accumulators remain live;
- shared `P` and `dS` layout conversions;
- FP32 score, dP, dQ, dK, and dV accumulator fragments in registers.

An A100's shared capacity makes this a heavy CTA and strongly suggests one
such CTA resident per SM by shared capacity. Eight resident warps provide
the CTA's internal latency-hiding pool. This remains a source-level capacity
argument, not a measured occupancy result.

## 7. FA1-To-FA2 Backward Delta

| Axis | FA1 v1.0.9 backward | FA2 v2.0.0 backward |
| --- | --- | --- |
| representative Q tile | 16 rows | 64 rows at d128 |
| CTA | 8 warps | 8 warps |
| outer work unit | K/V tile body, optionally serial across J | always one sequence-K CTA per J in the public path |
| serial deterministic option | `num_splits=1` | absent in the v2.0.0 public API |
| dQ within one CTA | eight reduction-axis warp partials then fixed local sum | warps partition dQ output dimensions; no eight-way local sum |
| cross-J dQ | serial ordinary update or parallel atomic | FP32 global atomic |
| dK/dV | one J owner, Q loop | one J owner, larger Q tiles and output-axis warp partition |
| movement | historical ordinary-load staging | SM80 `cp.async` plus Q double buffering |
| protocol | one main kernel in serial mode; extra work in parallel mode | preprocess + sequence-K main + dQ convert |
| shared footprint | 152 KiB at the audited d128 path | 144 KiB at the representative d128 path |

FA2 improves useful work partitioning and removes the CTA-local dQ partial
combine, but the initial release does not solve cross-CTA dQ ordering.

## 8. Scoped Determinism Verdict

The main kernel performs scalar FP32:

```text
atomicAdd(&dq_accum[element], dQ_from_this_J[element])
```

for independent K/V tile CTAs. Atomicity prevents lost updates, but the source
does not impose a fixed order among J contributors. Because FP32 addition is
not associative, source-level bitwise repeatability for dQ must not be
assumed.

The pinned public Python interface has no `deterministic` argument. Therefore
there is no user-selectable serial/fixed-reduction alternative in this
version.

The v2.0.0 test suite nevertheless runs a limited FP16, d64, no-dropout
matrix 200 times and exact-compares forward, dQ, dK, and dV. That is useful
empirical maintainer evidence for those tested shapes/devices. It is not a
proof that CUDA schedules all K/V CTAs in one portable atomic arrival order.

For dense MHA in the studied path:

| Output | Source-level verdict | Reason |
| --- | --- | --- |
| `dQ` | do not assume bitwise repeatable | independent J CTAs atomically add to the same FP32 elements |
| `dK`,`dV` | expected repeatable in a fixed artifact | one J CTA owns the complete tile and uses a fixed descending Q loop |
| `D=dot(G,O)` | expected repeatable in a fixed artifact | one Q-tile CTA and fixed local reduction |

With dropout, the Philox state must also match. MQA/GQA creates per-Q-head
expanded dK/dV tensors and then calls a PyTorch group sum; that extra
reduction is a separate implementation boundary and is not covered by the
dense-MHA verdict.

## 9. Stop Line

This note does not establish:

- behavior of later FA2 releases or their deterministic workspace reduction;
- Hopper/FA3 or Blackwell/FA4 backward;
- a measured A100 speedup, occupancy, or atomic contention constant;
- deterministic behavior of the PyTorch MQA/GQA post-sum;
- batch invariance or inference-engine behavior.

The next useful validation is a GPU run that records the resolved traits,
hashes `D`, `dq_accum`, final dQ/dK/dV across repeats, and profiles atomic/L2
traffic for several `B,H,N,D,causal` regimes.
