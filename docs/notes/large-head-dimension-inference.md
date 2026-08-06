# Large-Head-Dimension Inference: Why `d=512` Changes The Schedule

Date: 2026-08-06

Status: source-backed inference-forward map and FA3/FA4 schedule comparison;
CPU-only reasoning, with production-kernel SASS, resource metadata, and
timeline profiling still pending

For the graphical reading surface, open the standalone
[large-d inference schedule comparison](../slides/d512-inference.html). This
Markdown file owns the claims and evidence boundaries used by that deck.

## Scope And Evidence

This note adds an explicit, narrow scope extension to the FA1--FA4 training
attention core. It covers the inference-forward question raised by
DeepSeek-style MLA and Gemma 4 global attention when the value/output head
dimension reaches 512. It does not qualify training backward, every paged-KV
backend, or one universal `d=512` kernel.

The comparison uses three evidence layers. The exact snapshots resolved on
2026-08-06 are:

| Source | Revision / record | What it supports |
| --- | --- | --- |
| FA3 pinned study | official source `3669b25206d5938e3cc74a5f7860e31c38af8204` | representative SM90 d128 roles and two-level overlap |
| FA3 current `LargeHeadDimV` | official source `d7e4dba3e568106b0f1b6323b07c1272f53679b3` | implemented SM90 `d_qk=64,d_v=512` tile and warpgroup ownership |
| FA4 | official source `849f660f73b176e5ad5670e7f822c7fa9f3eaf8b` as pinned in the FA4 note | representative SM100 d128 TMEM/tcgen05 schedule |
| DeepSeek-V4-Pro reference | Hugging Face revision `b5968e9190ef611bbf34a7229255be88a0e937c1` | readable TileLang/model ownership and allocation map |
| Gemma 4 31B config | Hugging Face revision `a77ab7d40c989d1d32f78353653c0769c6291146` | sliding/global head-dimension selection and K-equals-V flag |
| FlashAttention PR 2422 | head `cedf07d25715ed75eb9adc9d4ffc72a2b08e683d`; open PR record | experimental CuTe `d_qk,d_v<=512` path and unresolved two-WG QK discussion |
| vLLM FlashAttention PR 130 | merge commit `f5bc33c`; merged PR record | downstream import of the still-open upstream hdim512 and SplitKV changes |
| vLLM PR 38835 | merge commit `e24e0a4`; merged PR record | selected FA4 SM90 paged-KV/hdim512 dispatch integration and Gemma 4 H200 test |
| FlashMLA | main/HEAD `15f13e5030374295491c5ce31b02d7e63a7772c6` | current support matrix, decode metadata, and sparse-prefill API |
| SGLang PR 25418 | merge commit `93173b2`; merged PR record | DeepSeek V4 sparse-prefill engine integration and reported measurement |

The evidence layers are:

- the pinned FA3 and FA4 mechanisms already documented in
  [FA3 on Hopper](fa3.md) and [FA4 on Blackwell](fa4.md), plus the current
  official FA3
  [`LargeHeadDimV` implementation](https://github.com/Dao-AILab/flash-attention/blob/d7e4dba3e568106b0f1b6323b07c1272f53679b3/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp);
- the public DeepSeek TileLang
  [model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)
  and
  [kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py)
  reference plus the pinned
  [FlashMLA support/API](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6);
- the public SM90 head-dim-512 design discussion in
  [FlashAttention PR 2422](https://github.com/Dao-AILab/flash-attention/pull/2422),
  the merged downstream [vLLM FlashAttention PR 130](https://github.com/vllm-project/flash-attention/pull/130),
  the merged [vLLM PR 38835](https://github.com/vllm-project/vllm/pull/38835),
  and the merged [SGLang PR 25418](https://github.com/sgl-project/sglang/pull/25418).

The status is not the binary statement "head dim 512 has no solution":

- Gemma 4 full/global attention resolves a 512-wide head, and vLLM has a
  merged SM90+FA4 integration tested by the PR author with
  `google/gemma-4-31B-it` on one H200;
- that vLLM integration imports the ongoing upstream PR 2422 into the vLLM
  FlashAttention fork, while PR 2422 itself remains open and its review still
  questions the two-warpgroup QK ownership;
- SGLang's DeepSeek V4 path is a different sparse-MLA problem. Its merged
  prefill integration calls FlashMLA's `flash_mla_sparse_fwd`, whose published
  MQA contract is `d_qk=576,d_v=512`, not ordinary dense 512/512 attention.

Therefore a working downstream serving integration exists, but this note does
not promote the open upstream 512/512 discussion to one canonical, reviewed
per-warp schedule. The readable DeepSeek reference establishes another
source-level ownership map; it is not proof that vLLM, SGLang, FlashMLA, and
every FA4 artifact use the same CTA shape. Production claims must name the
actual engine image, backend revision, GPU, and dispatch.

## 1. Separate `d_qk` From `d_v`

For

$$
S=QK^\top,\qquad P=\operatorname{softmax}(S),\qquad O=PV,
$$

the two large dimensions have different roles:

| Dimension | QK role | PV role | Main pressure |
| --- | --- | --- | --- |
| `d_qk` | reduction axis | absent | number of QK MMA updates and operand bytes |
| `d_v` | absent | output-column axis | output accumulator width and output bytes |

Ordinary attention often has `d_qk == d_v`, but an MLA backend need not. The
current FlashMLA support table describes its MQA mode as `head_dim_k=576` and
`head_dim_v=512`. The useful mental decomposition is a 512-wide latent/NoPE
part plus a 64-wide RoPE part on QK, while the value/output remains 512 wide.
Therefore a log line saying only `d=512` is insufficient to reconstruct the
QK instruction count.

MQA reduces K/V duplication and permits reuse of one K/V tile by many Q
heads. It does not remove either fact:

```text
one score still reduces the complete d_qk
one output head still contains d_v output columns
```

The model cases also resolve differently:

| Model path | Head relation | Resolved large dimension |
| --- | --- | --- |
| DeepSeek-style compressed MLA/MQA | many Q heads share one compressed KV entry | current FlashMLA interface uses `d_qk=576`, `d_v=512` |
| Gemma 4 sliding attention | model-specific GQA; 31B config sets K equal to V | `head_dim=256` |
| Gemma 4 global/full attention | same model family, different layer type | `global_head_dim=512` |

For Gemma 4, dispatch must therefore resolve the layer type rather than read
one top-level dimension and apply it to every layer. The pinned
[31B configuration](https://huggingface.co/google/gemma-4-31B-it/blob/a77ab7d40c989d1d32f78353653c0769c6291146/config.json)
is model evidence, not a kernel-selection guarantee.

## 2. Why The MMA-To-Softmax Ratio Changes

For one fixed score tile with `B_r` logical Q rows and `B_c` KV rows:

```text
QK work       ~= 2 * B_r * B_c * d_qk FLOPs
PV work       ~= 2 * B_r * B_c * d_v  FLOPs
softmax state ~= O(B_r * B_c) elements
```

If both head dimensions grow from 128 to 512 while `B_r` and `B_c` remain
fixed, QK and PV contain about four times as much MMA work while the number of
score elements presented to softmax is unchanged. Address generation,
barriers, scheduler work, and parts of pipeline setup also do not scale by
four. Large `d` can therefore make the forward kernel more compute-dense and
make pointwise softmax easier to overlap with independent matrix or movement
work.

This does not mean the operation becomes cheaper. Q/K/V movement, output
traffic, accumulator state, and total FLOPs all grow. The accurate statement
is:

> A large-d inference-forward kernel can achieve better Tensor Core
> utilization while consuming more absolute work and approaching sharper
> register, SMEM, TMEM, and occupancy limits.

MQA reuse is the stronger arithmetic-intensity lever. Loading one shared KV
tile once and using it for many Q heads adds QK/PV work without duplicating
the KV bytes for every head. Increasing `d` alone often increases both work
and bytes, so it does not guarantee a higher FLOP/byte ratio.

## 3. The Readable DeepSeek Tile

The public DeepSeek TileLang reference uses this sparse-attention grid:

```text
grid = (query_position, batch)
one CTA = all h_local query heads for one query position
```

It walks selected KV entries in blocks of 64. Its core shapes are:

```text
Q_shared  : [h_local, 512] BF16
KV_shared : [64,      512] BF16

acc_s = Q @ KV^T : [h_local, 64]  FP32 fragment
acc_o = P @ KV   : [h_local, 512] FP32 fragment
```

With `h_local=16`, the generated score tile is `[16,64]`. The first axis is
16 local query heads, not 16 query positions. Softmax operates along the 64
selected-KV-token axis; each individual score still reduces the complete head
dimension.

The explicit shared-memory payload is:

```text
Q_shared  [16,512] BF16 = 16 KiB
KV_shared [64,512] BF16 = 64 KiB
O_shared  [16,512] BF16 = 16 KiB
P_shared  [16, 64] BF16 =  2 KiB
---------------------------------
explicit shared memory          = 98 KiB
```

The main FP32 fragment payloads are 32 KiB for `acc_o` and 4 KiB for
`acc_s`, distributed across the CTA in the reference lowering. Compiled
registers/thread, spills, and occupancy remain JIT/GPU measurements.

This source explains feasibility without assuming that Hopper or Blackwell
has a larger per-CTA SMEM pool: inference keeps one bounded KV tile, reuses it
across local heads, and has no `dQ/dK/dV` partial arrays.

## 4. FA3 `d=128`: Two Q-Row Owners Ping-Pong

The representative FA3 Hopper forward path uses TMA, WGMMA, warp
specialization, and two levels of overlap. Its two consumer warpgroups own
disjoint Q/output row fragments:

```text
producer WG: TMA Q/K/V into circular SMEM stages

consumer WG1: Q rows   0:64  -> QK -> softmax -> PV -> O rows   0:64
consumer WG2: Q rows 64:128  -> QK -> softmax -> PV -> O rows 64:128
```

The first overlap level pipelines TMA movement against consumer computation.
The second interleaves asynchronous QK/PV WGMMA with softmax across current
and next score fragments and across the two independent row owners:

```text
WG1: GEMM(j)    softmax(j)  GEMM(j+1) ...
WG2: softmax(k) GEMM(k+1)   softmax(k+1) ...
```

Each consumer owns its rows through the complete K/V traversal. No consumer
needs another consumer's probability tile to compute its PV result.

## 5. FA4 `d=128`: TMEM And A Deeper Asymmetric Pipeline

The representative FA4 Blackwell note also uses `d=128`, but its motivation
is a different hardware imbalance: matrix throughput grows faster than SMEM
and exponential throughput. `tcgen05` places matrix accumulators in TMEM and
supports fully asynchronous issue, while specialized softmax/correction roles
consume score fragments and update online state.

The key comparison point is not merely WGMMA versus `tcgen05`:

```text
FA3 d128:
  consumer warpgroups own disjoint Q-row slices
  score and O fragments primarily live in distributed registers

FA4 d128:
  TMEM decouples matrix accumulator residence from ordinary registers
  deeper role/pipeline decomposition attacks the faster-MMA / fixed-exp gap
  selected paths may use 2-CTA cooperation
```

Both defaults start from a row-oriented logical owner graph. Neither default
alone explains how to avoid several replicated 512-wide partial outputs.

## 6. A Concrete Implemented Large-`d_v` Warpgroup Map

### Why the ordinary FA3 d128 row-owner schedule cannot simply be widened

The representative FA3 d128 forward tile has `B_M=B_N=128`. Two consumer
warpgroups own disjoint 64-row slices, and each group keeps one complete
`O[64,128]` FP32 accumulator fragment. Ignoring fragment-layout padding, that
is

```text
64 * 128 FP32 values / 128 threads = 64 FP32 values per thread
```

Naively changing only `d_v` to 512 while retaining the same row ownership
would make each consumer group own `O[64,512]`:

```text
64 * 512 FP32 values / 128 threads = 256 FP32 values per thread
```

This is already one register per FP32 accumulator value on average. The
[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)
publishes a maximum of 255 registers per thread, so the O
accumulator alone crosses the per-thread limit before score fragments,
softmax state, pointers, predicates, or loop state are counted. This is a
capacity argument for changing ownership; it is not compiled `ptxas` metadata
for one selected binary.

The wide V tile also tightens shared-memory staging. At
`B_N=64,d_v=512,BF16`, one logical V stage is
`64*512*2 = 64 KiB`; two such stages are already 128 KiB before Q, K, P,
barriers, and layout padding. The same NVIDIA guide gives H100 up to 227 KiB
per CTA with opt-in,
so this does not by itself prove that every two-stage layout is illegal. It
does prove that the old register-resident-P / row-owner schedule is not a free
drop-in: accumulator ownership, tile size, stage count, and the P handoff must
be designed together.

The important correction is therefore:

> The first hard wall in the naive d128-to-dv512 widening is the RMEM O
> accumulator. Wide V staging simultaneously consumes much more SMEM budget.

The clearest implemented public response at revision `d7e4dba...` is the
BF16/FP16 SM90 `LargeHeadDimV` specialization with

```text
d_qk = 64, d_v = 512
B_M = 64, B_N = 64
Q [64, 64], K_j [64, 64], S_j/P_j [64, 64]
V_j [64, 512], O [64, 512]
```

The tile-size dispatch returns `{64,64,false,false}`. `TiledMmaQK` has one
consumer warpgroup, while `TiledMmaPV` uses
`AtomLayoutPV=(1,d_v/256,1)`: two warpgroups when `d_v=512`. The kernel maps
the CTA as follows:

```text
producer group (physical WG0 slots):
  one elected warp issues TMA for Q, K_j, V_j

consumer WG1 (128 threads = 4 warps, WGMMA collective):
  S_j[64,64] = Q[64,64] @ K_j[64,64]^T
  P_j[64,64] = online_softmax(S_j)
  publish P_j to SMEM
  O[:,0:256] += P_j @ V_j[:,0:256]

consumer WG2 (128 threads = 4 warps, WGMMA collective):
  wait for the same P_j in SMEM
  O[:,256:512] += P_j @ V_j[:,256:512]
```

WG1 therefore performs QK, softmax, and the first PV column slice. WG2 does
not read Q or K and does not recompute softmax; it performs only the second PV
column slice. Both traverse every K/V tile `j`, receive the same online-softmax
rescale factors through SMEM, and store disjoint final O columns. There is no
cross-WG numeric sum of O.

The four warps inside one consumer warpgroup must be described as one WGMMA
collective. The accumulator fragments are interleaved across its 128 threads;
the source does not assign warp 0 a clean contiguous 16-row rectangle, warp 1
the next 16 rows, and so on. The stable teaching unit here is:

```text
WG1 owns O[64 rows, columns   0:256]
WG2 owns O[64 rows, columns 256:512]
```

Each logical FP32 O slice is `64*256*4 = 64 KiB`, distributed across one
consumer warpgroup, or 128 FP32 accumulator values per thread on average.
This restores register headroom relative to the naive 256-value case. The
cost is that P can no longer remain private to one row owner: WG1 must publish
the shared `P[64,64]` tile and online-softmax rescale state through SMEM for
WG2, using `PFull/PEmpty` barriers. The schedule change is thus an ownership
change plus an explicit register-to-SMEM handoff.

### Why not choose `B_M=32`?

It is a valid alternative specialization, not a correctness problem. For
`d_v=512`, an unsplit logical FP32 O tile at `B_M=32` is also 64 KiB. But it is
not a drop-in change for this path:

- Hopper WGMMA uses a 64-row M collective in this mapping; a 32-row tile needs
  predication, packing with another logical tile, or another MMA layout;
- the sequence has twice as many Q-tile owners as `B_M=64`, so each staged K/V
  tile is reused across half as many Q rows and fixed CTA work repeats twice;
- more CTAs can improve parallelism, so only compiled resource metadata and
  profile measurements decide which specialization wins.

There is no universal Tensor Core rule that `B_M` must be at least 32. The
implemented choice is `B_M=64` because it matches this WGMMA M tile while two
PV warpgroups split the wide accumulator.

### What about ordinary `d_qk=d_v=512`?

That is a different kernel. Gemma 4 supplies a real 512-wide full-attention
use case, and vLLM has merged a serving integration that imports the relevant
ongoing FlashAttention changes into its downstream fork. Upstream PR 2422
changes the CuTe path to accept both widths up to 512, selects
`B_M=B_N=64`, one stage, and experiments with two MMA warpgroups along N. Its
review remains unresolved: the author describes QK N-axis splitting, while
the maintainer questions the stated QK dimensions and recommends the existing
`LargeHeadDimV` producer/PV pattern. The correct boundary is therefore
"working downstream integration, upstream schedule not yet canonical," not
"no implementation exists."

### What the imported Gemma 4 SM90 code actually establishes

The pinned Gemma 4 configuration uses `head_dim=256` for sliding-attention
layers and `global_head_dim=512` for full-attention layers. In the full layer,
Q, K, V, and O therefore use a 512-wide per-head attention dimension. The
`attention_k_eq_v=true` flag says that the K projection is reused as V; it does
not reduce the QK reduction width to 64.

The downstream implementation imported by vLLM FlashAttention PR 130 contains
two commits from the open upstream PR 2422. At the imported revision, source
inspection establishes:

```text
tile_m = 64
tile_n = 64
num_stages = 1 when d_qk > 256 or d_v > 256
PV WGMMA output-N tiler = min(256, d_v)
atom_layout_n = 2 when d_qk > 256 or d_v > 256
```

The `atom_layout_n=2` choice is applied to both QK and PV tiled-MMA builders in
the imported WIP commit. That is enough to say that the downstream artifact
uses two MMA warpgroups for the wide case and caps one PV WGMMA output slice at
256 columns. It is not enough to publish one stable, canonical Gemma 4
per-warpgroup picture: the upstream review explicitly questions whether the
two QK warpgroups split the score-N work correctly or duplicate work, and asks
for clarification of the stated QK MMA dimensions.

The safe teaching boundary is:

- **model fact:** Gemma 4 full attention is a real `d_qk=d_v=512` case;
- **downstream source fact:** vLLM imports a `64x64`, one-stage, two-MMA-WG WIP
  specialization and has a merged H200 serving test;
- **open design question:** the exact QK ownership in that two-WG CuTe path is
  not yet settled upstream, so the implemented `d_qk=64,d_v=512`
  `LargeHeadDimV` owner map must not be copied onto Gemma 4.

## 7. How Much Softmax Is Actually Hidden

The large-d path inherits the Hopper opportunity to overlap TMA movement,
asynchronous matrix work, and pointwise softmax across pipeline iterations.
The intended steady-state dependency graph is:

```text
TMA K/V tile j+1
        ||
QK WGMMA for tile j
        -> wait for S_j
        -> online softmax produces P_j
        -> publish P_j
        -> two PV WGMMA consumers update disjoint O columns
```

Across tiles, independent work may overlap `softmax_j`, `PV_j`, and movement
or QK for a neighboring stage. However, the exact FA3 d128 ping-pong schedule
cannot be copied verbatim: the large-d design makes one probability tile a
shared producer/consumer boundary, and one warpgroup may participate in both
QK/softmax and one half of PV.

Therefore source structure establishes an overlap opportunity, not a measured
percentage. Claims that softmax is fully hidden require the selected binary,
WGMMA/TMA wait placement, register/SMEM metadata, and an Nsight/SASS timeline.

## 8. Prefill And Decode Are Different Engine Paths

SGLang's merged V4 sparse-prefill integration calls
`flash_mla_sparse_fwd` rather than routing prefill through the paged
decode-style `flash_mla_with_kvcache` path. The PR attributes the change to
direct TMA loading and simpler loading logic and reports approximately 1.35x
kernel speedup for its stated setup. That is a paper/PR-reported measurement,
not a local measurement.

FlashMLA decode exposes a separate scheduler/combine contract:

```text
get_mla_metadata(...) -> tile_scheduler_metadata, num_splits
flash_mla_with_kvcache(..., num_splits, ...) -> O, LSE
```

Output-column partition and SplitKV must not be conflated:

```text
output-column split inside one work unit:
    owners write disjoint O columns
    combine = concat/disjoint store

SplitKV across sequence work units:
    owners produce partial online-softmax O/LSE states
    combine = mathematically meaningful later reduction
```

The split count and combine artifact are part of any repeatability or exact
numerics contract.

## 9. What Is And Is Not Hardware-Friendly

Large-d inference can be friendly to Tensor Core utilization because:

- more QK/PV MMA work amortizes fixed per-score and scheduling costs;
- MQA permits one KV tile to feed many Q heads;
- output-column ownership gives substantial independent PV work;
- forward avoids the many-writer gradient state of training backward.

It is simultaneously resource-constrained because:

- operand and accumulator footprints grow;
- wider output state can reduce occupancy or force a smaller row/head tile;
- publishing P adds an SMEM round trip and synchronization;
- decode can remain KV-cache bandwidth bound;
- spills or an unfavorable tile boundary can create a performance cliff.

The useful summary is:

> Large-d inference is more compute-dense but more resource-constrained. A
> successful kernel turns the extra work into regular Tensor Core work through
> MQA reuse, ownership changes, and pipelining; it does not make the attention
> mathematically cheaper.

## 10. Validation Ledger

Before accepting a production `d=512` claim, record:

- model layer type and resolved `d_qk/d_v`;
- MHA/GQA/MQA ratio and local heads after TP;
- engine, backend, source revision, GPU architecture, and kernel artifact;
- prefill or decode, dense/sparse indices, KV-cache dtype, and paged layout;
- Q/KV/head/output tile shapes and warpgroup/CTA/cluster roles;
- registers/thread, SMEM/CTA, TMEM/cluster, occupancy, and spills;
- SplitKV count and combine implementation;
- SASS/Nsight evidence for realized softmax/MMA/TMA overlap;
- repeated exact hashes for O and LSE on the fixed artifact.

Inference-forward success does not establish that a deterministic large-d
training backward exists.
