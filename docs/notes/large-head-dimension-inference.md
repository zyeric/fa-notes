# Large-Head-Dimension Inference: Why `d=512` Changes The Schedule

Date: 2026-08-05

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
2026-08-05 are:

| Source | Revision / record | What it supports |
| --- | --- | --- |
| FA3 | official source `3669b25206d5938e3cc74a5f7860e31c38af8204` | representative SM90 d128 roles and two-level overlap |
| FA4 | official source `849f660f73b176e5ad5670e7f822c7fa9f3eaf8b` as pinned in the FA4 note | representative SM100 d128 TMEM/tcgen05 schedule |
| DeepSeek-V4-Pro reference | Hugging Face revision `b5968e9190ef611bbf34a7229255be88a0e937c1` | readable TileLang/model ownership and allocation map |
| Gemma 4 31B config | Hugging Face revision `a77ab7d40c989d1d32f78353653c0769c6291146` | sliding/global head-dimension selection and K-equals-V flag |
| FlashAttention PR 2422 | head `cedf07d25715ed75eb9adc9d4ffc72a2b08e683d`; open PR record | SM90 head-dim-512 design direction |
| vLLM PR 38835 | head `ad526440279417eee5fabb07066e9fdbfc507f7b`; merged PR record | selected FA4 SM90 paged-KV/hdim512 dispatch integration |
| FlashMLA | main/HEAD `15f13e5030374295491c5ce31b02d7e63a7772c6` | current support matrix, decode metadata, and sparse-prefill API |
| SGLang PR 25418 | head `d54a5457ab9098ff0e22eb763d625301ec096ad0`; merged PR record | sparse-prefill engine integration and reported measurement |

The evidence layers are:

- the pinned FA3 and FA4 mechanisms already documented in
  [FA3 on Hopper](fa3.md) and [FA4 on Blackwell](fa4.md);
- the public DeepSeek TileLang
  [model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)
  and
  [kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py)
  reference plus the pinned
  [FlashMLA support/API](https://github.com/deepseek-ai/FlashMLA/tree/15f13e5030374295491c5ce31b02d7e63a7772c6);
- the public SM90 head-dim-512 design discussion in
  [FlashAttention PR 2422](https://github.com/Dao-AILab/flash-attention/pull/2422),
  the merged [vLLM PR 38835](https://github.com/vllm-project/vllm/pull/38835),
  and the merged [SGLang PR 25418](https://github.com/sgl-project/sglang/pull/25418).

The readable DeepSeek reference establishes one source-level ownership map;
it is not proof that vLLM, SGLang, FlashMLA, and every FA4 artifact use the same
CTA shape. PR 2422 remains an open upstream design discussion at this date,
whereas vLLM carries a merged integration for its selected backend. Production
claims must name the actual engine image, backend revision, GPU, and dispatch.

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

## 6. The Large-`d_v` Ownership Pivot

For a small `d_v`, partitioning the PV reduction axis can be reasonable:

```text
owner 0 computes a complete-width partial O from KV rows J0
owner 1 computes a complete-width partial O from KV rows J1
final O = partial_O0 + partial_O1
```

At `d_v=512`, every replicated complete-width FP32 partial becomes expensive.
The public SM90 large-head-dimension discussion instead points toward an
output-column partition:

```text
QK / softmax producer:
    compute S and P once
    publish P through shared memory

PV consumer WG0:
    O[:,   0:256] = P @ V[:,   0:256]

PV consumer WG1:
    O[:, 256:512] = P @ V[:, 256:512]
```

The two outputs are concatenated or stored to disjoint addresses; they are
not added. This removes replicated complete-width partial O state, but both
PV owners now need the complete probability rows. Relative to the FA3 d128
row-owner schedule, the communication boundary moves:

| Schedule | Owner split | Cross-owner payload | Final combine |
| --- | --- | --- | --- |
| FA3 d128 representative | Q/output rows | none for P | disjoint row stores |
| FA4 d128 representative | row-oriented logical owner with TMEM roles | TMEM/SMEM handoffs inside its deeper pipeline | disjoint logical output stores |
| SM90 large-`d_v` direction | output columns after shared QK/softmax | complete P tile | concat/disjoint column stores |

This is not the only legal large-d schedule. A production backend may use a
fixed head group, another score tile, different SMEM staging, or a cluster.
The reusable rule is that splitting a reduction axis creates numeric partials
and a combine obligation, whereas splitting an output axis creates disjoint
results and a data-distribution obligation.

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
