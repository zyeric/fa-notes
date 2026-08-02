# FlashAttention-1: From Online Softmax To Determinism

Date: 2026-07-24

Status: FA1 algorithm and historical CUDA forward/backward source study
complete through the deterministic and sequence-parallel paths; CPU-only
reasoning, with GPU validation still pending

## Scope And Reading Contract

This note stops at FlashAttention-1 (FA1). It is the mathematical and
mechanical foundation for the broader
[FlashAttention source audit](current-implementation-and-determinism.md); it does not expand FA2,
FA3, FA4, CuTe, paged attention, or inference-engine scheduling.
The separate axis-training-dev-tools
[large-head-dimension attention investigation](https://github.com/rStar-RL/axis-training-dev-tools/blob/master/contexts/megatron/ongoing/yi/determinism_foundations/operators/large_head_dim_attention.md) uses this
FA1 layout as a counterfactual for DeepSeek-V4 and Gemma 4 `d=512` paths
without attributing FA1's work partition to those later implementations.

The core derivation first uses one batch element, one attention head, no mask,
no dropout, and equal query/key sequence length. These features are added back
only after the basic dataflow is clear. Batch elements and heads are
independent copies of the same calculation.

Pinned evidence:

- FA1 paper:
  [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,
  arXiv:2205.14135v2](https://arxiv.org/abs/2205.14135v2);
- historical FA1 implementation:
  [Dao-AILab/flash-attention v1.0.9,
  commit `6d48e14a6c2f551db96f0badc658a6279a929df3`](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3);
- hardware/ISA references:
  [NVIDIA A100 Tensor Core GPU Architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
  and
  [CUDA 11.0 PTX ISA: `mma.m16n8k16` fragments and `mma.sync`](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type).

The paper and the implementation are separate evidence. The paper proves that
the tiled algorithm computes the same mathematical attention function. The
CUDA source is needed to decide who writes each output, which reductions run
concurrently, and what the `deterministic` option actually changes.

### Document role and reader frontier

This is both the human learning narrative and the pinned FA1 source audit. It
follows the source repo's
[kernel and communication research workflow](https://github.com/rStar-RL/axis-training-dev-tools/blob/master/contexts/kernel_communication_research.md)
without splitting one tightly coupled operator across several files.

The assumed reader frontier is:

- ordinary attention and the broad purpose of FlashAttention are familiar;
- online softmax, CUDA work ownership, state residence, tile dispatch, and
  reduction order need to be rebuilt explicitly;
- generic CTA/SM admission and occupancy background lives in
  [`gpu-hardware-notes`](https://github.com/zyeric/gpu-hardware-notes/blob/main/docs/notes/gpu-execution-model.md);
- forward has completed a reader-question pass;
- backward has been traced through its mathematical reductions, tile
  ownership, state residence, pipeline, and the `num_splits` implementation
  boundary; dropout and GPU measurements remain natural follow-up frontiers.

### Recommended reading paths

For a short re-entry before a later-generation comparison, read the
[FA1 one-page checkpoint](fa1-checkpoint.md). It compresses the
forward/backward ownership, state residence, and scoped determinism verdict
without replacing the evidence in this file.

For a graphical forward-only reading surface, open the standalone
[FA1 forward visual map](../slides/fa1-forward.html). It reorganizes the
same source-audited conclusions into beginner-oriented 16:9 diagrams. It first
introduces A100 compute/memory architecture and the
grid/CTA/warp/register/synchronization model before tiling, then follows a
`B=1,H=32,N=8192,d=128` example through online softmax, ownership, memory
movement, shared-bank swizzles, the four-warp pipeline, modes, and the
determinism proof. This Markdown file remains the source of truth.

For the corresponding backward reading surface, open the standalone
[FA1 backward visual map](../slides/fa1-backward.html). It assumes the
forward deck's CUDA vocabulary, then follows the same
`B=1,H=32,N=8192,d=128` example from the five gradient equations through
recomputed P, eight-warp K/V-tile ownership, shared-memory transpose
handoffs, serial versus sequence-K `dQ` accumulation, and the scoped
determinism verdict. It also uses the concrete `P`/`dS` transpose path to show
why live register data is not automatically reusable under a different MMA
fragment contract.

For the completed forward track, read straight through Sections 1--9. The
lowering path is:

```text
ordinary attention
  -> stable and online softmax
  -> paper Q/K/V tiling
  -> v1.0.9 dispatch and CTA ownership
  -> memory residence and Tensor Core reductions
  -> causal/varlen scheduling
  -> fixed-envelope forward determinism
```

For backward, first finish the forward track, then read Sections 10--12 before
the shared verdict and verification sections.

### Lowering and evidence map

| Layer | FA1 object | Primary section/evidence |
| --- | --- | --- |
| mathematical contract | $O=\operatorname{softmax}(QK^\top)V$ | Sections 1--5 |
| reference algorithm | materialized scores/probabilities | Section 3 |
| execution rewrite | tiled online softmax without $N^2$ HBM state | Sections 5--6 |
| forward kernel pattern | one owner over Q rows, sequential K/V traversal | Section 9 |
| CUDA model | CTA grid, warps, shared/global state, MMA chain | Section 9 |
| hardware envelope | SM80/SM90 resources and Tensor Core instructions | Section 9 |
| backward contract | recompute $P$; produce $dQ,dK,dV$ | Sections 10--11 |
| backward implementation | single owner or sequence-K partial `dQ` writers | Section 12 |
| scoped claim | conditional forward/backward repeatability | Sections 13--18 |

## 1. Four Terms That Must Not Be Collapsed

### 1.1 Exact attention

FA1 is called **exact** because, over real-number arithmetic, it computes

$$
O = \operatorname{softmax}(QK^\top)V
$$

without replacing dense attention by a low-rank, sparse, or otherwise
approximate model. It does not mean that a FA1 kernel returns the same bits as
a particular unfused PyTorch implementation.

### 1.2 Numerically stable

Subtracting a row maximum before exponentiation avoids overflow and underflow
as much as practical. This changes the evaluation procedure, not the
real-number softmax value.

### 1.3 Numerically close

Two implementations can compute the same real-number function but differ by a
small floating-point error because they group additions differently, use
different tile sizes, or lower matrix multiplication differently.

### 1.4 Bitwise deterministic

For this study, deterministic means that repeated execution with the same
inputs, state, dispatch, binary, and hardware envelope returns the same bits.
It is a property of the realized execution graph, not of the attention formula
alone.

The first useful conclusion is therefore:

> `exact attention` answers which mathematical function is computed;
> `deterministic attention` answers whether one resolved implementation
> repeats the same floating-point execution result.

## 2. Minimal Vocabulary

Let:

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $N$ | scalar | sequence length |
| $d$ | scalar | head dimension |
| $Q,K,V$ | $N \times d$ | query, key, and value matrices |
| $S$ | $N \times N$ | attention scores |
| $P$ | $N \times N$ | row-wise softmax probabilities |
| $O$ | $N \times d$ | attention output |
| $B_r$ | scalar | number of query rows in a tile |
| $B_c$ | scalar | number of key/value rows in a tile |

A **tile** is a rectangular submatrix. A **CTA** or thread block is a group of
GPU threads that can synchronize and share on-chip memory. The paper's tiled
pseudocode does not by itself specify the CTA mapping; that mapping must be
read from the implementation.

Two memory levels matter here:

- HBM is large off-chip GPU memory. Reading and writing it is relatively
  expensive.
- SRAM/registers are small on-chip storage. They are much faster but cannot
  hold the full $N \times N$ score or probability matrix for long sequences.

On A100, warp schedulers, the register file, the shared-memory/L1 pool,
load/store units, CUDA-core/SFU paths, and Tensor Cores are all components
inside each SM. They are not a second GPU-wide layer beside the SMs. L2 is
GPU-wide and HBM is outside the SMs; the deck's architecture diagram uses a
separate “zoom into one SM” box to make that boundary explicit.

A matrix multiplication already contains reductions:

$$
C_{ij} = \sum_k A_{ik}B_{kj}.
$$

The output element $C_{ij}$ has one logical value, but an implementation must
choose which threads compute its partial sums and in which order they are
combined. Since floating-point addition is not associative, this reduction
tree is part of a bitwise determinism claim.

## 3. Ordinary Attention

For clarity, absorb the usual scale $1/\sqrt d$ into $Q$ for now. Ordinary
dense attention performs three conceptual steps:

$$
S = QK^\top,\qquad
P_{ij} = \frac{e^{S_{ij}}}{\sum_t e^{S_{it}}},\qquad
O = PV.
$$

Softmax is row-wise. Query row $i$ needs every key column $j$ because its
normalizer contains the entire row:

$$
O_i =
\frac{\sum_j e^{S_{ij}}V_j}
     {\sum_j e^{S_{ij}}}.
$$

A conventional implementation can materialize $S$ and $P$ in HBM between
kernels. That requires $O(N^2)$ intermediate storage and repeatedly moves
quadratic-sized tensors through HBM. FA1's main question is not how to avoid
the $O(N^2d)$ arithmetic; it is how to avoid materializing those $N^2$
intermediates.

One useful materialized baseline is:

```text
GEMM kernel:    Q,K -> S
HBM boundary:   write/read S
softmax kernel: S -> P
HBM boundary:   write/read P
GEMM kernel:    P,V -> O
```

For $B=1,H=32,N=8192,d=128$ in BF16, QK plus PV perform about
$4BHN^2d=1.10\times10^{12}$ FLOPs. Merely writing and rereading both
$S$ and $P$ moves about 16 GiB, bounding intensity at roughly 64 FLOP/byte
even before other traffic. An A100 dense-BF16 Tensor-Core/HBM roofline anchor
is about
$312\ \mathrm{TFLOP/s}/1.555\ \mathrm{TB/s}\approx200$ FLOP/byte. The
materialized baseline is therefore on the bandwidth side of this simplified
roofline. FA1 moves the operational point right by removing the quadratic HBM
intermediate traffic. This is an upper-bound model, not proof that the fused
kernel is compute-bound: actual Q/O state traffic, tile reuse, `num_splits`,
causal work, occupancy, and achieved bandwidth still matter.

## 4. Safe Softmax

Directly computing $e^x$ can overflow. For a vector $x$, define:

$$
m = \max_j x_j,\qquad
\ell = \sum_j e^{x_j-m},\qquad
\operatorname{softmax}(x)_j = \frac{e^{x_j-m}}{\ell}.
$$

Subtracting $m$ is valid because the same factor cancels from numerator and
denominator:

$$
\frac{e^{x_j-m}}{\sum_t e^{x_t-m}}
=
\frac{e^{-m}e^{x_j}}{e^{-m}\sum_t e^{x_t}}
=
\frac{e^{x_j}}{\sum_t e^{x_t}}.
$$

The state $(m,\ell)$ is sufficient to normalize one complete row, but FA1
cannot see the complete row at once. The next step is to merge the state from
separate column tiles.

## 5. Online Softmax, One Merge At A Time

Suppose the scores already seen have state $(m_\mathrm{old},
\ell_\mathrm{old})$, and the next score tile has local state
$(m_\mathrm{tile},\ell_\mathrm{tile})$.

The merged maximum is:

$$
m_\mathrm{new} = \max(m_\mathrm{old},m_\mathrm{tile}).
$$

The old and new exponentials were measured relative to different maxima.
Before adding their sums, both must be expressed relative to
$m_\mathrm{new}$:

$$
\ell_\mathrm{new}
=
e^{m_\mathrm{old}-m_\mathrm{new}}\ell_\mathrm{old}
+
e^{m_\mathrm{tile}-m_\mathrm{new}}\ell_\mathrm{tile}.
$$

This is the essential online-softmax identity. It does not approximate the
missing columns; it changes coordinates before combining two exact partial
sums.

### 5.1 Attach the values

For one query row, maintain an unnormalized output accumulator:

$$
u =
\sum_{\text{scores seen }j} e^{S_j-m}V_j.
$$

When a new tile arrives:

$$
u_\mathrm{new}
=
e^{m_\mathrm{old}-m_\mathrm{new}}u_\mathrm{old}
+
\sum_{j\in\mathrm{tile}} e^{S_j-m_\mathrm{new}}V_j.
$$

After the last tile:

$$
O = \frac{u}{\ell}.
$$

The paper stores a normalized running $O$ rather than this unnormalized $u$.
The two views are equivalent because $u=\ell O$. The unnormalized form is
usually easier to reason about: rescale the old numerator, add the new
numerator, then divide once conceptually.

This also answers whether each transient $P$ tile is already the final
globally normalized probability tile. It is not, and the denominator division
does not happen before the current PV matrix multiply. For the current tile
the kernel first merges the running maximum and forms

$$
\widetilde P_\mathrm{tile}=e^{S_\mathrm{tile}-m_\mathrm{new}}
$$

and immediately consumes
$C_\mathrm{tile}=\widetilde P_\mathrm{tile}V_\mathrm{tile}$. Only after PV
does it merge the old numerator and denominator and divide:

$$
\alpha=e^{m_\mathrm{old}-m_\mathrm{new}},\qquad
u_\mathrm{new}=\alpha u_\mathrm{old}+C_\mathrm{tile},
$$

$$
\ell_\mathrm{new}
=\alpha\ell_\mathrm{old}+\sum\widetilde P_\mathrm{tile},\qquad
O_\mathrm{new}=u_\mathrm{new}/\ell_\mathrm{new}.
$$

Thus $m_\mathrm{new}$ affects the weights before PV, while
$\ell_\mathrm{new}$ normalizes the combined output after PV. A future tile can
change both state values, but FA1 does not return to old probability elements.
It rescales the aggregate old numerator/output state. The mathematical
$(m,\ell,u)$ state and v1.0.9's stored normalized partial $O$ plus LSE are
equivalent sufficient-state representations. After each merge, the stored
partial $O$ is normalized for the key prefix seen so far; only after the last
K/V tile is it the complete attention output.

### 5.2 The invariant

After processing any prefix of key/value tiles:

1. $m$ is the maximum score seen so far;
2. $\ell=\sum_{\text{seen }j}e^{S_j-m}$;
3. $u=\sum_{\text{seen }j}e^{S_j-m}V_j$;
4. $u/\ell$ is attention restricted to exactly the columns seen so far.

The initialization $m=-\infty$, $\ell=0$, and $u=0$ satisfies the empty-prefix
case. The rescaling formulas preserve all four statements when one tile is
added. Induction over the tiles proves that the final result is ordinary
attention over all columns.

### 5.3 A scalar example

Take scores $[1,2,3]$, scalar values $[10,20,30]$, and process scores
$[1,2]$ before $[3]$.

After the first tile:

$$
m_1=2,\quad
\ell_1=e^{-1}+1,\quad
u_1=10e^{-1}+20.
$$

For the second tile, $m_2=3$. Rescale the old state by $e^{2-3}=e^{-1}$:

$$
\ell_2=e^{-1}(e^{-1}+1)+1=e^{-2}+e^{-1}+1,
$$

$$
u_2=e^{-1}(10e^{-1}+20)+30
=10e^{-2}+20e^{-1}+30.
$$

Thus $u_2/\ell_2$ is exactly the stable-softmax weighted sum obtained by
processing all three scores together with global maximum 3.

## 6. FA1 Forward Tiling

The FA1 paper chooses $Q$ row tiles $Q_i$ and $K,V$ column tiles $K_j,V_j$.
Its abstract loop is:

```text
initialize O = 0, l = 0, m = -infinity
for each K/V tile j:
    load K_j and V_j
    for each Q tile i:
        load Q_i and the running O_i, l_i, m_i
        S_ij = Q_i K_j^T
        compute the tile row-max and exponential row-sum
        merge m_i and l_i
        rescale and merge O_i
        write O_i, l_i, m_i
```

At no point does the full $S$ or $P$ matrix need to reside in HBM. A
$B_r\times B_c$ score/probability tile exists transiently on chip.

For the running $N=8192,B_r=16,B_c=128$ example, each head has 512 Q
tiles and **64 sequential K/V-tile steps**. The K/V index
$j=0,\ldots,63$ is the outer-loop tile index; it is unrelated to the four
warps cooperating inside one CTA. At each $(i,j)$ interaction, those warps
jointly process one $16\times128$ score tile, containing 2048 current scores.

The word "running" does not imply that every row's state stays on chip across
the complete K/V traversal. In the paper's K/V-tile-outer loop, one K/V tile
is reused over many Q tiles. The running $O_i,\ell_i,m_i$ for all those Q
tiles cannot remain in the limited SRAM simultaneously, so the pseudocode
loads and writes them in HBM at each K/V-tile step. The state has a stable
logical owner, but its residence can be global memory between steps.

Nor does "global row state" imply a GPU-global synchronization or a
cross-CTA softmax collective. One CTA is the unique owner of each Q-row tile.
For current K/V tile $j$, that owner:

```text
loads O_old[i] and LSE_old[i] from global memory
                                                    # first j means O=0,LSE=-inf
computes current S_ij in acc_p registers
reduces each current S_ij row to tile_max[row]
chooses a[row] = max(LSE_old[row], tile_max[row])
forms old_weight = exp(LSE_old - a)
forms P_tilde = exp(S_ij - a) in registers            # no denominator yet
computes C_tile = P_tilde V_j as four warp partials
combines those partials through CTA shared memory
denom = old_weight + sum(P_tilde)
O_new = (old_weight O_old + C_tile) / denom
LSE_new = a + log(denom)
stores O_new[i] and LSE_new[i] back to global memory
```

Thus the current-tile weights `P_tilde` and current numerator `C_tile` are
not normalized by the new denominator. After the CTA combines the four PV
partials, however, `O_new` divides the old-plus-current numerator by
`denom`. It is already the exact normalized attention output over the K/V
tiles visited so far. It becomes the full attention output only after the
last K/V tile.

Here "global" means that `O_old/LSE_old` summarize every key position already
seen in the row prefix $0,\ldots,j-1$. It does not mean that all 64 K/V tiles
first compute local summaries, synchronize across the grid, and then return
to perform PV. The same row owner advances its state sequentially. In
v1.0.9, intermediate normalized O uses an FP32 `o_tmp` global buffer and LSE
uses FP32 `softmax_lse`; the final step writes O in the requested output dtype.
For $B=1,H=32,N=8192,D=128$, these allocations are approximately 128 MiB and
1 MiB respectively.

This source representation is equivalent to the proof's $(m,\ell,u)$ state,
but it does not save the old row maximum separately. Since
`LSE_old = log(sum_seen(exp(S)))`, the implementation may use
$a=\max(\operatorname{LSE}_{old},\max S_{ij})$ as a numerically safe common
exponent origin. It rescales the old normalized aggregate with
$\exp(\operatorname{LSE}_{old}-a)$ and the current scores with
$\exp(S_{ij}-a)$. Any common origin gives the same ratio in exact arithmetic;
this choice avoids reconstructing the proof representation.

The paper proves:

- the returned value is $O=\operatorname{softmax}(QK^\top)V$ in real
  arithmetic;
- the arithmetic remains $O(N^2d)$;
- additional storage beyond inputs and output is $O(N)$ rather than
  $O(N^2)$;
- for the analyzed SRAM regime, HBM traffic is reduced from the conventional
  attention algorithm's quadratic intermediate traffic.

The speedup comes from doing more useful work while data is on chip and
avoiding HBM round trips. "Flash" is not a different attention definition.

## 7. Add Scale, Mask, And Dropout Back

The omitted features attach at different places:

1. The softmax scale changes the score to $S=\tau QK^\top$, normally with
   $\tau=1/\sqrt d$.
2. A causal or padding mask sets invalid scores to $-\infty$ before softmax.
   Those positions contribute zero after exponentiation. The online-softmax
   invariant is unchanged over the valid positions.
3. Dropout is applied to $P$ after softmax. With keep-mask $Z$ and dropout
   probability $p$, the value path uses
   $P^\mathrm{dropped}=P\circ Z/(1-p)$. Dropout does not change the softmax
   normalizer, but backward must regenerate or save the same $Z$.

Fully masked rows and padding beyond the logical sequence length are special
cases. Their output convention and whether their storage is initialized must
be checked in source and excluded appropriately from bytewise probes.

## 8. Why FA1 Can Differ Bitwise From Ordinary Attention

The correctness proof uses real-number algebra. A GPU runs finite-precision
operations:

- a GEMM has a particular partial-sum tree;
- each tile has a particular max and sum reduction tree;
- online softmax repeatedly rescales and adds partial state;
- a different $B_c$ changes where partial sums are grouped;
- fused and unfused paths round at different boundaries.

Therefore these two claims can both be true:

1. FA1 and ordinary attention compute the same mathematical function;
2. FA1 and ordinary attention do not produce identical floating-point bits.

Likewise, two deterministic FA1 kernels with different tile plans can each
repeat exactly within their own plan and still disagree with each other.

## 9. FA1 v1.0.9 Source Map And Forward Audit

The v1.0.9 code is an optimized implementation, not a literal transcription of
the paper loop. The important source landmarks are:

- `flash_attn/flash_attn_interface.py`: public `deterministic` option and its
  mapping to backward `num_splits`;
- `csrc/flash_attn/src/fmha_fwd_launch_template.h`: forward grid and
  query-row split heuristic;
- `csrc/flash_attn/src/fmha_fprop_kernel_1xN.h`: sequential K/V-tile loop and
  online softmax/output state;
- `csrc/flash_attn/src/fmha/kernel_traits.h`, `fmha/gemm.h`, and the
  `fmha_fwd_hdim*.cu` files: CTA tile shapes, four-warp QK/PV decomposition,
  and head-dimension dispatch;
- `csrc/flash_attn/src/fmha_kernel.h` and `fmha/gmem_tile.h`: packed
  variable-length offsets, 16-byte vector loads, and predication;
- `csrc/flash_attn/src/fmha/utils.h` and `fmha/smem_tile.h`: ordinary
  global-load helpers, register-to-shared stores, shared-memory padding/XOR
  swizzles, and fragment loads;
- `csrc/flash_attn/src/fmha/mask.h` and `fmha/softmax.h`: causal/tail
  validity checks, score masking, and the intra-/cross-warp max/sum reduction;
- `csrc/flash_attn/src/fmha_bwd_launch_template.h`: single-CTA versus
  sequence-K-parallel backward dispatch;
- `csrc/flash_attn/fmha_api.cpp`: rounded backward shapes, FP32 `softmax_d`
  and `dq_tmp` workspaces, sequence-parallel zeroing, and final `dQ` copy;
- `csrc/flash_attn/src/fmha_bwd_hdim*.cu`: backward $B_c$, eight-warp CTA,
  and V-in-register specialization choices;
- `csrc/flash_attn/src/fmha_dgrad_kernel_1xN_loop.h`:
  non-sequence-parallel K/V loop, sequence-parallel `blockIdx.z` entry point,
  `dQ` accumulation, and the shared one-K/V-tile computation body;
- `csrc/flash_attn/src/fmha/gmem_tile.h`: FP32 `atomicAdd` implementation;
- `tests/test_flash_attn.py`: repeated-output race tests and the explicit
  comment that sequence-K parallelism makes `dQ` nondeterministic.

### 9.1 Forward audit at a glance

Boundary contract:

| Input/state | Role in the studied path |
| --- | --- |
| Q, K, V | FP16/BF16 tensors with equal head count; packed-varlen offsets are separate metadata |
| scale and causal mode | applied while forming/masking score tiles |
| dropout RNG | logical-coordinate Philox state that must replay exactly |
| O and LSE | semantic output plus saved normalization state |
| padded LSE / scratch | implementation storage, not automatically valid comparison bytes |

Ownership and order:

| Value | Contributors | Physical owner/combine | Order |
| --- | --- | --- | --- |
| one $O_i$ Q-row tile | all K/V tiles | one CTA owns the complete running state | fixed K/V traversal |
| one $S_{ij}$ fragment | head-dimension chunks | one dependent MMA accumulator chain | fixed in the compiled kernel |
| row max/sum | columns in the current score tile | intra-CTA reduction | fixed tree for the resolved specialization |
| one current-tile $P_{ij}V_j$ | four warp partials over $B_c$ | CTA shared-memory reduction | warp partials are loaded and added in fixed index order |
| distinct heads/batches | none across each other | disjoint CTAs and output regions | scheduler completion order is irrelevant |

State residence:

| State | Residence/lifetime |
| --- | --- |
| global-load fetch vectors | per-thread transit registers until committed to shared memory |
| Q tile | double-buffered shared memory, then distributed per-thread MMA fragment registers |
| K/V tile | one shared-memory region reused V-then-K; fragments retained in registers across owned Q tiles |
| score $S$ and GEMM accumulators | distributed FP32 Tensor Core accumulator/register state |
| probability $P$ | packed FP16/BF16 per-thread fragments consumed directly by PV; optional global return only when requested |
| row reduction scratch | registers plus shared memory |
| current-tile partial O | one distributed FP32 accumulator per warp, then four slices in shared memory |
| partial normalized O and LSE | FP32 global buffers between K/V-tile steps when required |
| final O | requested output dtype in global memory |

Resource and progress ledgers:

- compute: QK MMA, row max/sum/exponential work, and PV MMA remain ordered by
  true data dependencies within one owner;
- data: larger K/V tiles increase reuse and reduce partial-state round trips
  but consume more registers/shared memory;
- concurrency: `batch * heads * num_splits` creates independent CTA work;
- liveness: forward has no cross-CTA producer/consumer ring or shared numeric
  destination, so progress does not depend on another CTA publishing a
  partial-O generation.

The following walkthrough supplies the source evidence and boundary cases
behind this compact record.

### 9.2 Forward execution walkthrough

#### Batch, heads, and packed variable lengths

The forward launch grid is:

```text
blockIdx.x = batch or packed-sequence index
blockIdx.y = attention-head index
blockIdx.z = query-tile split index
```

Ordinary multi-head attention does not change the inner online-softmax
algorithm. Heads are independent attention matrices until the later output
projection, and separate CTAs write disjoint $O[b,h,:,:]$ regions. The number
of heads changes the number of schedulable CTAs and can change the
`num_splits` occupancy decision, but it does not introduce a cross-head
forward reduction. This historical interface expects Q, K, and V to have the
same number of heads; MQA/GQA is outside this FA1 path.

The variable-length interface packs tokens as:

```text
Q: [sum(seqlen_q), heads, d]
K: [sum(seqlen_k), heads, d]
V: [sum(seqlen_k), heads, d]
cu_seqlens_q: [0, len_q_0, len_q_0 + len_q_1, ...]
cu_seqlens_k: [0, len_k_0, len_k_0 + len_k_1, ...]
```

For each batch index, `BlockInfoPadded` loads the packed starting offsets and
the actual Q/K lengths. Global-memory tile loaders use those values to
predicate tail loads, and Q loops stop at the logical sequence end. Varlen
therefore adds address calculation, predicates, early exits, and possible
load imbalance; it does not change the per-head online-softmax recurrence or
create another writer for one logical output row.

`max_seqlen_q` and `max_seqlen_k` still matter. They select one rounded kernel
specialization and temporary-buffer shape for the whole call, while shorter
sequences use predicates. Padded or unused LSE storage is not a semantic
output and must be excluded from bytewise comparisons.

#### Forward `num_splits`

Forward `num_splits` is a scheduling partition over distinct Q-row tiles. It
is not split-K and does not divide one row's softmax normalization among
several CTAs.

Let:

$$
B_r=16,\qquad
T_r=\left\lceil\frac{N_q}{B_r}\right\rceil,\qquad
R=\texttt{num\_splits}.
$$

The launch creates `batch * heads * R` CTAs. Split $r$ owns Q-tile indices:

$$
r,\quad r+R,\quad r+2R,\quad\ldots
$$

For example:

```text
num_splits = 1:
    CTA 0 -> Q tiles 0, 1, 2, 3, ...

num_splits = 4:
    CTA 0 -> Q tiles 0, 4, 8, 12, ...
    CTA 1 -> Q tiles 1, 5, 9, 13, ...
    CTA 2 -> Q tiles 2, 6, 10, 14, ...
    CTA 3 -> Q tiles 3, 7, 11, 15, ...
```

Each split independently loads the K/V tiles needed by its Q rows, so
increasing $R$ creates more CTA parallelism at the cost of repeated K/V HBM
traffic. `num_splits=0` asks the host heuristic to choose $R$. It estimates
the number of CTA waves using:

- `batch * heads * R`;
- the actual GPU's SM count;
- the occupancy API's active-CTAs-per-SM result for this kernel.

It chooses the smallest split count within 95% of the best estimated wave
efficiency. The variable does not reserve $R$ physical SMs for one attention
matrix; it creates $R$ independently schedulable CTA work units, which the GPU
may place on any SM.

More explicitly, define:

$$
\text{slots}=\text{SMs}\times\text{active CTAs/SM},\qquad
\text{work}=BHR,\qquad
\text{waves}=\frac{\text{work}}{\text{slots}}.
$$

Completing the grid needs at least $\lceil\text{waves}\rceil$ CTA waves. The
last wave can be only partly full, so the source estimates average slot
utilization as:

$$
\text{efficiency}
=\frac{\text{work}}
       {\lceil\text{waves}\rceil\text{slots}}
=\frac{\text{waves}}{\lceil\text{waves}\rceil}.
$$

This is a grid-shape utilization estimate, not the execution efficiency of one
CTA. For example, if occupancy returns 2 CTAs/SM on an A100, there are
$108\times2=216$ slots per wave. With $B H=32$, $R=4$ creates 128 CTAs:
`waves=0.593`, so estimated efficiency is 59.3%. $R=13$ creates 416 CTAs:
`waves=1.926`, requiring two waves and yielding $1.926/2=96.3\%$ estimated
efficiency.

For the running $B\cdot H=32$ example on an A100 with 108 SMs, the exact
source heuristic gives the following conditional results when testing
$R=1,\ldots,30$:

| occupancy API result for selected D128 binary | chosen $R$ | whole grid $BHR$ |
| ---: | ---: | ---: |
| 1 CTA/SM | 10 | 320 CTAs |
| 2 CTAs/SM | 13 | 416 CTAs |
| 3 CTAs/SM | 10 | 320 CTAs |
| 4 CTAs/SM | 13 | 416 CTAs |

The 40.625 KiB dynamic-shared request bounds the D128 path to at most roughly
four CTAs/SM by shared capacity alone, while compiled register pressure can
lower it. This CPU-only study does not have the selected binary's ptxas
register count, so it records the conditional table instead of pretending
that `R=4` is the A100 result. `R=4` in the visual walkthrough is only simple
arithmetic: four CTAs per $(b,h)$ and $BHR=128$ CTAs for the whole grid.

The same parameter name has a materially different backward consequence:

| Path | Split dimension | Shared numeric destination? |
| --- | --- | --- |
| forward | different Q-row tiles | no; each split owns different $O_i$ |
| sequence-parallel backward | different K/V tiles | yes; splits contribute to the same `dQ` |

This is why forward splitting preserves unique output ownership while the
historical sequence-parallel backward uses atomic `dQ` accumulation.

#### From the grid to physical SM residence

`blockIdx.z` is the Q-tile **split identity**, not one Q-tile number and not an
SM number. The three layers are:

```text
logical work:
    Q tiles and K/V tiles

CUDA launch:
    CTA (blockIdx.x=b, blockIdx.y=h, blockIdx.z=r)

physical execution:
    the GPU admits each pending CTA onto an SM with enough free resources
```

The source's loop nesting is also easy to misread unless the CTA boundary is
shown:

```text
parallel over CTAs (b, h, r):
    for K/V tile j:                     # device_1xN_loop
        load K_j and V_j
        for Q tile i = r, r+R, r+2R:   # device_1xN_
            load Q_i and prior state[i]
            update state[i] with K_j, V_j
            store state[i]
```

For a non-causal worked example, let:

```text
Nq = 64, Br = 16  -> four Q tiles Q0..Q3
Nk = 512, Bc = 128 -> four K/V tiles KV0..KV3
R = num_splits = 2
```

Then CTA `(b,h,0)` owns Q0 and Q2, while CTA `(b,h,1)` owns Q1 and Q3. For
every K/V-tile iteration, both CTAs separately load that K/V tile and update
only their own Q rows:

```text
CTA z=0: KV0 -> (Q0,Q2), KV1 -> (Q0,Q2), ... KV3 -> (Q0,Q2)
CTA z=1: KV0 -> (Q1,Q3), KV1 -> (Q1,Q3), ... KV3 -> (Q1,Q3)
```

The K/V-outer order lets one CTA retain the current K/V tile on chip while it
visits several owned Q tiles. Different CTAs cannot reuse one another's
ordinary shared-memory allocation, so increasing `num_splits` duplicates K/V
loads. This is the physical source of the parallelism-versus-traffic tradeoff.

Shared memory is physically SRAM in an SM, but it is logically CTA-private.
When the scheduler makes several CTAs resident on one SM, each gets a disjoint
allocation from that SM's shared-memory pool. In the normal CUDA execution
model used here, a resident CTA executes on one SM; `blockIdx.z` does not bind
it to a particular SM.

#### Compile-time specialization, launch-time choice, and CTA admission

FA1 v1.0.9 separates three decisions:

```text
build time:
    compile template families for dtype/mode and head-dimension buckets
    -> ptxas fixes the machine instruction stream and register allocation

host launch time:
    select one compiled kernel from shape/mode
    -> compute the requested dynamic shared-memory size for this launch
    -> query cudaOccupancyMaxActiveBlocksPerMultiprocessor
    -> choose num_splits and grid(batch, heads, num_splits)

GPU execution time:
    admit pending CTAs to SMs as resource slots become available
    -> schedule ready warps from resident CTAs
```

“Precompiled” means precompiled code and resource metadata, not precomputed
attention results. The host is choosing from that code menu; this historical
path is not compiling a new kernel for each input. JIT-based systems can add a
compile/cache step at runtime, but that is a different implementation model.

Concretely, the host C++ validates dtype/device/shape/strides and semantic
modes, rounds sequence bounds, allocates output/LSE/`o_tmp`, chooses the
D32/D64/D128 family from actual $d$, selects the compiled BF16/FP16 and
causal/dropout/return-softmax function, queries occupancy if $R$ is automatic,
sets dynamic shared memory, and launches grid $(B,H,R)$. At SM execution time,
hardware does not read C++ or PTX. Each resident warp has a program counter
into the selected binary's SASS instruction stream; instruction fetch/decode
supplies the next machine instruction, the scoreboard and active mask
determine eligibility, and register/operand paths provide that instruction's
lane operands when a warp is issued.

The occupancy call is made for the selected kernel function with its block size
and dynamic shared-memory request. Its answer is a theoretical maximum active
CTA count per SM for that device and launch configuration. It does not inspect
which SM is momentarily busy, and it does not choose the final CTA-to-SM map.
At admission time, registers, shared memory, threads/warps, the architectural
block limit, and other hardware constraints all have to fit. Allocation
granularity makes the exact calculation subtler than simply dividing headline
capacities.

This also explains why a source branch cannot normally “use fewer registers
this time” and increase residency: register allocation and static shared-memory
requirements belong to the selected compiled kernel, while dynamic shared
memory and launch dimensions belong to the launch. Theoretical occupancy is
only a capacity bound; achieved occupancy and end-to-end performance still
depend on available work, memory stalls, tile reuse, and extra traffic.

#### Where the running state resides

Within one $(Q_i,K_j,V_j)$ tile operation:

- Q/K/V fragments occupy shared memory and registers;
- $S_{ij}$ occupies Tensor Core accumulator registers;
- row-max, row-sum, and exponential state use registers plus
  shared-memory reduction scratch;
- the current $P_{ij}V_j$ result uses registers/shared memory.

Across different K/V tiles, the v1.0.9 source does not keep every Q row's
running state in shared memory:

- it stores partial normalized $O_i$ in an FP32 global `o_tmp` buffer when
  more than one K/V tile is required;
- it stores the combined
  $\operatorname{LSE}_i=m_i+\log\ell_i$ in an FP32 global `softmax_lse`
  buffer instead of storing $m_i$ and $\ell_i$ separately;
- the same CTA reloads, rescales, and merges that state for the next K/V tile;
- the final valid K/V step writes the output in the requested FP16/BF16 dtype.

"Global memory" is the program-visible residence; an access may be served by
cache rather than physical DRAM. The determinism argument uses ownership and
order, not an assumption that the state physically remains in SRAM.

This placement follows FA1's reuse choice:

```text
for K/V tile j:
    keep K_j and V_j on chip
    for many owned Q tiles i:
        load, update, and store state[i]
```

One K/V tile is reused over many Q tiles, but all those Q rows' running state
cannot simultaneously fit on chip.

#### Forward tile shapes in v1.0.9

The compiled kernel trait has the form:

```text
FMHA_kernel_traits<S, D, STEP=16, WARPS_M=1, WARPS_N=4>
```

The template parameter happens to be named `S` in the source, but here it is
the K/V tile width $B_c$ and is unrelated to the runtime `num_splits` $R$ used
in the preceding subsection.

It gives:

- $B_r=\texttt{STEP}=16$ Q rows;
- $B_c=S=128$ or 256 K/V rows;
- a compiled head-dimension bucket $D=32,64,$ or 128;
- four warps, or 128 threads, per CTA.

For both A100/SM80 and H100/SM90, this revision selects:

| Actual head dimension | Compiled $D$ | `max_seqlen_k <= 128` | `max_seqlen_k > 128` |
| --- | ---: | ---: | ---: |
| 8--32, multiple of 8 | 32 | $16\times128$ | $16\times256$ |
| 40--64, multiple of 8 | 64 | $16\times128$ | $16\times256$ |
| 72--128, multiple of 8 | 128 | $16\times128$ | $16\times128$ |

The first dimension in the table is Q rows and the second is K/V rows, so it
is also the shape of the transient score tile. An actual $d$ such as 40
dispatches to the $D=64$ family; predicated loads zero the inactive compiled
columns.

The selection balances several pressures:

1. $B_r=16$ matches the Tensor Core M granularity and limits simultaneous
   score, softmax, and output state.
2. A larger $B_c$ reuses each K/V load over more score columns and reduces the
   number of K/V-loop iterations and partial-O/LSE global round trips.
3. K/V storage grows as $B_cd$, score state grows as $B_rB_c$, and Q/K/V and
   output fragments also consume registers. For $d>64$, $B_c=256$ creates too
   much register/shared-memory pressure, so the source uses 128.
4. If the maximum K length is at most 128, a 256-column tile would only add
   padding and resource use without removing another K/V iteration.

A100 provides up to 164 KB of shared memory per SM, while H100 provides up to
228 KB, but both expose 64K 32-bit registers and at most 64 active warps per
SM. See the NVIDIA
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
and
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).
The FA1 v1.0.9 forward dispatch does not use the architecture to select a
larger H100 tile: it uses head dimension and rounded K length. Architecture
can still change the occupancy result, SM count, and therefore `num_splits`.
Hopper-specific TMA and warp-specialized pipelines are not used here.

#### What the 128 threads do over time

The most useful concrete specialization is:

```text
Br = 16, Bc = 128, D = 64, BF16
CTA = 128 threads = four 32-thread warps
```

There are no dedicated Q-loader, K-loader, V-loader, or compute warps in this
path. All four warps execute the same program and change collective roles over
time:

| Phase | Four-warp decomposition |
| --- | --- |
| Q/K/V global load and shared store | all active threads collectively cover complete tile rows with 16-byte vectors |
| QK | warps partition the 128 score columns; each warp covers 32 K rows |
| softmax | each warp reduces its score-column slice locally, then the CTA combines four partial max/sums through shared scratch |
| next-Q prefetch | the same threads issue Q loads into future-value fetch registers; there is no producer warp |
| PV | the same four warp identities become four $B_c$-reduction slices |
| O epilogue | each warp stores one FP32 partial-O slice; threads load and add the four slices in a fixed order |

This is a phase-dependent logical mapping, not a permanent partition of SM
functional units. A warp that issues an LDG in one phase can issue Tensor Core,
FP/SFU, shared-memory, and global-store instructions in later phases.

##### Collective global-load mapping

For $D=64$ BF16, one row is 128 bytes. A 16-byte vector load therefore uses
eight threads per row. For Q:

```text
thread   0..7   -> Q row 0, eight adjacent 16-byte vectors
thread   8..15  -> Q row 1
...
thread 120..127 -> Q row 15
```

Equivalently, for thread index $t$:

```text
Q row          = floor(t / 8)
Q vector index = t mod 8
```

The $[128,64]$ K and V tiles require eight load rounds. In round $r$:

```text
K/V row        = floor(t / 8) + 16r
K/V vector     = t mod 8
r              = 0..7
```

Thus a D64 thread has one 16-byte Q fetch vector and eight 16-byte fetch
vectors for each of K and V at the initial global-load point. These are transit
registers. After shared staging, `ldsm`/`ldsmt` constructs different,
warp-distributed compute-fragment registers.

##### QK column ownership and softmax combine

The QK trait is logically:

```text
Cta_tile_p = <M=16, N=128, K=64, WARPS_M=1, WARPS_N=4, WARPS_K=1>
```

For this specialization:

```text
warp 0 -> score[:,   0:32]
warp 1 -> score[:,  32:64]
warp 2 -> score[:,  64:96]
warp 3 -> score[:, 96:128]
```

All four warps need the same 16 Q rows, while each consumes its own K-row
slice. No single lane owns one complete score row; accumulator elements are
distributed by the MMA layout.

Each thread first reduces the score elements held in its registers. Small
lane groups combine local values, selected lanes write one partial per
row/warp to shared scratch, and `__syncthreads()` makes all four warp partials
available for the cross-warp max/sum combine. The barrier fixes data
availability; the compiled reduction code fixes the actual tree.

##### PV turns the same warps into reduction slices

The second GEMM trait reinterprets the same 128 threads as:

```text
Cta_tile_o = <M=16, N=64, K=128, WARPS_M=1, WARPS_N=1, WARPS_K=4>
```

The four score-column slices now become four pieces of the PV reduction:

```text
warp 0 -> P[:,   0:32] @ V[  0:32, :]
warp 1 -> P[:,  32:64] @ V[ 32:64, :]
warp 2 -> P[:,  64:96] @ V[ 64:96, :]
warp 3 -> P[:, 96:128] @ V[96:128, :]
```

Each warp produces a distributed FP32 partial contribution to the complete
$[16,64]$ output tile. `Smem_tile_o` stores all four partials, a CTA barrier
makes them visible, and its load path adds partial indices 0, 1, 2, and 3 in
that fixed order. This is an intra-CTA split reduction, not another CTA owner
or an atomic destination.

##### Why partition score columns and accept partial O

The key-position axis changes roles between the two GEMMs. Partition it into
four sets $J_0,\ldots,J_3$, and partition K/V rows the same way:

$$
K=\begin{bmatrix}K_0\\K_1\\K_2\\K_3\end{bmatrix},
\qquad
V=\begin{bmatrix}V_0\\V_1\\V_2\\V_3\end{bmatrix}.
$$

In QK, key position is an output-column axis:

$$
S=QK^\top
=
\begin{bmatrix}
QK_0^\top & QK_1^\top & QK_2^\top & QK_3^\top
\end{bmatrix}.
$$

Each warp can therefore own exact score columns without first combining
partial scores. Softmax still needs the row max/sum across those column
slices, but every individual score element has one warp owner.

After softmax, the same axis is the PV reduction dimension:

$$
O=PV
=
P_0V_0+P_1V_1+P_2V_2+P_3V_3.
$$

Keeping the partition therefore turns the four exact score/P slices into four
partial O contributors. This is deliberate:

```text
warp w already owns P_w and the corresponding V_w fragments
  -> compute P_w V_w locally
  -> reduce four much more structured partial-O slices through shared memory
```

The main alternatives move the communication boundary rather than removing
it:

| Alternative | What it avoids | What it introduces |
| --- | --- | --- |
| keep key-column ownership, as FA1 does | repartitioning transient P | four partial O values per output element |
| repartition PV by output-D columns | partial-O reduction | every output-D owner needs the complete P row, so P must be exchanged/reloaded across warps |
| split QK along head dimension D | score-column partition | partial score values must be combined before the nonlinear softmax |
| partition by Q rows | cross-warp row softmax and partial O | with $B_r=16$, insufficient independent 16-row Tensor Core M tiles unless the CTA grows to more Q rows and more state |

For this $B_r=16$ design, QK has only one natural 16-row M tile but many key
columns, so `WARPS_N=4` supplies useful parallelism without a pre-softmax
split-K reduction. Reusing that ownership in PV avoids materializing or
redistributing the transient P tile. FA1 pays for one CTA-local FP32
partial-O reduction instead.

This is not a mathematical requirement for attention. It is a local
implementation tradeoff among Tensor Core tile granularity, register
locality, shared-memory traffic, CTA size, and occupancy. Later attention
kernels should be compared by asking whether they preserve this partition,
repartition P, enlarge the Q-row tile, or change the Tensor Core operand path.

For large output width, the output-column alternative becomes more attractive.
If two warp groups own disjoint halves of a 512-column output, they compute:

```text
group 0: O[:,   0:256] = P[:, :] @ V[:,   0:256]
group 1: O[:, 256:512] = P[:, :] @ V[:, 256:512]
```

Those results are concatenated/disjointly stored rather than added. This
removes replicated complete-width partial O, but both groups need the complete
P rows, so the implementation must redistribute, reload, or publish P through
a shared producer/consumer boundary. The detailed V4/Gemma/FA4 source signals
and the distinction from cross-CTA SplitKV live in
[the large-head-dimension investigation](https://github.com/rStar-RL/axis-training-dev-tools/blob/master/contexts/megatron/ongoing/yi/determinism_foundations/operators/large_head_dim_attention.md).

##### Why distributed K/V can fit in registers

One $[128,64]$ BF16 K or V tile is 16 KiB. Distributed over 128 threads, that
is 128 payload bytes, or 32 32-bit-register equivalents, per thread per
operand. K plus V therefore has a lower-bound payload of about 64 registers
per thread before Q/P/O fragments, accumulators, pointers, and loop state.

The tile is "in registers" only collectively:

```text
each warp owns its K/V row slice
each lane owns only the fragment elements assigned by the MMA layout
all lanes together represent the complete CTA tile
```

Once every lane has loaded the required V fragments, a barrier allows shared V
addresses to be overwritten by K. Once K fragments are also resident and the
CTA has completed the corresponding reads, the K/V shared region can be reused
as O/softmax scratch. Register liveness enables shared-memory lifetime reuse,
but increases register pressure and can lower resident CTAs per SM. The
aggregate SM register file is large; the per-thread allocation and occupancy
limit still make it finite.

#### Forward memory movement and layout audit

Tile computation alone does not explain FA1 performance. For every later
attention implementation, compare the same six questions:

| Question | FA1 v1.0.9 answer |
| --- | --- |
| 1. How does global memory get accessed? | predicated, 16-byte-per-thread vector loads arranged across complete Q/K/V rows |
| 2. What role do L1 and L2 play? | ordinary hardware-managed caching only; no source-level residency guarantee or persistence policy |
| 3. How does global data reach shared memory? | global load into per-thread fetch registers, then an explicit shared-memory store |
| 4. How is shared memory allocated and laid out? | Q double buffer; K/V and later O/scratch lifetime reuse; padded and XOR-swizzled physical layout |
| 5. How does shared data reach Tensor Cores? | `ldsm`/`ldsmt`-style fragment loads into distributed per-thread registers, then MMA |
| 6. What overlaps, and what synchronizes reuse? | next-Q software prefetch plus fixed CTA barriers; no async-copy producer warp |

These are performance questions first, but they also identify correctness
boundaries. A cache miss should change latency only; an incorrect predicate,
swizzle, barrier, or buffer-generation transition can change values.

##### 1. Global load shape, vector width, and coalescing

For one K/V outer-loop step, the logical tiles cover the complete compiled
head-dimension bucket:

```text
Q_i: [Br, D]
K_j: [Bc, D]
V_j: [Bc, D]
```

The outer loop advances along K/V sequence rows, not along $D$. If the actual
head dimension is $d=40$ and dispatch selects $D=64$, columns 0--39 contain
input values and columns 40--63 are zero-filled by predicates. QK then walks
the complete compiled $D$ in fixed 16-wide MMA chunks; it is not a cross-CTA
split of the reduction dimension. For PV, $B_c$ is the reduction dimension and
$D$ is the output width.

`Gmem_tile_qkv` gives each active thread a `uint4` fetch, or 16 contiguous
bytes. With BF16/FP16, that is eight adjacent elements. Threads assigned to one
row cover adjacent vectors, and the CTA repeats that pattern across tile rows.
For $D=64$, eight threads cover one 128-byte row; 128 CTA threads can cover 16
such rows in one load round. Larger $B_c$ uses several rounds per thread.

For the running $D=128$ BF16 example, one row is 256 bytes and therefore has
sixteen 16-byte chunks. In one round, threads 0--15 fetch chunks 0--15 of row
0, threads 16--31 fetch row 1, and so on through threads 112--127 for row 7.
Here $t=\texttt{threadIdx.x}\in[0,127]$ is one CTA thread identity; its warp
is $\lfloor t/32\rfloor$ and its warp-relative lane is $t\bmod32$. The
variable $r$ is the software global-load loop iteration, `row(t,r)` is the
logical K/V tile row fetched by thread $t$ in round $r$, and `chunk(t,r)` is
that row's 16-byte chunk index. For K/V load round $r=0,\ldots,15$:

```text
row(t,r)   = floor(t / 16) + 8r
chunk(t,r) = t mod 16

t0   -> (row 0,chunk 0), (row 8,chunk 0), ..., (row 120,chunk 0)
t15  -> (row 0,chunk15), (row 8,chunk15), ..., (row 120,chunk15)
t16  -> (row 1,chunk 0), (row 9,chunk 0), ..., (row 121,chunk 0)
t127 -> (row 7,chunk15), (row15,chunk15), ..., (row 127,chunk15)
```

Thus 128 threads fetch eight complete rows per round. Q $[16,128]$ needs two
rounds; K and V $[128,128]$ each need sixteen. A lane does not load one whole
matrix row, and a row is not subsequently split into bank words by the lane:
each issued global load already names one explicit 16-byte chunk.

This layout is designed for coalesced transactions, while the row/head strides,
alignment, varlen tail, and $d<D$ predicate determine which requests are
active. The source establishes the address pattern; final transaction counts
remain a profiler/SASS question.

This step is only the first path arrow:

```text
logical K/V in HBM -> LDG -> lane-private fetch registers
```

No shared-memory bank is involved yet. Its performance vocabulary is global
coalescing, cache service, and HBM/global transactions. Bank conflicts become
relevant only when lanes store to or load from shared-memory physical
addresses, especially on the later shared-to-fragment path.

##### 2. L1 and L2 are opportunities, not owned state

The v1.0.9 forward source uses ordinary typed global loads. It does not express
an L1/L2 cache modifier, an L2 persistence/access-policy window, a preferred
L1/shared carveout, or a cache-resident correctness assumption.

Hardware caching can still help:

- each Q tile is reloaded when the CTA advances to another K/V outer step;
- different forward `num_splits` CTAs reload the same K/V rows for disjoint Q
  rows;
- partial O/LSE state is written to and later read from program-visible global
  memory.

Those patterns create possible L1/L2 reuse, especially L2 reuse across SMs, but
the note must not call them guaranteed cache hits. The implementation's durable
optimization is explicit register/shared-memory reuse. Whether a particular
global access is served by L1, L2, or HBM affects latency and traffic, not the
logical owner or reduction order.

Shared memory and L1 are also not synonyms. On A100/H100 they share an SM-local
physical SRAM capacity pool, but shared memory is an explicitly allocated,
CTA-private address space while L1 is an evictable hardware cache. A CTA's
dynamic shared-memory request is not a direct continuous subtraction from the
headline unified-pool size; the architecture/runtime selects supported
carveout configurations and then admits resident CTA allocations within them.
The reusable hardware model lives in
[`gpu-memory-hierarchy.md`](https://github.com/zyeric/gpu-hardware-notes/blob/main/docs/notes/gpu-memory-hierarchy.md).

##### 3. Global-to-shared uses fetch registers

The source-level path is:

```text
gmem_q/k/v.load()
    ordinary per-thread global vector loads
    -> each thread's fetch_[] registers

gmem_q/k/v.commit(...)
    per-thread shared-memory stores
    -> padded/XOR-swizzled physical addresses in the CTA shared-memory tile
```

This is not a direct bulk copy. Each thread owns only a vector subset; all 128
threads collectively construct the tile. The tile later returns from shared
memory to a different set of per-thread MMA fragment registers, so two
register roles must not be collapsed:

```text
transit/fetch registers: global -> registers -> shared
compute fragments:       shared -> registers -> MMA
```

The swizzle is applied while the global chunk is committed into shared
memory, not as a separate rearrangement pass after a row-major shared tile has
already been written. HBM keeps the ordinary logical tensor layout;
`gmem_k.load()` fetches a logical row chunk into lane-private transit
registers, `gmem_k.commit(smem_k)` computes its swizzled physical shared
address, and the later `smem_k.load()` path uses the matching layout rule.
The value is unchanged; only its shared-memory physical slot differs.

A100/SM80 supports `cp.async`, and H100/SM90 supports TMA, but this historical
source explicitly uses neither. Running the same implementation on H100 does
not automatically turn its LDG-plus-shared-store protocol into a TMA pipeline.
A later implementation must be audited from its selected source/SASS rather
than from hardware capability alone.

##### 4. Shared-memory allocation, lifetime reuse, and swizzle

For BF16/FP16, let $W=4$ be the same four physical warps interpreted as
score-column partitions in QK and reduction-$B_c$ partitions in PV. The default
forward trait uses:

$$
S_Q = 2B_rD\cdot 2
$$

bytes for a double-buffered Q tile,

$$
S_{K/V}=B_cD\cdot 2
$$

bytes for one shared region reused first by V and then by K, and

$$
S_O=B_rD W\cdot 4,\qquad
S_{\text{softmax}}=B_rW\cdot2\cdot4
$$

bytes for FP32 output-reduction and softmax scratch. K/V storage and the later
O/softmax phase have disjoint lifetimes, so the base allocation is:

$$
S_{\text{base}}
=S_Q+\max(S_{K/V},S_O+S_{\text{softmax}}).
$$

If there is more than one K/V outer step, the launch adds a two-buffer FP32
summary scratch:

$$
S_{\text{extra-LSE}}=2B_r\cdot4.
$$

The compiled families therefore request:

| Compiled $D$ | $B_c$ | base shared memory | with multiple K/V steps |
| ---: | ---: | ---: | ---: |
| 32 | 128 | 10.5 KiB | 10.625 KiB |
| 32 | 256 | 18 KiB | 18.125 KiB |
| 64 | 128 | 20.5 KiB | 20.625 KiB |
| 64 | 256 | 36 KiB | 36.125 KiB |
| 128 | 128 | 40.5 KiB | 40.625 KiB |

For example, $B_r=16,B_c=128,D=64$ uses 4 KiB for double-buffered Q,
16 KiB for K/V, 16 KiB for FP32 O scratch, and 0.5 KiB for softmax scratch:

```text
4 KiB + max(16 KiB, 16 KiB + 0.5 KiB) = 20.5 KiB
```

The 16 KiB O term already includes four copies:

```text
one warp partial: 16 rows * 64 columns * 4-byte FP32 = 4 KiB
four warp partials: 4 * 4 KiB = 16 KiB
```

They are four contributors to one CTA-owned current-tile O, not four semantic
outputs. The later shared load reduces the contributors before the CTA writes
the current online-softmax state.

The allocation is smaller than a naive simultaneous Q+K+V sum because both
K/V and later phases reuse addresses. It is larger than the BF16 payload alone
in other places because O/softmax scratch is FP32.

Logical tile shape is still not the physical shared layout. `Smem_tile_*`
rounds/packs rows for supported access widths and applies an XOR-derived
column mapping based on the row. Conceptually:

```text
logical column:  col
physical column: col XOR f(row)
```

This permutation preserves tensor values but spreads the addresses requested
by warp lanes across shared-memory banks and matches the fragment-load pattern.
A shared-memory bank conflict is not an L1 cache-set conflict: shared accesses
have explicit addresses and bank service rules, while L1 performs tag lookup
and cache-line replacement.

The simplest useful A100 teaching model starts from the flat physical
shared-memory byte address after the layout transform. Shared memory is
byte-addressed: one address identifies one byte, while four consecutive byte
addresses make one 32-bit word:

```text
word_index = floor(byte_address / 4)
bank       = word_index mod 32
```

A100 shared memory has 32 banks in this model. Bytes 0--3 map to bank 0,
bytes 4--7 to bank 1, and bytes 124--127 to bank 31; bytes 128--131 wrap back
to bank 0. Therefore one warp instruction reading words 0, 1, 2, ... hits
different banks, while different addresses at words 0, 32, 64, ... contend
for bank 0 and require multiple service rounds. All lanes reading exactly the
same address is a broadcast special case rather than the different-address
conflict case. BF16 packing, vector width, and `ldmatrix` transaction rules add
detail, but they do not change the basic question: which physical shared
addresses does one warp instruction request, and which banks serve them?
One BF16 element is two bytes, so two adjacent BF16 elements occupy one
four-byte bank word in this simplified address view.

A minimal four-bank analogy makes the XOR intent visible. It deliberately
assumes a four-lane instruction, four banks, and one 32-bit word per matrix
cell. Lane 0 loads $a_{00}$, lane 1 loads $a_{10}$, lane 2 loads $a_{20}$,
and lane 3 loads $a_{30}$. No lane loads a complete row or column: each lane
gets one private result register, while the four registers collectively form
the logical column fragment.

Without swizzle, that logical column uses row-major word addresses 0, 4, 8,
and 12, all bank 0 modulo four. Apply the toy mapping
`physical_col = logical_col XOR row` to all 16 values:

```text
logical rows:                 physical shared-memory slots:
[a00 a01 a02 a03]            [a00 a01 a02 a03]
[a10 a11 a12 a13]     ->     [a11 a10 a13 a12]
[a20 a21 a22 a23]            [a22 a23 a20 a21]
[a30 a31 a32 a33]            [a33 a32 a31 a30]
```

The logical-column-0 values now occupy words 0, 5, 10, and 15 and reach toy
banks 0, 1, 2, and 3. Store and load both compute the same physical slot, so
the logical matrix is unchanged. FA1 uses a more specific padding/XOR-derived
layout matched to its `ldsm`/`ldsmt` fragment pattern; the toy formula is
explanatory rather than a literal reconstruction. In real fragment loads,
each lane can receive several packed values rather than the toy's one word,
but the tile remains distributed across lane-private registers.

##### 5. Shared-to-register fragment formation

After shared stores become visible, Q/K use `ldsm`-style loads and V uses a
transposed `ldsmt`-style path to construct each thread's fragment registers.
No thread holds the whole tile; the warp's distributed fragments collectively
describe the MMA operands.

The word "transpose" must be qualified by layer here. For logical
$K,V\in\mathbb{R}^{B_c\times D}$ stored row-major, FA1's global-load stage
uses the same kind of coalesced row chunks for both operands. It does not
materialize either $K^\top$ or $V^\top$ in HBM. The difference appears when
the shared tile is converted into the B-operand fragment required by the two
MMA operations:

| Operand | Mathematical B operand | HBM/global fetch | Shared-to-fragment path |
| --- | --- | --- | --- |
| K | $K^\top[D,B_c]$ in $QK^\top$ | row-major K `[key,D]`, vector loads along D | `ldsm`-style |
| V | $V[B_c,D]$ in $PV$ | row-major V `[key,D]`, vector loads along D | transposed `ldsmt`-style |

The K case works without a physical global transpose because the contiguous
bytes of one row-major K row are exactly one contiguous column of
column-major $K^\top$. A $2\times2$ flattened example makes the equivalence
visible:

```text
row-major K bytes:            [k00, k01, k10, k11]
column-major K^T bytes:       [k00, k01, k10, k11]
```

V has no mathematical transpose in $PV$. Its row-major bytes do not directly
have the column-major B-operand ordering:

```text
row-major V bytes:            [v00, v01, v10, v11]
column-major V bytes:         [v00, v10, v01, v11]
```

The `t` in the V `ldsmt`-style path describes this local
shared-to-distributed-register layout conversion. It does **not** mean that
the high-level operator computes $PV^\top$, nor that FA1 writes a materialized
$V^\top$ tensor to HBM. Thus "K is transposed in the equation" and "V uses a
transposed fragment load" are both true, but they refer to different layers.

This is the register-level contract around Tensor Core work:

```text
swizzled shared tile
  -> ldsm/ldsmt-style warp instruction
  -> A/B operand fragments distributed over 32 lane-private register tuples
  -> mma.sync warp instruction
  -> FP32 accumulator fragments distributed over lane-private registers
```

Register and fragment are not two physical storage tiers. A register is
lane-private storage allocated from the SM register file. "Fetch register,"
"fragment register," and "accumulator register" name different data roles,
layouts, and lifetimes in that same physical register system. A fragment is
the logical contract saying which register elements across all 32 lanes
collectively represent an MMA operand or result; there is no separate
fragment memory.

The lane-private rule is logical, not a claim that hardware gives each lane a
contiguous private SRAM array. A useful SASS-level model is `Rk[lane]`: when
one warp instruction names `R8`, every active lane accesses its own `R8`
value. The backing storage is one SM-level banked/partitioned register file;
adjacent logical register numbers do not prove adjacent physical cells. The
compiler maps source values to machine register numbers, keeps simultaneously
live values distinct, and may reuse the same numbers after an earlier value's
lifetime ends. See the reusable
[`gpu-memory-hierarchy.md` register model](https://github.com/zyeric/gpu-hardware-notes/blob/main/docs/notes/gpu-memory-hierarchy.md#registers)
for the generic hardware account.

In this FA1 path, QK writes distributed FP32 `acc_p` values, and ordinary
CUDA-core/SFU instructions subsequently read each lane's own accumulator
registers for mask, max, exponential, and sum work. Cross-lane or cross-warp
row reductions still need shuffle/shared communication. The resulting
softmax-derived weights are converted and packed into FP16/BF16 `frag_p`
operand-register tuples for PV. `acc_p -> frag_p` is therefore a
dtype/layout/role transition inside the register system, not a copy from
"accumulator memory" into a separate "fragment memory"; physical register
reuse remains a compiler/liveness decision.

An MMA operand or result is therefore not one matrix row per lane. The complete
logical matrix fragment exists only across all participating lanes. Q/K or
P/V fragment registers feed Tensor Core matrix multiplies. QK produces FP32
`acc_p` registers containing distributed S elements; ordinary CUDA-core/SFU
instructions then perform masking, row max, exponential, and sum. Softmax
packs the current weights into `frag_p` registers for PV, whose FP32 result is
distributed `acc_o`. Lanes cannot arbitrarily address another lane's private
registers; shuffle instructions, shared memory, or the hardware-defined
collective instruction mapping provide the required exchange.

`mma.sync` is a warp-wide instruction: all 32 lanes must execute the same MMA
with the same qualifiers, and their register tuples collectively supply the
matrices. It should not be modeled as one lane calling a matrix unit, or as
one logical instruction permanently bound to one physical Tensor Core. The
A100 SM has four third-generation Tensor Cores, together rated for 1024 dense
FP16/FP32 FMAs per clock; BF16 runs at the same mixed-precision rate. The SM
routes/decomposes warp-level MMA work through those Tensor Core pipelines;
exact issue and latency are microarchitectural/SASS questions beyond the
logical PTX contract.

For a concrete SM80 BF16 example, PTX defines:

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

as one warp computing $D=A B+C$, where $A$ is $16\times16$ row-major, $B$ is
$16\times8$ column-major, and $C/D$ are $16\times8$. Per lane:

- the A fragment has eight BF16 values packed into four 32-bit registers;
- the B fragment has four BF16 values packed into two 32-bit registers;
- the C/D fragment has four FP32 accumulator registers.

Let `group = lane_id >> 2` and `thread_in_group = lane_id % 4`. The PTX ISA
defines exact coordinate formulas from those values. For lane 0:

| Fragment | Logical coordinates in lane 0 registers |
| --- | --- |
| A | `(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)` |
| B | `(0,0),(1,0),(8,0),(9,0)` |
| C/D | `(0,0),(0,1),(8,0),(8,1)` |

Thus one lane's fragment is neither a complete row nor a complete column, and
need not be one long contiguous slice. In QK, interpret $A=Q[\text{query},k]$
and $B=K^\top[k,\text{key}]$. Lane 0's example B fragment corresponds to
original-K values `K[key=0,d={0,1,8,9}]`: several short pairs from one K row
for this micro-MMA, not the full row.

MMA does not take a runtime transpose boolean. The PTX opcode's `.row.col`
qualifiers are part of its static layout contract. The shared layout and
`ldsm`/`ldsmt`-style load path arrange values into the required register
slots. This lets QK consume K as column-major $K^\top$ without materializing
a separate transposed K tensor in HBM.

The K/V-shared region is reused in this order:

```text
commit V to shared
  -> CTA barrier
  -> load all V fragments into registers
  -> CTA barrier before overwrite

commit K to the same shared region
  -> CTA barrier
  -> load K fragments into registers
```

V fragments remain in registers while the CTA traverses its owned Q tiles for
this K/V outer step. K fragments are likewise reused by the QK operations. For
$D=64$, QK consumes four dependent $K=16$ MMA chunks. If actual $d=40$, the
last 24 compiled columns are predicated zeros; this is still the fixed $D=64$
instruction path, not a special $D=40$ reduction.

For the running $B_c=D=128$ BF16 case, one K row is 256 bytes, or sixteen
16-byte global-load chunks. One 128-thread load round covers eight rows; sixteen
rounds collectively fetch the 32 KiB K tile. Each chunk first occupies its
thread's transit/fetch registers, is committed to a swizzled shared address,
and is later loaded into a different per-thread MMA fragment mapping. The
complete current tile exists only collectively across the CTA's private
register fragments.

The default shared-K/V path has this exact lifetime interpretation:

```text
global K and V -> separate fetch registers
V -> shared K/V region -> all frag_v registers
barrier
K -> the same shared K/V addresses -> all frag_k registers
barrier / completed K fragment loads
the same shared addresses -> partial-O reduction scratch
```

Thus “K overwrites V” does mean that V has first been read from shared memory
into registers and its shared addresses are then reused by K. After K is also
resident in fragments, both `frag_k` and `frag_v` remain register-resident for
the current K/V outer step even though their former shared region is available
for `smem_o`.

The score/probability path should not be added to shared-memory accounting as
another full $B_r\times B_c$ tile. QK creates $S$ in distributed FP32
`acc_p`/softmax register state. Row max and sum use register reductions plus a
small shared scratch for cross-warp partials. Softmax then packs $P$ into
FP16/BF16 `frag_p` registers consumed directly by PV. Only the optional
`Return_softmax` path writes those values to global memory. PV produces FP32
`acc_o` registers; the four warp-local partial outputs then use `smem_o` for
the fixed CTA-local combine.

##### 6. Software pipeline, synchronization, and reuse

FA1 v1.0.9 is fused and software-pipelined, but it is not an FA3-style
producer/consumer warp pipeline. For one K/V tile, the flow is approximately:

```text
ordinary global-load K_j, V_j, Q_i into fetch registers
commit V_j and Q_i to shared
barrier
load V_j and Q_i fragments

barrier before K overwrites the shared K/V region
commit K_j
barrier
load K_j fragments

compute Q_i K_j^T
issue ordinary global prefetch of Q_{i+1} into fetch registers
softmax current scores
commit Q_{i+1} into the alternate Q shared buffer
compute P_ij V_j
store/reduce O scratch through shared memory
barrier before reading the completed O reduction
store current partial O/LSE
switch the shared-memory Q buffer

compute Q_{i+1} K_j^T
prefetch Q_{i+2}
...
```

There are two distinct Q pipelines:

1. across Q tiles, shared buffer A holds $Q_i$ while future $Q_{i+1}$ is
   loaded into fetch registers and then committed to shared buffer B;
2. inside one QK, two Q fragment-register slots ping-pong over successive
   16-wide $D$ chunks, loading the next fragment before issuing MMA on the
   current fragment.

For $D=64$, the second pipeline is conceptually:

```text
load Q fragment D[0:16]
load D[16:32] -> MMA D[0:16]
load D[32:48] -> MMA D[16:32]
load D[48:64] -> MMA D[32:48]
                  MMA D[48:64]
```

The steady-state cross-Q source order is:

```text
finish this warp's Q_i K_j^T

if Q_{i+1} exists:
    select alternate shared write buffer
    issue ordinary LDG Q_{i+1} -> per-thread fetch registers

mask / reduce / exponentiate current score
commit Q_{i+1} fetch registers -> alternate shared buffer
compute P_ij V_j
store four warp partial-O slices
CTA barrier
reduce/read O and finish the current state

select alternate shared read buffer
load the first Q_{i+1} fragment
```

The same threads hold current score/P/O state and future-Q fetch state at the
same time. There is no explicit `cp.async`, no dedicated producer warp, and no
pipeline that removes the true dependency between `QK -> softmax -> PV` or
between successive online-softmax states.

An ordinary Q LDG can be outstanding while later instructions that do not read
its destination registers execute. The SM scoreboard records those fetch
registers as pending; a later shared store that consumes them cannot issue
until the load completes. Warp scheduling can also run another ready warp
while one warp waits. This is operand-readiness and latency-hiding machinery,
not cross-thread synchronization.

The barriers protect concrete lifetime transitions: shared stores before
fragment loads, V reads before K overwrites the shared K/V region, and O
partial stores before their reduction/readback. The O-reduction barrier also
ensures every participating thread has committed the next Q tile before the
CTA switches the shared read generation. A scoreboard cannot prove that
another warp has finished its shared store; that requires the CTA barrier.

The general hardware distinction between a warp PC, instruction issue,
scoreboard dependencies, and synchronization lives in
[`gpu-execution-model.md`](https://github.com/zyeric/gpu-hardware-notes/blob/main/docs/notes/gpu-execution-model.md).
Q double buffering permits some load/compute overlap, but ordinary LDG results
still occupy fetch registers and the same CTA warps perform both movement and
compute.

For determinism, swizzle and staging are value-preserving address/data-movement
rewrites under correct predicates and barriers. They do not add another
floating-point contributor. A missing wait or premature shared-buffer reuse
would instead be a correctness race; replacing this pipeline with `cp.async` or
TMA creates new barrier/stage proof obligations even if the attention
mathematics is unchanged.

#### Comparison frame for FA2 and later implementations

The point of this FA1 baseline is not to pre-claim how FA2, FA3, or FA4 work.
Each later pass should fill the same comparison table from its own source:

| Dimension | Question to resolve |
| --- | --- |
| work owner | Does one CTA still own a complete output-row reduction, and how many logical tiles does it visit? |
| global access | What is the vectorization, predication, coalescing, and expected cache reuse? |
| copy primitive | ordinary LDG/STS, `cp.async`, TMA, or another mechanism? |
| copy actors | all compute threads, selected producer threads, or dedicated producer warps? |
| shared layout | allocation formula, stage count, swizzle, bank-conflict constraints, and lifetime overlays? |
| Tensor Core feed | shared-to-register fragments, WGMMA/UMMA path, or another operand route? |
| accumulator residence | registers, shared scratch, TMEM, or global workspace? |
| overlap protocol | which stages overlap and which barrier/counter proves safe reuse? |
| numeric order | did the new decomposition introduce split reductions, atomics, or another combine order? |

This makes “new hardware optimization” falsifiable: identify the changed
movement/compute mechanism, its resource tradeoff, and whether it preserves the
same ownership and reduction proof.

#### Tensor Core reduction over head dimension

For one score element:

$$
S_{ij}=\sum_{r=0}^{d-1}Q_{ir}K_{jr}.
$$

The source uses warp-level Tensor Core instructions such as
`mma.sync.aligned.m16n8k16`. If $d=64$, the logical accumulator chain is:

```text
acc = 0
acc = MMA(Q[0:16],  K[0:16],  acc)
acc = MMA(Q[16:32], K[16:32], acc)
acc = MMA(Q[32:48], K[32:48], acc)
acc = MMA(Q[48:64], K[48:64], acc)
```

The later MMA reads the previous accumulator, so the same output fragment has
a fixed dependency order in one compiled kernel. Warp scheduling may
interleave unrelated instructions or pause this warp, but it cannot execute a
dependent accumulator update before its predecessor.

The forward configuration has `WARPS_K=1`: warps divide the score tile mainly
along key columns, not by producing independent partial sums along $d$ for the
same $S_{ij}$. There is no cross-warp or cross-CTA split-K combine for QK.
Actual head dimensions that are not multiples of 16, such as 40, use a larger
compiled bucket and predicated zero-filled columns.

This supports bitwise repetition for the QK reduction under one fixed
architecture and kernel binary. It does not prove equality with an unfused
reference, across a different tile/compiler artifact, or between SM80 and
SM90; the internal Tensor Core arithmetic and association belong to that
implementation envelope.

#### Causal skipping and load balance

For standard causal self-attention, key position $k$ is valid for query
position $q$ only when $k\le q$. At tile level:

| | early K tile | middle K tile | late K tile |
| --- | --- | --- | --- |
| early Q tile | diagonal/masked | skip | skip |
| middle Q tile | full | diagonal/masked | skip |
| late Q tile | full | full | diagonal/masked |

The source avoids completely invalid tile pairs by starting the Q loop for K
tile $j$ near:

$$
\text{first Q tile}=\frac{jB_c}{B_r}.
$$

It therefore does not load early Q/partial-O state that cannot use the current
K/V tile. Within a tile that crosses the diagonal, it still performs the tile
QK operation and sets positions with $k>q$ to $-\infty$ before softmax.
Tail predicates separately cover actual variable lengths. A K/V tile is still
loaded if later owned Q rows need it; it can be skipped entirely only when no
owned valid Q row remains or the K tile is beyond the logical K length.

Causal work is lighter for early Q tiles and heavier for late Q tiles. The
round-robin `num_splits` mapping deliberately mixes them. In a simplified
example:

```text
Q-tile work: 1 1 1 1 | 2 2 2 2 | 3 3 3 3 | 4 4 4 4

split 0 owns 0,4,8,12  -> 1+2+3+4
split 1 owns 1,5,9,13  -> 1+2+3+4
split 2 owns 2,6,10,14 -> 1+2+3+4
split 3 owns 3,7,11,15 -> 1+2+3+4
```

Real ratios do not always align perfectly, but interleaving is much better
than assigning one CTA a contiguous early range and another a contiguous late
range. The occupancy heuristic creates enough CTA waves when beneficial, and
the GPU schedules another pending CTA when one finishes. Residual imbalance
can still matter for small `batch * heads`, the last CTA wave, or a varlen
batch with highly skewed sequence lengths. FA1 has no global persistent work
queue or cost-aware work stealing.

#### Why the complete forward repeats

For one resolved kernel, ordinary forward has:

1. one CTA owner for each output Q-row tile;
2. a fixed K/V-tile traversal for that owner;
3. fixed intra-CTA QK, row-max, row-sum, and PV reduction trees;
4. no inter-CTA floating-point combine into the same $O_i$;
5. fixed logical-coordinate RNG state when dropout is replayed.

The running state can be stored in global memory without creating a race:
only its owner CTA reads and updates that address, in program order. Other
CTAs may run in any scheduler order because they write disjoint Q rows.

Sequential accumulation is therefore part of the explanation, but it is not
sufficient by itself. The decisive combination is unique ownership plus a
fixed local reduction order. A design in which several CTAs each accumulated
sequentially and then atomically combined their partial $O_i$ would not inherit
this conclusion.

The source race test repeatedly exact-compares forward outputs. This supports
fixed-plan repeatability, while the ownership and reduction structure explain
why there is no visible cross-CTA numeric race.

## 10. Backward, Derived Before It Is Tiled

Start from one batch element and one head:

$$
Q\in\mathbb{R}^{N_q\times d},\quad
K,V\in\mathbb{R}^{N_k\times d},\quad
R=QK^\top,\quad
S=\tau R,\quad
P=\operatorname{softmax}(S),\quad
O=PV.
$$

Let $G=dO=\partial\phi/\partial O$ be the incoming gradient. This note reserves
$dS$ for the derivative with respect to the **scaled** score $S$. The scale
$\tau$ therefore appears when the gradient crosses from $S$ back to
$QK^\top$.

The backward is five equations plus their reduction axes. The CUDA kernel
changes their grouping and residence, not their mathematical contract.

### 10.1 The two output-side matrix products: `dV` and `dP`

Since $O=PV$:

$$
dV=P^\top G,
$$

or elementwise:

$$
dV_{jr}=\sum_{i=1}^{N_q}P_{ij}G_{ir}.
$$

Thus one $dV$ row belongs naturally to one K/V row, but it receives
contributions from **all query rows**.

Again from $O=PV$:

$$
dP=GV^\top,\qquad
dP_{ij}=\sum_{r=1}^{d}G_{ir}V_{jr}.
$$

This is a different kind of reduction: each scalar $dP_{ij}$ reduces over the
head dimension $d$. In the implementation it is produced by a Tensor Core
GEMM; it is not a sequence-wide atomic reduction.

### 10.2 The softmax Jacobian and the row scalar `D`

For one query row $i$, write $p=P_i$ and $g=dP_i$. The softmax Jacobian is:

$$
\frac{\partial p}{\partial S_i}
=\operatorname{diag}(p)-pp^\top.
$$

Multiplying it by $g$ gives:

$$
dS_i
=\left(\operatorname{diag}(p)-pp^\top\right)g
=p\odot\left(g-(p^\top g)\mathbf{1}\right).
$$

Define the one-scalar-per-query-row correction:

$$
D_i=P_i^\top dP_i=\sum_jP_{ij}dP_{ij}.
$$

Then:

$$
dS_{ij}=P_{ij}(dP_{ij}-D_i).
$$

At first $D_i$ appears to require the full probability row. Associativity over
real numbers exposes a cheaper identity:

$$
\begin{aligned}
D_i
&=\sum_jP_{ij}(G_i^\top V_j) \\
&=G_i^\top\left(\sum_jP_{ij}V_j\right) \\
&=G_i^\top O_i.
\end{aligned}
$$

Forward already saved $O_i$, so backward computes:

$$
D_i=\sum_{r=1}^{d}G_{ir}O_{ir}
$$

with an $O(d)$ row dot product. This has two implementation consequences:

1. no full $N_q\times N_k$ probability matrix is needed merely to compute
   the softmax correction;
2. every K/V-tile worker needs the same $D_i$, so the scheduling mode must
   decide who computes it and how the other work obtains it.

### 10.3 Crossing back through the score GEMM: `dQ` and `dK`

Since $S=\tau QK^\top$:

$$
dQ=\tau\,dS\,K,\qquad
dK=\tau\,dS^\top Q.
$$

Elementwise:

$$
dQ_{ir}=\tau\sum_{j=1}^{N_k}dS_{ij}K_{jr},
\qquad
dK_{jr}=\tau\sum_{i=1}^{N_q}dS_{ij}Q_{ir}.
$$

The critical ownership asymmetry is now visible:

- one $dQ_i$ row reduces over **all K/V rows**;
- one $dK_j$ row reduces over **all query rows**;
- one $dV_j$ row also reduces over **all query rows**.

`dQ`, `dK`, and `dV` are all many-to-one sums, but the kernel does not have to
parallelize or combine them in the same way.

### 10.4 What is saved, recomputed, and transient

Forward saves the row maximum $m_i$ and exponential sum $\ell_i$, or
equivalently the log-sum-exp:

$$
\operatorname{LSE}_i=m_i+\log\ell_i.
$$

For a recomputed score tile:

$$
P_{ij}=e^{S_{ij}-\operatorname{LSE}_i}
=\frac{e^{S_{ij}-m_i}}{\ell_i}.
$$

Backward can therefore recreate one $P_{ij}$ tile from $Q_i,K_j$ and the
linear-sized saved normalization state. It trades recomputation for avoiding
an $N\times N$ saved tensor.

For the v1.0.9 path, the useful state ledger is:

| State | Backward treatment |
| --- | --- |
| $Q,K,V$ | saved input tensors, reloaded tile by tile |
| $O$ | saved forward output; used with $G$ to compute $D_i$ |
| LSE | saved FP32 row normalization state; used to reconstruct $P_{ij}$ |
| dropout Philox state | saved only when dropout must be replayed |
| $G=dO$ | incoming gradient |
| $D_i$ | recomputed once, then stored in FP32 `softmax_d` |
| $S_{ij},P_{ij},dP_{ij},dS_{ij}$ | recomputed/transient tile state |
| full $P$ or full score matrix | never materialized in HBM |
| `dq_tmp` | FP32 implementation workspace when more than one K/V tile exists |

The source folds the final factor $\tau$ into the `dQ`/`dK` epilogues. That is
an implementation placement choice; mathematically $dS$ above is still the
gradient with respect to the scaled score.

### 10.5 Reduction ledger

Before looking at CUDA, write every sum and its contributor axis:

| Quantity | Formula | Reduction axis | Later owner/combine |
| --- | --- | --- | --- |
| $dP_{ij}$ | $\sum_rG_{ir}V_{jr}$ | head dimension $d$ | one CTA-local GEMM fragment |
| $D_i$ | $\sum_rG_{ir}O_{ir}$ | head dimension $d$ | one row reduction, stored FP32 |
| $dS_{ij}$ | $P_{ij}(dP_{ij}-D_i)$ | no new reduction | pointwise tile operation |
| $dQ_{ir}$ | $\tau\sum_jdS_{ij}K_{jr}$ | all K rows | serial K-tile combine or cross-CTA atomic |
| $dK_{jr}$ | $\tau\sum_idS_{ij}Q_{ir}$ | all Q rows | one K-tile owner loops Q tiles |
| $dV_{jr}$ | $\sum_iP_{ij}G_{ir}$ | all Q rows | one K/V-tile owner loops Q tiles |

This table predicts the final determinism result. The only sequence-wide sum
whose ownership changes between the two v1.0.9 launch modes is `dQ`.

## 11. From The Paper Loop To A Concrete Ownership Example

The paper's abstract backward loop is:

```text
initialize dQ, dK, dV to zero
for each K/V tile j:
    keep local dK_j and dV_j accumulators on chip
    for each Q tile i:
        load Q_i, O_i, G_i, LSE_i
        recompute S_ij and P_ij
        D_i = rowsum(G_i * O_i)
        dP_ij = G_i V_j^T
        dS_ij = P_ij * (dP_ij - D_i)
        dV_j += P_ij^T G_i
        dK_j += tau * dS_ij^T Q_i
        dQ_i += tau * dS_ij K_j
    write dK_j and dV_j once
```

This exposes three different ownership shapes:

| Output | Contributions | Natural owner in the paper loop |
| --- | --- | --- |
| $dQ_i$ | all K/V tiles $j$ | query row tile $i$, revisited across $j$ |
| $dK_j$ | all query tiles $i$ | key tile $j$ |
| $dV_j$ | all query tiles $i$ | value tile $j$ |

The serial pseudocode gives $dQ_i$ a fixed $j=1,2,\ldots,T_c$ update order.
It does not prove that a parallel CUDA implementation preserves that order.
Parallelizing different $j$ tiles is attractive for occupancy, but then
several CTAs produce partial contributions to the same $dQ_i$.

Use two Q tiles $I_0,I_1$ and two K/V tiles $J_0,J_1$:

```text
work for J0:
    visit I0, I1 in order
    accumulate the complete dK_J0 and dV_J0
    produce partial dQ_I0 <- J0 and dQ_I1 <- J0

work for J1:
    visit I0, I1 in order
    accumulate the complete dK_J1 and dV_J1
    produce partial dQ_I0 <- J1 and dQ_I1 <- J1
```

The contribution matrix is:

| Work unit | Owns final `dK`/`dV` | Contributes to `dQ_I0` | Contributes to `dQ_I1` |
| --- | --- | --- | --- |
| $J_0$ | `dK_J0`, `dV_J0` | yes | yes |
| $J_1$ | `dK_J1`, `dV_J1` | yes | yes |

There is exactly one owner for every final `dK`/`dV` tile. There are two
contributors for each `dQ` tile. A serial implementation can make one CTA do
$J_0$ and then $J_1$; a sequence-parallel implementation can make two CTAs do
them concurrently. The local arithmetic for one table row can remain the same
while the `dQ` combine protocol changes.

This is the bridge from the math to the determinism question:

> Find every many-to-one sum, then ask whether one worker owns the complete
> sum or whether independent workers combine partial sums into a shared
> destination.

## 12. FA1 v1.0.9 Backward Audit

The source uses one shared computation body for one K/V tile. The two backward
modes differ mainly in whether one CTA invokes that body repeatedly for all
K/V tiles or many CTAs each invoke it once.

### 12.1 Dispatch, tile families, and workspace

The Python interface has a slightly misleadingly named integer:

- `num_splits=0`: let the host heuristic choose between the two modes;
- `num_splits=1`: disable sequence-K parallelism;
- any `num_splits>1`: select the sequence-parallel implementation.

For backward, a value such as 4 does **not** mean “launch exactly four K/V
splits.” Once sequence parallelism is selected, the launch grid contains one
CTA for every K/V tile:

```text
non-sequence-parallel grid:
    (batch, heads, 1)

sequence-parallel main grid:
    (batch, heads, ceil(Nk / Bc))
```

Keep the requested mode selector separate from the resolved fan-out:

| Requested backward `num_splits` | Meaning in v1.0.9 | Actual main-kernel z dimension |
| --- | --- | --- |
| `0` | run the host heuristic | either `1` or the full K/V-tile count |
| `1` | select the serial-K mode | `1` |
| `2`, `3`, `4`, ... | select the same sequence-K kernel | `ceil(Nk / Bc)`, independent of the requested value |

Thus the exact user-supplied value above one is not a tunable degree of
parallelism in this backward implementation. The **resolved K/V-tile count**
still matters: it determines the CTA count, the number of potential atomic
contributors to a `dQ` element, contention, and the space of possible
floating-point arrival orders. For the running $N_k=8192,B_c=128$ example,
passing either `num_splits=2` or `num_splits=8` launches 64 K/V-tile CTAs per
batch/head. Causality can make some query rows receive fewer active
contributions, but it does not turn the requested integer into an exact CTA
count.

The default heuristic compares CTA-wave utilization for `batch * heads`
against the parallel K/V-tile count. It discounts sequence parallelism because
that mode must zero an FP32 `dq_tmp`, run a separate $D_i$ kernel, and copy/cast
the final `dq_tmp` to `dQ`. Causal attention receives a smaller discount
because K/V-tile parallelism also improves load balance.

The public `deterministic=True` maps directly to backward `num_splits=1`.
It does not ask CUDA for a general deterministic mode; it disables this one
known cross-CTA `dQ` accumulation path.

The compiled v1.0.9 backward family uses:

```text
Br = 16 query rows
Bc = 128 or 256 K/V rows
CTA = 8 warps = 256 threads
```

This does not contradict the four-warp forward configuration. A CTA has no
universal warp count: the selected forward kernel launches 128 threads, while
the separately compiled backward kernel launches 256. For $B_c=128$, a useful
logical comparison is:

```text
forward:  4 warps * 32 K/V rows per warp = 128 rows
backward: 8 warps * 16 K/V rows per warp = 128 rows
```

The backward body has more independent matrix products and output-gradient
state to produce, so this implementation exposes more intra-CTA parallelism.
The extra warps do not increase the SM's issue bandwidth; they give its warp
schedulers more resident work to choose from and change the per-CTA resource
footprint. On A100/H100, the architectural limit of 64 resident warps per SM
would permit at most 16 four-warp CTAs or eight eight-warp CTAs by the warp
limit alone. Registers, shared memory, CTA slots, and allocation granularity
usually impose a lower bound. The complete admission and issue distinction is
explained in
[`gpu-execution-model.md`](https://github.com/zyeric/gpu-hardware-notes/blob/main/docs/notes/gpu-execution-model.md#resident-capacity-is-not-simultaneous-issue-width).

#### One eight-warp CTA still resides on one SM

CUDA assigns an ordinary CTA as one unit to one SM. All eight backward warps,
their registers, CTA-private shared-memory allocation, and CTA barriers remain
on that SM; the CTA is not divided across two SMs. This is required for
ordinary shared-memory addressing and `__syncthreads()` to have CTA scope.
Other SMs run other batch/head/K-tile CTAs.

An A100 SM is physically divided into four **SMSPs**, conventionally expanded
as Streaming Multiprocessor Sub-Partitions. Each sub-partition contains one
warp scheduler and one third-generation Tensor Core pipeline; the full SM has
four Tensor Cores. The
[A100 architecture description](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
documents four Tensor Cores per SM and one per SM partition. SMSP placement is
a microarchitectural model, not a CUDA indexing contract: source code cannot
select a stable `warp_id -> SMSP` mapping.

A useful, deliberately non-contractual picture for one eight-warp CTA is:

```text
one A100 SM
    SMSP 0 / Tensor Core pipeline 0 <- some two resident warps
    SMSP 1 / Tensor Core pipeline 1 <- some two resident warps
    SMSP 2 / Tensor Core pipeline 2 <- some two resident warps
    SMSP 3 / Tensor Core pipeline 3 <- some two resident warps
```

This picture separates three quantities:

1. **resident:** all eight warp contexts can be live on the SM;
2. **issued:** the four SMSP schedulers choose eligible warps, subject to
   issue slots and the target execution pipeline accepting work;
3. **in flight:** several previously issued MMA or memory operations can
   overlap in pipelined hardware.

Eight resident warps therefore do not mean eight MMA instructions start in
one cycle. Nor does one warp-level `mma.sync` mean that the CTA acquires a new
physical Tensor Core. A resident warp is serviced by the execution pipelines
of its SM sub-partition, and those pipelines are time-multiplexed among ready
warps. The exact initiation interval depends on the compiled MMA instruction,
shape, and dtype.

Warp instructions issue in program order, but completion is pipelined. After
an MMA request is accepted, the same warp may issue later instructions whose
source operands do not depend on the MMA result. A later MMA that reads and
writes the same accumulator cannot proceed until the scoreboard marks that
accumulator ready. During that wait the scheduler can choose another ready
warp:

```text
warp 0: issue MMA(acc0) -> issue independent address/load work
                        -> dependent MMA(acc0) waits on scoreboard
warp 4:                                      ready work can issue instead
```

This is instruction- and warp-level latency hiding, not an explicit
`cp.async` protocol. SM80 `mma.sync` remains a warp-scoped instruction whose
operand and accumulator fragments are distributed across 32 lanes, as
summarized by the
[CUTLASS warp-level MMA guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html).

All eight warps participate in movement and compute. Their main logical
decompositions are:

| Phase | Eight-warp decomposition |
| --- | --- |
| QK and $dP=GV^\top$ | split the $B_c$ score columns across eight warps |
| $dQ=dS K$ | eight partial reductions over disjoint $B_c$ slices |
| $dK=dS^\top Q$, $dV=P^\top G$ | split the $B_c$ output rows across eight warps; each warp accumulates its owned rows across Q tiles |

For $B_c=128$, a warp covers 16 K/V rows; for $B_c=256$, it covers 32. The
backward K/V tile selection is:

| Compiled $D$ | Rounded K length 128 | Rounded K length 256 or larger on A100/H100 |
| ---: | ---: | ---: |
| 32 | $B_c=128$ | $B_c=256$ |
| 64 | $B_c=128$ | $B_c=256$ |
| 128 | $B_c=128$ | $B_c=128$ |

The D64/$B_c=256$ A100/H100 specialization deliberately does **not** retain V
fragments across the whole Q loop. The source comment says this avoids
register spilling at the cost of more shared memory. This is a concrete
example of tile choice being constrained by register liveness, not just by
whether the BF16 payload fits in aggregate SM storage.

The dynamic shared-memory request is:

$$
S_{\rm bwd}
=2S_Q+S_V(1+\mathbf{1}_{V\notin regs})+S_{dQ}+2S_S,
$$

where the first term is the already-double-buffered Q layout used separately
for Q and $G$, $S_{dQ}$ stores eight FP32 warp partials, and the two score-sized
buffers hold/transposes $P$ and $dS$. For these supported FP16/BF16 shapes:

$$
\begin{aligned}
S_Q &= 2B_rD\cdot2, \\
S_V &= B_cD\cdot2, \\
S_{dQ} &= 8B_rD\cdot4, \\
S_S &= B_rB_c\cdot2
\end{aligned}
$$

bytes. For example, $D=64,B_c=128$ uses:

```text
Q and G double buffers:  2 * 4 KiB =  8 KiB
shared K/V region:                    16 KiB
eight FP32 dQ partials:               32 KiB
P and dS transpose buffers: 2 * 4 =   8 KiB
                                      --------
                                      64 KiB
```

The source-selected families request:

| $D$ | $B_c$ | V retained in registers | dynamic shared memory |
| ---: | ---: | --- | ---: |
| 32 | 128 | yes | 36 KiB |
| 32 | 256 | yes | 52 KiB |
| 64 | 128 | yes | 64 KiB |
| 64 | 256 on A100/H100 | no | 120 KiB |
| 128 | 128 | no | 152 KiB |

These numbers describe shared-memory allocation only. Register fragments,
FP32 Tensor Core accumulators, pointers, predicates, and fetch registers are
additional resources fixed by compilation.

A mechanical `D=512,B_c=128` substitution, which is not a supported FA1
specialization, would require 456 KiB even if V remained in registers and
584 KiB if another shared V tile were needed. The eight FP32
`[B_r,D]` `dQ` copies alone consume 256 KiB. This counterfactual explains why
modern large-d backward kernels must change the tile, partial count,
accumulator residence, or multi-stage ownership rather than simply recompiling
the FA1 layout. It does not estimate a DeepSeek-V4 production kernel.

### 12.2 One K/V-tile computation body

Fix one K/V tile $J$. The common `one_iter` body keeps `dK_J` and `dV_J`
accumulators live in FP32 registers while it visits all valid 16-row Q tiles
in increasing order:

```text
load fixed K_J and V_J
initialize FP32 acc_dK_J and acc_dV_J

for Q tile I in increasing sequence order:
    load Q_I, G_I, LSE_I, D_I
    recompute score S_IJ and probability P_IJ
    dP_IJ = G_I V_J^T
    dS_IJ = P_IJ * (dP_IJ - D_I)

    local_dQ_I_from_J = tau * dS_IJ K_J
    acc_dV_J += P_IJ^T G_I
    acc_dK_J += tau * dS_IJ^T Q_I

    reduce/write this J work unit's dQ contribution

write final dK_J and dV_J once
```

The code interleaves these operations more aggressively than the pseudocode,
but the dependency graph is the same.

#### State residence

| State | Residence and lifetime in the common body |
| --- | --- |
| current/future Q and $G$ | separate double-buffered shared tiles; ordinary global prefetch uses per-thread fetch registers |
| current K/V tile | global-to-register-to-shared staging; fragments then feed the GEMMs; V may remain in registers in selected specializations |
| LSE and $D$ | FP32 global row buffers, loaded for the current Q tile |
| score/$P$/$dP$/$dS$ | distributed per-thread registers; $P$ and $dS$ also pass through score-shaped shared buffers for transposed consumption |
| eight local `dQ` partials | FP32 accumulator registers, then eight slices in `smem_dq` |
| `dK_J`,`dV_J` | FP32 register accumulators kept across all Q tiles |
| cross-K-tile `dQ` state | FP32 global `dq_tmp` |

`P` and `dS` need two orientations. QK and $dP$ naturally produce a
$[B_r,B_c]$ warp layout, while $dV=P^\top G$ and $dK=dS^\top Q$ consume the
transpose. The source stores them through swizzled shared-memory tiles and
reloads the transposed fragments instead of materializing a global tensor.

#### One-tile dataflow and layout transitions

For one current Q tile $I$ and fixed K/V tile $J$, the dependency graph is:

```text
Q_I, K_J
    -> S_IJ [query,key] score fragments
       QK: [B_r,D] x [D,B_c] -> [B_r,B_c]
       reduce D; warps partition output key columns B_c
    -> P_IJ = exp(S_IJ - LSE_I)
       |\
       | \-> store P tile in shared
       |      -> transposed fragment reload P_IJ^T [key,query]
       |      -> acc_dV_J += P_IJ^T G_I
       |         [B_c,B_r] x [B_r,D] -> [B_c,D]
       |         reduce B_r; warps partition output key rows B_c
       |
G_I, V_J -> dP_IJ = G_I V_J^T
                [B_r,D] x [D,B_c] -> [B_r,B_c]
                reduce D; warps partition output key columns B_c
                \
P_IJ, dP_IJ, D_I -> dS_IJ = P_IJ * (dP_IJ - D_I)
                       |\
                       | \-> original [query,key] fragments
                       |      -> local dQ_I_from_J = dS_IJ K_J
                       |         [B_r,B_c] x [B_c,D] -> [B_r,D]
                       |         reduce B_c; warps partition B_c into eight partial dQ tiles
                       |
                       \-> store dS tile in shared
                              -> transposed fragment reload dS_IJ^T [key,query]
                              -> acc_dK_J += dS_IJ^T Q_I
                                 [B_c,B_r] x [B_r,D] -> [B_c,D]
                                 reduce B_r; warps partition output key rows B_c
```

This graph separates mathematical values from their instruction-facing
layouts:

| Stage value | Semantic coordinates | Residence/layout at that stage | Consumer |
| --- | --- | --- | --- |
| $S_{IJ}$ | `[query,key]` | FP32 QK accumulator fragments distributed across lanes | pointwise mask and LSE-based reconstruction of $P$ |
| $P_{IJ}$ | `[query,key]` | score-oriented registers; current tile is also published through swizzled shared memory | pointwise $dS$ path and transpose handoff |
| $P_{IJ}^\top$ | `[key,query]` | newly loaded MMA operand fragments after the shared-memory layout conversion | $dV=P^\top G$ |
| $dP_{IJ}$ | `[query,key]` | FP32 `G V^T` accumulator fragments | pointwise formation of $dS$ |
| $dS_{IJ}$ | `[query,key]` | score-oriented registers; current tile is also stored to shared memory | $dQ=dS K$ and transpose handoff |
| $dS_{IJ}^\top$ | `[key,query]` | newly loaded MMA operand fragments after the shared-memory layout conversion | $dK=dS^\top Q$ |

The five matrix products use three distinct partition patterns:

| Product | Tile shapes | Mathematical reduction axis | Eight-warp partition in this FA1 backward specialization | CTA-local consequence |
| --- | --- | --- | --- | --- |
| $S=QK^\top$ | $[B_r,D][D,B_c]\to[B_r,B_c]$ | $D$ | partition the output's $B_c$ key columns | each warp produces a disjoint score-column slice |
| $dP=GV^\top$ | $[B_r,D][D,B_c]\to[B_r,B_c]$ | $D$ | partition the output's $B_c$ key columns | each warp produces a disjoint $dP$-column slice |
| $dQ=dS K$ | $[B_r,B_c][B_c,D]\to[B_r,D]$ | $B_c$ | partition the **reduction** axis $B_c$ | eight warps produce eight partial $[B_r,D]$ `dQ` tiles that must be combined |
| $dV=P^\top G$ | $[B_c,B_r][B_r,D]\to[B_c,D]$ | $B_r$ | partition the output's $B_c$ key rows | each final $dV$ row has one warp owner |
| $dK=dS^\top Q$ | $[B_c,B_r][B_r,D]\to[B_c,D]$ | $B_r$ | partition the output's $B_c$ key rows | each final $dK$ row has one warp owner |

For example, when $B_c=128$, each of the eight warps owns a 16-key slice.
That slice means output columns for QK/$dP$, a reduction slice for $dQ$, and
output rows for $dK$/$dV$. These are logical warp-level partitions, not a
claim that one whole tile product maps to one hardware instruction. Each warp
issues multiple MMA instructions over instruction-sized microtiles, and the
lane/register fragment mapping still depends on the compiled specialization.

`P^T` and `dS^T` are not additional full tensors and never become
sequence-sized global intermediates. They are the same current-tile values
reassigned to different lane/register tuples so a later MMA sees the operand
shape and role it requires.

The transposed consumers occur late in one Q-tile iteration because they
depend on the shared store, visibility synchronization, and transposed reload.
They are not postponed until after all Q tiles:

```text
for each I:
    form P_IJ and dS_IJ
    update current dQ_I contribution
    acc_dV_J += transposed P_IJ contribution
    acc_dK_J += transposed dS_IJ contribution

after the complete I loop:
    store final dV_J and dK_J once
```

#### Within one Q-tile iteration

The approximate source order is:

```text
1.  load G fragment
2.  recompute Q K^T, apply mask, reconstruct P from LSE
3.  store P to shared for its transposed dV use
4.  prefetch the next Q tile
5.  initialize dP registers with -D, then accumulate G V^T
6.  multiply pointwise by P to obtain dS
7.  prefetch the next G and, when needed, O
8.  store dS to shared for its transposed dK use
9.  compute eight warp-local dQ = dS K partials
10. store the eight partials to smem_dq
11. reload transposed P and accumulate dV
12. commit the prefetched next Q into the alternate shared buffer
13. reload/reduce smem_dq and combine it with any earlier K-tile dQ
14. reload transposed dS and accumulate dK
15. commit the prefetched next G and switch shared-buffer generations
```

The next-Q/next-$G$ prefetch is the same style as forward: ordinary loads can
be outstanding while independent instructions issue, and their destination
registers are tracked by the scoreboard. It is not a dedicated producer warp
or an asynchronous-copy pipeline.

CTA barriers protect reuse of `smem_dq` and the double-buffered Q/$G$ storage.
Warp synchronization protects score-tile transpose handoffs where only
warp-level communication is needed. The fixed source instructions and barriers
define the local reduction order for one resolved specialization.

#### Why `dQ` has eight shared-memory partials

For the $dQ$ GEMM, $B_c$ is the reduction dimension. The eight warps each
multiply their own K/V-row slice:

```text
warp 0: dS[:, J0] @ K[J0, :]
warp 1: dS[:, J1] @ K[J1, :]
...
warp 7: dS[:, J7] @ K[J7, :]
```

They therefore produce eight partial versions of the same $[B_r,D]$ `dQ`
tile. `smem_dq` stores all eight FP32 copies. Its reload helper adds partial
indices 0 through 7 in a fixed loop. This is a CTA-local fixed-order reduction,
not the nondeterministic atomic reduction discussed below.

`dK` and `dV` make the opposite ownership choice: the eight warps partition
the output K/V rows, so a warp can keep its own output fragments in registers
and add Q tiles in increasing order. The CTA writes each final `dK_J`/`dV_J`
tile once.

#### Why forward uses four warps while backward uses eight

More warps are a tradeoff, not free parallelism. For `B_c=128`, the pinned
forward divides score columns into four 32-column slices. Those four warps
already provide one ready-warp candidate per A100 SMSP in the simplest
physical picture. Increasing forward to eight warps would create eight
16-column slices, but every current K/V tile would then have:

- eight rather than four partial row maxima and exponential sums to combine;
- eight rather than four partial `O` tiles because the same K-column axis is
  the reduction dimension of `P @ V`;
- more cross-warp communication, synchronization, and partial-state storage.

For the representative `B_r=16,D=128` case, one FP32 partial `O` is 8 KiB.
The source-selected four-warp layout uses 32 KiB for four copies; a mechanical
eight-warp version would need 64 KiB before accounting for the additional
softmax partials and reduction work. The smaller forward CTA can instead gain
more ready warps from another independent CTA when register/shared-memory
residency and the forward `num_splits` grid permit it.

Backward has a different benefit/cost ratio:

| Property per current `(I,J)` tile | Four-warp forward | Eight-warp backward |
| --- | --- | --- |
| row normalization | cross-warp max and sum for online softmax | reconstruct `P=exp(S-LSE)` pointwise from saved LSE; no new row max/sum |
| main reduction partial | four partial `O` tiles must combine | eight partial `dQ` tiles must combine |
| other output work | no additional gradient outputs | `dK`/`dV` rows are partitioned and single-owned by warps |
| matrix products | QK and PV | QK, dP, dQ, dK, and dV |
| available second CTA in running D128 case | smaller forward footprint may permit more CTAs | 152 KiB shared request strongly limits the SM to one CTA |

This does **not** mean backward has fewer barriers. It has shared-memory
handoffs for the `P` and `dS` transpose layouts, Q/$G$ buffer generations, and
`smem_dq` reuse. The narrower claim is:

> Relative to its much larger amount of matrix work, backward has fewer
> cross-warp **numeric reductions**: saved LSE removes the forward max/sum
> combine, and `dK`/`dV` use disjoint output-row ownership. Only `dQ` needs the
> eight-way numeric combine.

The source proves the warp counts, ownership, barriers, and storage sizes. The
claim that this is the optimal 4-versus-8 performance point is a design
inference; it would require compiled variants and A100 profiling to establish
empirically.

### 12.3 Backward with `num_splits == 1`

The non-sequence-parallel kernel launches one CTA per batch/head and calls the
common body for K/V tile indices:

```text
J = 0, 1, 2, ..., ceil(Nk / Bc) - 1
```

in that fixed order. This is more specific than saying merely that atomics are
absent:

```text
CTA(b,h):
    J0 body: loop Q tiles -> write first FP32 dQ state
    J1 body: loop Q tiles -> load prior dQ, add J1 partials, write FP32 state
    ...
    last J body: load prior dQ, add last partials, scale/cast final dQ
```

Within K tile $J$ the eight warp partials are added by ascending partial index.
Across K tiles, the same CTA first loads the prior FP32 `dq_tmp` value and then
adds the current tile's eight partials. The effective source-level order is:

```text
J0: partial[0], partial[1], ..., partial[7]
J1: prior_J0, partial[0], ..., partial[7]
J2: prior_J1, partial[0], ..., partial[7]
...
```

Only that CTA accesses the batch/head's in-progress `dQ` as an owner. The
global temporary is needed because one CTA cannot keep every Q row's `dQ`
live on chip while the outer loop advances over K/V tiles; global residence
does not imply shared ownership.

The first K/V-tile iteration also computes $D_i=G_i^\top O_i$ while visiting Q
tiles and stores it in `softmax_d`. Later K/V iterations reload the value.
There is no need for a separate dot-product kernel because the single CTA
itself establishes “compute first, consume later” program order.

For fixed binary, inputs, shape, mask/RNG state, and specialization, this path
has:

1. one CTA owner for the cross-K `dQ` sum;
2. a fixed K/V-tile traversal;
3. fixed eight-warp local combines;
4. one K/V-tile owner for each complete `dK` and `dV` sum;
5. no cross-CTA floating-point destination.

That is the source-level reason `num_splits=1` is expected to repeat
bitwise—not a generic guarantee that every GEMM or every CUDA instruction is
deterministic under an arbitrarily changed implementation envelope.

### 12.4 What changes with `num_splits > 1`

Sequence parallelism changes the outer ownership:

```text
CTA blockIdx.z = J0 -> run the common body once for J0
CTA blockIdx.z = J1 -> run the common body once for J1
...
```

The inner Q-tile order, local formulas, eight-warp `dQ` reduction, and
one-owner `dK_J`/`dV_J` epilogues are reused. Three surrounding protocols must
change.

#### Change 1: compute `D` before all K/V-tile CTAs

No CTA-local barrier can make one K/V-tile CTA publish $D_i$ and safely release
all other CTAs in the same launch. v1.0.9 therefore first launches a separate
grid over Q-row tiles to compute:

$$
D_i=G_i^\top O_i
$$

and store `softmax_d`. CUDA stream ordering makes those writes visible before
the sequence-parallel main kernel.

#### Change 2: zero a shared FP32 `dq_tmp`

Every K/V-tile CTA produces a contribution to every visited `dQ_I`. The host
creates or zeroes the FP32 `dq_tmp` before launching those CTAs. This is why
sequence parallelism has setup overhead even though the mathematical backward
is unchanged.

#### Change 3: atomically add each K/V-tile contribution

After the common body reduces one CTA's eight warp partials in fixed order, it
uses FP32 `atomicAdd` on `dq_tmp`:

```text
dq_tmp[I] += local_dQ_from_J   # atomic per element
```

The exact source path is:

```text
eight warp-local FP32 dQ accumulator fragments
    -> eight CTA-private slices in smem_dq
    -> barrier and fixed partial-index reduction
    -> dq_out uint4 values in per-thread registers
    -> Gmem_tile_o<Cta_tile_dq, 4>::atomic_add
    -> four scalar FP32 atomicAdd operations per active uint4
    -> global FP32 dq_tmp
```

`smem_dq` and `dq_tmp` solve different combine problems. The former is
CTA-private shared memory used to reduce the eight warp partials for one
K/V-tile work unit. The latter is a program-visible global-memory tensor used
to combine contributions from independent K/V-tile CTAs. Its global address
may be served or retained by the L2 path while the kernel runs; “global
address” does not mean that every atomic read-modify-write must make a fresh
round trip to HBM.

The wrapper does not issue one 128-bit vector atomic. A `uint4` carries four
FP32 values bitwise, and the unrolled inner loop invokes scalar
`atomicAdd(ptr + k, value[k])` for `k=0,1,2,3`. For the representative
`D=128` path:

```text
one FP32 dQ row                         = 128 * 4 B = 512 B
one active thread's store/atomic chunk = 16 B = four FP32 elements
threads needed per row                 = 512 / 16 = 32

thread 0  -> local q row 0, d[0:4]
thread 1  -> local q row 0, d[4:8]
...
thread 31 -> local q row 0, d[124:128]
thread 32 -> local q row 1, d[0:4]
```

Thus one 256-thread CTA covers eight rows per store group; the 16-row `dQ`
tile uses two such groups. Boundary-row and head-dimension predicates disable
invalid threads. Within one full warp, lanes normally target distinct scalar
`dQ` words. The same-word contention comes from different `blockIdx.z=J`
CTAs for the same batch/head reaching the same Q row and feature element.

##### Why `dq_tmp` is separate from the returned `dQ`

The public gradient has the input dtype, normally FP16 or BF16, while
`dq_tmp` is FP32. The source wrapper statically requires four-byte elements
for this atomic path. Accumulating every K/V-tile contribution directly into
a 16-bit `dQ` would round after every contribution and would also depend on
architecture-specific 16-bit atomic support. Instead the implementation:

```text
zero FP32 dq_tmp
    -> atomically accumulate all K/V-tile contributions in FP32
    -> stream-ordered dq.copy_(dq_tmp)
    -> one final FP32-to-output-dtype conversion
```

The separate workspace is therefore related to the atomic plan, but an atomic
API does not intrinsically require a second buffer. If the requested output
were FP32, a compatible implementation could use it directly after zeroing.
The serial path also uses FP32 global temporary state when it must spill the
cross-K prefix for all Q rows; there it remains single-owned and is updated by
ordinary loads/stores rather than by a cross-CTA atomic.

##### Issue, completion, and contention

CUDA C++ `atomicAdd` returns the old value, but this call site discards it. A
compiler can therefore lower it to a no-destination reduction instruction
such as PTX/SASS `red` rather than a returning `atom`; the exact choice remains
a compiled-binary fact to confirm with `cuobjdump` or `nvdisasm`. PTX defines
[`atom`](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)
with a destination register for the old value and
[`red`](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red)
without one.

A no-return reduction is asynchronous only in a limited, informal sense:
after the warp successfully issues the request, there is no old-value
register dependency preventing later independent instructions from issuing.
The memory/L2 atomic path can complete the operation while warp schedulers run
other ready work. It is not an explicit `cp.async`-style protocol:

- a full load/store or atomic request queue creates backpressure;
- updates to one FP32 word still have to be serialized;
- dependent memory ordering would require the appropriate synchronization;
- kernel completion must drain the updates before the stream-ordered final
  copy can consume `dq_tmp`.

Therefore an atomic is neither a whole-buffer lock nor free fire-and-forget
work. Different words can progress in parallel, and the many Q rows, features,
heads, and staggered CTA compute provide latency-hiding opportunities. The
same-word serialization remains a real throughput cost.

Public measurements support the qualitative model but do not provide a
portable FA1/A100 constant. NVIDIA's
[warp-aggregated atomic example](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)
shows a single heavily contended counter becoming a bottleneck as the atomic
rate rises. A 2025
[modern-GPU atomic microbenchmark study](https://publikationen.bibliothek.kit.edu/1000188141/170539767)
finds fast uncontended operations and nontrivial hardware handling for small
request groups, followed by scaling costs as contention grows. Its consumer
GPUs, operation mix, and single-variable patterns are not direct performance
numbers for FA1 on A100.

For this representative shape, sequence-K parallelism expands the main grid:

```text
serial:   B * H                  = 1 * 32      = 32 CTAs
parallel: B * H * (Nk / Bc)      = 1 * 32 * 64 = 2048 CTAs
```

That is the reason to accept the atomic protocol: 32 resource-heavy CTAs
cannot occupy an A100's 108 SMs, while 2048 K/V-tile work units expose many
waves and improve causal load balance. An alternative would store one complete
partial `dQ` per K/V tile and run a deterministic reduction kernel, but that
requires roughly `num_k_tiles * size(dQ)` workspace plus extra global writes,
reads, and launch work. FA1 instead keeps one full-size FP32 `dq_tmp` and
accepts contention and an unspecified cross-CTA floating-point order.

The atomic makes each read-modify-write indivisible, so contributions are not
lost. It does not impose a K/V-tile arrival order. Extend the ownership example
to three K/V tiles. One run can serialize:

```text
zero -> contribution(J0) -> contribution(J1) -> contribution(J2)
```

while another can serialize:

```text
zero -> contribution(J1) -> contribution(J2) -> contribution(J0)
```

These correspond to different parenthesizations of three FP32 partials.
FP32 addition is not associative, so correct executions can differ in their
low bits. With exactly two ordinary finite partials, swapping the two additions
normally gives the same value because addition is commutative; the first clear
non-associativity case needs three partials. With only one K/V tile there is
only one atomic writer and therefore no cross-CTA numeric race at all. The
source-level warning is consequently:

> Do not assume sequence-parallel `dQ` is bitwise repeatable when multiple,
> especially three or more, K/V-tile CTAs contribute to the same element.

In causal attention the contributor count is row-dependent: early query rows
may see only the first K/V tile, while later rows see several.

Once all main-kernel CTAs finish, a stream-ordered copy/cast converts `dq_tmp`
into the requested `dQ` dtype.

The exact boundary is:

| Property | Preserved from the common body? |
| --- | --- |
| Q-tile traversal inside one K/V work unit | yes, fixed |
| eight-warp local `dQ` combine | yes, fixed |
| `dK_J` and `dV_J` single-owner accumulation | yes |
| order among different K/V contributions to one `dQ` element | **no**; atomic arrival order |

This is why scheduler freedom is numerically relevant only for `dQ` in this
path. `dK_J` and `dV_J` still have disjoint K/V-tile owners; changing the order
in which those CTAs finish does not change which arithmetic chain writes one
output element.

The v1.0.9 tests encode the same diagnosis: forward, `dK`, and `dV` are
exact-compared across repeats, while sequence-parallel `dQ` is allowed a small
arithmetic tolerance and is annotated as nondeterministic.

### 12.5 Dropout state

With dropout, deterministic replay also requires the same random mask. The
source stores Philox RNG state in forward and constructs offsets from logical
batch, head, lane, and 16-by-16 attention-block coordinates so backward can
regenerate the same mask even though its traversal differs.

This solves forward/backward mask correspondence within the implementation.
It does not excuse a caller from replaying the same seed and offset. An
earlier random operation that consumes a different number of counters can
change attention dropout even if the attention kernel itself is unchanged.

## 13. The FA1 Determinism Verdict

For the pinned v1.0.9 CUDA implementation:

| Path | Source-level verdict | Reason |
| --- | --- | --- |
| forward, fixed RNG state | expected deterministic within a fixed implementation envelope | one CTA owns each output row tile |
| backward, `deterministic=True` / `num_splits=1` | expected deterministic within a fixed implementation envelope | avoids sequence-K-parallel `dQ` atomic accumulation |
| default backward when heuristic selects `num_splits>1` | `dQ` is not bitwise safe to assume deterministic when multiple K-tile CTAs contribute | K-tile CTAs atomically add partial `dQ` in an unspecified arrival order |
| `dK`, `dV` in the studied sequence-parallel path | repeated exactly in the historical race test | K/V-tile ownership avoids the corresponding shared atomic destination |

The implementation envelope still includes the commit, CUDA entry point,
GPU architecture, head dimension, dtype, sequence metadata, mask/dropout,
tile specialization, compiler, binary, runtime, and stream/state discipline.

This verdict is intentionally narrower than:

- equality with unfused reference attention;
- equality between two tile sizes;
- equality between two GPU architectures;
- a claim about FA2 or later implementations;
- a claim about every Triton, ROCm, inference, sparse, or KV-cache path.

## 14. Reusable Proof Obligations Learned From FA1

For a new attention kernel, ask these questions in order:

1. **Semantic function:** Is it dense softmax attention, or has masking,
   sparsity, approximation, quantization, or state mutation changed the
   function?
2. **Saved/recomputed state:** Which of $O$, max, sum, LSE, RNG state, mask,
   and probabilities are saved, and which are recomputed?
3. **Output ownership:** Which CTA owns each element or tile of $O,dQ,dK,dV$?
4. **Many-to-one edges:** Which output needs contributions from several Q,
   K/V, head, batch, or split tiles?
5. **Combine mechanism:** Does one owner accumulate in a fixed loop, or do
   workers use atomics, reduce-add instructions, split workspaces, semaphores,
   or a later reduction kernel?
6. **Order source:** If the combine is deterministic, what concretely fixes
   the order: program order, a static reduction tree, split index, or an
   explicit synchronization protocol?
7. **Intra-CTA reductions:** Are GEMM, row-max, row-sum, and dot-product trees
   fixed for the resolved kernel?
8. **Compile/launch/admission split:** Which resource requirements and
   instruction choices are fixed in the binary, which launch parameters are
   selected by the host, and which placement decisions belong to the GPU?
9. **Dispatch identity:** Can shape, architecture, SM count, autotuning,
   workspace, or compiler state select another plan?
10. **Mutable state:** Are Philox seed/offset, dropout masks, streams, or
    workspaces replayed and initialized identically?
11. **Atomic lowering:** What address space and scalar words are updated, what
    local reduction precedes the atomic, is the old value consumed, and is the
    destination a separate accumulation dtype/workspace?
12. **Atomic performance:** Which workers actually reach the same word, can
    independent requests hide latency, and when do queue backpressure and
    same-address serialization outweigh the extra CTA waves?
13. **Claim boundary:** Is the claim same-process repeatability, fresh-process
    repeatability, clean-build repeatability, or cross-hardware equality?

An atomic instruction is evidence of a possible multi-writer sum, not an
automatic verdict by itself. It matters whether multiple concurrent workers
can reach the same address. Conversely, the absence of an atomic is not a
complete proof if buffers alias or another kernel performs an unordered
reduction.

## 15. Counterexamples That Prevent Overclaiming

1. A deterministic kernel and a reference kernel can disagree bitwise because
   deterministic does not mean reference-identical.
2. An atomic path can happen to return the same bits in ten trials. Repeated
   equality is evidence, not proof that the race is impossible.
3. A unique output owner can contain nondeterministic behavior through an
   external library or mutable state; ownership is necessary evidence, not the
   entire environment contract.
4. Reusing the same random seed is insufficient if the Philox offset entering
   attention changed.
5. `deterministic=True` can remove one race while a different backend,
   preprocessing reduction, or grouped-head combine remains outside the
   analyzed path.
6. A fixed source commit can produce a different reduction plan after a
   compiler, architecture, or generated-artifact change.

## 16. Future FA1 Component Probe

The smallest useful GPU probe should capture one resolved FA1 call, not merely
compare final training loss.

### 16.1 `GuardSpec`

Fingerprint before the call:

- $Q,K,V$ bytes, strides, dtype, shape, device, and storage aliasing;
- packed-sequence offsets and maximum lengths;
- softmax scale, causal/mask mode, dropout probability;
- Philox seed and offset or saved RNG state;
- public `deterministic` value and resolved backward `num_splits`;
- source commit, extension binary hash, GPU architecture, SM count, CUDA
  runtime/driver, PyTorch version, and stream identity.

Fingerprint after forward:

- $O$;
- valid LSE rows;
- returned RNG state;
- optional test-only probability/dropout-mask representation.

Fingerprint around backward:

- incoming $dO$;
- $dQ,dK,dV$;
- every FP32 accumulation workspace before and after use.

Padding and intentionally uninitialized regions must be excluded from equality
checks; otherwise allocator garbage can be mistaken for a numerical
divergence.

### 16.2 Probe matrix

1. Repeat identical captured inputs at least 100 times with dropout disabled.
2. Repeat with dropout enabled while restoring exact RNG state.
3. Compare `deterministic=False` and `True`.
4. Include a long-enough K sequence for the heuristic to choose
   sequence-parallel backward.
5. Cover causal and noncausal, fixed and variable lengths, FP16 and BF16, and
   representative head dimensions separately.
6. Report exact byte hashes first; use max/mean error only to characterize an
   already-detected mismatch.
7. Repeat in a fresh process and after a clean extension rebuild, while
   recording artifact identity.

The expected diagnostic signature for the pinned sequence-parallel backward
is: identical inputs and forward output, then the first exact mismatch at
`dQ`, while `dK` and `dV` remain equal for the studied path.

## 17. Model-Level Guard Placement

In a real model, place guards at:

1. the projected/position-encoded $Q,K,V$ and attention metadata;
2. attention $O$ and LSE/RNG state;
3. incoming $dO$;
4. returned $dQ,dK,dV$ before projection-gradient combines.

Interpret the first mismatch:

- if $Q,K,V$ already differ, attention is downstream damage;
- if inputs and RNG match but $O$ differs, inspect forward dispatch,
  artifact, ownership, and uninitialized state;
- if forward matches and `dQ` first differs, inspect K-tile parallelism and
  shared accumulation;
- if `dK/dV` first differ, inspect their owner mapping and any head/group
  reduction rather than assuming the `dQ` explanation applies.

This is more precise than waiting for loss or gradient norm to diverge, and it
turns the source-level ownership hypothesis into a falsifiable runtime check.

## 18. Stop Line And Next Questions

The completed forward track has answered:

- why tiled online softmax is exact attention;
- how multi-head and packed-varlen indexing wrap the same per-head algorithm;
- where forward state resides, how Q-row `num_splits` schedules work, and how
  causal skipping is balanced;
- how logical Q/K/V tiles map to CTAs and then to physical SM residence;
- what build-time specialization, host-side occupancy/launch choice, and
  hardware CTA admission each decide;
- which A100/H100 forward tiles the historical source selects, how it
  pipelines them, and why the Tensor Core $d$ reduction repeats;
- how its global access, cache opportunity, register-mediated copy, shared
  layout/swizzle, fragment formation, and synchronization answer the
  six-question memory-movement audit;
- how the same 128 threads partition QK columns, cooperate on softmax, become
  four PV reduction slices, and combine their partial O through shared memory;
- why preserving key-column ownership avoids redistributing transient P but
  turns that axis into a four-way PV reduction;
- why next-Q prefetch uses same-thread future-value registers, scoreboarding,
  alternate shared buffers, and CTA barriers rather than a dedicated producer
  warp;
- which fixed comparison dimensions should be reused for FA2 and later
  implementations instead of describing each generation with unrelated
  vocabulary;

The completed backward source pass has answered:

- how the five backward equations induce distinct head-dimension, Q-sequence,
  and K-sequence reductions;
- why $D_i=dO_i^\top O_i$ lets backward recompute $P$ from linear-sized saved
  state;
- how an eight-warp CTA processes one K/V tile, transposes $P/dS$ through
  shared memory, and accumulates `dK`/`dV` across Q tiles;
- why all eight warps still reside on one SM, how they enlarge the ready-work
  pool over four A100 SMSPs/Tensor Core pipelines, and why resident, issued,
  and in-flight MMA counts differ;
- why warp instruction issue remains in program order while independent work
  can overlap an in-flight MMA and true accumulator dependencies wait on the
  scoreboard;
- why forward's max/sum and partial-`O` combines make a mechanical eight-warp
  design expensive, while backward uses saved LSE and disjoint `dK`/`dV`
  ownership to amortize eight warps despite having many shared-memory
  barriers;
- where Q, $dO$, K/V, LSE, $D$, transient score state, warp-local `dQ`,
  `dK`/`dV`, and cross-K `dq_tmp` reside;
- why `num_splits=1` gives one batch/head CTA a fixed K/V-tile and local-warp
  reduction order;
- why `num_splits>1` is a mode selector for one CTA per K/V tile, not the
  literal number of CTAs;
- why the parallel mode needs a separate $D$ kernel, zeroed FP32 workspace,
  atomic `dQ`, and a final copy/cast;
- how 256 threads map CTA-local `dQ` fragments to four scalar FP32 global
  atomics per active 16-byte chunk, and why same-address contention is
  cross-CTA rather than normally cross-lane;
- why a no-return reduction request can overlap independent warp work without
  becoming an explicit asynchronous-copy protocol or eliminating
  same-address serialization;
- why `dq_tmp` is a separate FP32 accumulation surface before the final
  output-dtype conversion;
- why this mode leaves `dK`/`dV` single-owned but makes only the cross-CTA
  `dQ` arrival order nondeterministic;
- what a future component and model-level verification must observe.

The next useful step is a reader-question pass over backward, especially the
subtleties that only become visible when tracing one concrete specialization.
Dropout's exact lane/counter mapping and GPU measurements of register count,
occupancy, workspace traffic, and repeatability remain open evidence tasks.

FA2 remains deliberately deferred. Before starting it, the useful check is
whether the reader can independently derive $D_i=dO_i^\top O_i$, draw the
one-K/V-tile owner diagram, explain the eight local `dQ` partials, and predict
the first mismatch introduced by parallel K/V-tile contributions.
