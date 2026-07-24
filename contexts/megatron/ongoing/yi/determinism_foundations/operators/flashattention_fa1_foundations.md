# FlashAttention-1: From Online Softmax To Determinism

Date: 2026-07-24

Status: FA1 algorithm and historical CUDA source study complete; CPU-only
source reasoning, with GPU validation still pending

## Scope And Reading Contract

This note stops at FlashAttention-1 (FA1). It is the mathematical and
mechanical foundation for the broader
[FlashAttention source audit](flashattention.md); it does not expand FA2,
FA3, FA4, CuTe, paged attention, or inference-engine scheduling.

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
  commit `6d48e14a6c2f551db96f0badc658a6279a929df3`](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3).

The paper and the implementation are separate evidence. The paper proves that
the tiled algorithm computes the same mathematical attention function. The
CUDA source is needed to decide who writes each output, which reductions run
concurrently, and what the `deterministic` option actually changes.

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

The word "running" does not imply that every row's state stays on chip across
the complete K/V traversal. In the paper's K/V-tile-outer loop, one K/V tile
is reused over many Q tiles. The running $O_i,\ell_i,m_i$ for all those Q
tiles cannot remain in the limited SRAM simultaneously, so the pseudocode
loads and writes them in HBM at each K/V-tile step. The state has a stable
logical owner, but its residence can be global memory between steps.

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

## 9. Backward, Derived Before It Is Tiled

Let $dO=\partial\phi/\partial O$ be the incoming gradient.

### 9.1 Gradient of V

Since $O=PV$:

$$
dV=P^\top dO.
$$

Elementwise, every query row contributes to each $dV_j$:

$$
dV_j=\sum_i P_{ij}dO_i.
$$

### 9.2 Gradient of P

Again from $O=PV$:

$$
dP=dO\,V^\top,\qquad
dP_{ij}=dO_i^\top V_j.
$$

### 9.3 Gradient through softmax

For one row, the softmax Jacobian gives:

$$
dS_{ij}=P_{ij}(dP_{ij}-D_i),
$$

where

$$
D_i=\sum_j P_{ij}dP_{ij}.
$$

The apparent reduction over the full probability row can be replaced by a
small dot product:

$$
D_i
=\sum_j P_{ij}(dO_i^\top V_j)
=dO_i^\top\left(\sum_jP_{ij}V_j\right)
=dO_i^\top O_i.
$$

This identity is crucial. Forward already saved $O_i$, so backward can compute
$D_i$ in $O(d)$ work without first materializing the full $P$ row.

### 9.4 Gradients of Q and K

For $S=QK^\top$:

$$
dQ=dS\,K,\qquad
dK=dS^\top Q.
$$

With a softmax scale $\tau$, $S=\tau QK^\top$, so both expressions gain the
factor $\tau$.

### 9.5 Recompute P instead of saving it

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

## 10. FA1 Paper Backward Tiling

The paper's abstract backward loop is:

```text
initialize dQ, dK, dV to zero
for each K/V tile j:
    keep local dK_j and dV_j accumulators on chip
    for each Q tile i:
        load Q_i, O_i, dO_i, dQ_i, l_i, m_i
        recompute S_ij and P_ij
        D_i = rowsum(dO_i * O_i)
        dP_ij = dO_i V_j^T
        dS_ij = P_ij * (dP_ij - D_i)
        dV_j += P_ij^T dO_i
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

This is the bridge from the math to the determinism question:

> Find every many-to-one sum, then ask whether one worker owns the complete
> sum or whether independent workers combine partial sums into a shared
> destination.

## 11. What The FA1 v1.0.9 CUDA Source Actually Does

The v1.0.9 code is an optimized implementation, not a literal transcription of
the paper loop. The important source landmarks are:

- `flash_attn/flash_attn_interface.py`: public `deterministic` option and its
  mapping to backward `num_splits`;
- `csrc/flash_attn/src/fmha_fwd_launch_template.h`: forward grid and
  query-row split heuristic;
- `csrc/flash_attn/src/fmha_fprop_kernel_1xN.h`: sequential K/V-tile loop and
  online softmax/output state;
- `csrc/flash_attn/src/fmha/kernel_traits.h` and the
  `fmha_fwd_hdim*.cu` files: CTA tile shapes and head-dimension dispatch;
- `csrc/flash_attn/src/fmha_kernel.h` and `fmha/gmem_tile.h`: packed
  variable-length offsets and predicated loads;
- `csrc/flash_attn/src/fmha/mask.h` and `fmha/softmax.h`: causal/tail
  validity checks and score masking;
- `csrc/flash_attn/src/fmha_bwd_launch_template.h`: single-CTA versus
  sequence-K-parallel backward dispatch;
- `csrc/flash_attn/src/fmha_dgrad_kernel_1xN_loop.h`: `dQ` accumulation and
  sequence-parallel atomic write;
- `csrc/flash_attn/src/fmha/gmem_tile.h`: FP32 `atomicAdd` implementation;
- `tests/test_flash_attn.py`: repeated-output race tests and the explicit
  comment that sequence-K parallelism makes `dQ` nondeterministic.

### 11.1 Forward Execution Walkthrough

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

The same parameter name has a materially different backward consequence:

| Path | Split dimension | Shared numeric destination? |
| --- | --- | --- |
| forward | different Q-row tiles | no; each split owns different $O_i$ |
| sequence-parallel backward | different K/V tiles | yes; splits contribute to the same `dQ` |

This is why forward splitting preserves unique output ownership while the
historical sequence-parallel backward uses atomic `dQ` accumulation.

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

#### Software pipeline

FA1 v1.0.9 is fused and software-pipelined, but it is not an FA3-style
producer/consumer warp pipeline. For one K/V tile, the flow is approximately:

```text
load K_j and V_j

load Q_i
compute Q_i K_j^T
prefetch Q_{i+1}
softmax current scores
compute P_ij V_j
store current partial O/LSE
switch the shared-memory Q buffer

compute Q_{i+1} K_j^T
prefetch Q_{i+2}
...
```

Q uses double-buffered shared memory, and the next Q global load is issued
before the current tile finishes softmax, PV, and output storage. The GEMM
K-dimension fragment loop also loads a following fragment while executing the
current MMA. There is no explicit `cp.async` path in this historical kernel,
no dedicated producer warp, and no pipeline that removes the data dependency
between `QK -> softmax -> PV` or between successive online-softmax states.

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

### 11.2 Backward with `num_splits == 1`

The non-sequence-parallel path launches one CTA for each batch/head pair. That
CTA walks the K/V tiles and preserves the accumulation order for the affected
gradient rows. There is no competing CTA using FP32 atomics to add another K
tile's contribution to the same `dQ`.

The v1.0.9 Python API maps `deterministic=True` to
`num_splits=1` in backward. The flag does not select a different mathematical
backward; it declines the faster cross-K-tile parallel decomposition that
would create multiple `dQ` writers.

### 11.3 Backward with `num_splits > 1`

The sequence-parallel path launches CTAs over K/V column tiles:

- each K/V tile can produce its own $dK_j,dV_j$;
- every such CTA also produces a partial contribution to many $dQ_i$ rows;
- those partial `dQ` values are accumulated into an FP32 temporary buffer with
  `atomicAdd`;
- the final temporary buffer is copied to the requested `dQ` dtype.

Atomic addition prevents lost updates, but it does not impose a stable arrival
order among CTAs. If partial values $a,b,c$ reach one element, the hardware may
realize `(a + b) + c` or `a + (b + c)`. Those expressions are equal over real
numbers and can differ in floating point.

The v1.0.9 tests encode exactly this diagnosis: forward, `dK`, and `dV` are
exact-compared across repeats, while sequence-parallel `dQ` is allowed a tiny
arithmetic tolerance and is annotated as nondeterministic.

### 11.4 Dropout state

With dropout, deterministic replay also requires the same random mask. The
source stores Philox RNG state in forward and constructs offsets from logical
batch, head, lane, and 16-by-16 attention-block coordinates so backward can
regenerate the same mask even though its traversal differs.

This solves forward/backward mask correspondence within the implementation.
It does not excuse a caller from replaying the same seed and offset. An
earlier random operation that consumes a different number of counters can
change attention dropout even if the attention kernel itself is unchanged.

## 12. The FA1 Determinism Verdict

For the pinned v1.0.9 CUDA implementation:

| Path | Source-level verdict | Reason |
| --- | --- | --- |
| forward, fixed RNG state | expected deterministic within a fixed implementation envelope | one CTA owns each output row tile |
| backward, `deterministic=True` / `num_splits=1` | expected deterministic within a fixed implementation envelope | avoids sequence-K-parallel `dQ` atomic accumulation |
| default backward when heuristic selects `num_splits>1` | `dQ` is not bitwise safe to assume deterministic | K-tile CTAs atomically add partial `dQ` |
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

## 13. Reusable Proof Obligations Learned From FA1

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
8. **Dispatch identity:** Can shape, architecture, SM count, autotuning,
   workspace, or compiler state select another plan?
9. **Mutable state:** Are Philox seed/offset, dropout masks, streams, or
   workspaces replayed and initialized identically?
10. **Claim boundary:** Is the claim same-process repeatability, fresh-process
    repeatability, clean-build repeatability, or cross-hardware equality?

An atomic instruction is evidence of a possible multi-writer sum, not an
automatic verdict by itself. It matters whether multiple concurrent workers
can reach the same address. Conversely, the absence of an atomic is not a
complete proof if buffers alias or another kernel performs an unordered
reduction.

## 14. Counterexamples That Prevent Overclaiming

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

## 15. Future FA1 Component Probe

The smallest useful GPU probe should capture one resolved FA1 call, not merely
compare final training loss.

### 15.1 `GuardSpec`

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

### 15.2 Probe matrix

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

## 16. Model-Level Guard Placement

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

## 17. Stop Line And Next Questions

This note has answered, through FA1:

- why tiled online softmax is exact attention;
- how multi-head and packed-varlen indexing wrap the same per-head algorithm;
- where forward state resides, how Q-row `num_splits` schedules work, and how
  causal skipping is balanced;
- which A100/H100 forward tiles the historical source selects, how it
  pipelines them, and why the Tensor Core $d$ reduction repeats;
- why backward can recompute $P$ from linear-sized saved state;
- where the $dQ,dK,dV$ reductions come from;
- how the historical FA1 CUDA implementation maps those reductions to CTAs;
- why its sequence-K-parallel `dQ` path is nondeterministic;
- why `deterministic=True` selects the single-split backward;
- what a future component and model-level verification must observe.

FA2 is deliberately not derived here. Before starting it, the useful checks
are whether the reader can independently explain the online-softmax invariant,
derive $D_i=dO_i^\top O_i$, and predict the first mismatch caused by parallel
K-tile contributions to `dQ`.
