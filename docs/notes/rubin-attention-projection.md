# Rubin Attention Opportunity And Challenge Map

Date: 2026-08-03

Status: architecture projection from public NVIDIA Rubin material and the
source-backed FA4/Blackwell model; no pinned Rubin FlashAttention
implementation, CUDA tuning guide, SASS, profile, or GPU measurement

Read the [FA4 Blackwell deep dive](fa4.md) first. This note asks what must be
re-searched when the same exact attention dependency graph moves from
Blackwell to Rubin. It is deliberately **not** named FA5: a hardware feature
list does not establish a new FlashAttention algorithm or implementation.

Generic Rubin specifications and links are owned by
[gpu-hardware-notes](https://zyeric.github.io/gpu-hardware-notes/notes.html#source-nvidia-gpu-generations).
This note owns only their Attention consequences.

## 1. Evidence And Scope

Public hardware evidence:

- [NVIDIA Rubin GPU architecture overview](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/):
  HBM4 bandwidth, doubled Tensor Core K-dimension processing, inline TMA
  descriptor update, faster `EX2`, activation 2:4 compression, fine-grained
  dependent-kernel triggering, and counted writes;
- [NVIDIA Rubin platform overview](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/):
  NVLink 6, the NVL72 all-to-all domain, ConnectX-9, and SHARP;
- [NVIDIA Vera Rubin NVL72 product page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/):
  preliminary platform throughput, memory, and fabric figures.

Inherited Attention evidence:

- exact dense contract: `O = softmax(QK^T)V`;
- FA4 source/paper model: Blackwell matrix throughput scales faster than SMEM
  and exponential throughput, motivating TMEM residency, deeper overlap,
  software-assisted exponential, conditional rescaling, and 2-CTA work;
- backward invariant: K/V-owner work items produce complete `dK/dV` but many
  work items can contribute partial `dQ`, so deterministic replay needs an
  explicit ordered-reduction protocol.

The public Rubin material does **not** yet pin shared-memory capacity or
bandwidth, TMEM organization, register file, occupancy limits, exact MMA/TMA
instruction forms, latencies, or a FlashAttention implementation. Every
schedule conclusion below is therefore a hypothesis to test, not a measured
Rubin result.

## 2. The Main Transition

FA4's Blackwell balance can be summarized as:

```text
matrix compute scales faster than SMEM and EX2
  -> move accumulator state to TMEM
  -> overlap more matrix and non-matrix work
  -> reduce exposed softmax and rescale work
```

Rubin changes more than one side of that balance:

```text
more K work per Tensor Core clock
+ faster EX2
+ 22 TB/s HBM4
+ finer producer-consumer triggering
+ faster NVLink and counted-write completion
  -> do not merely recompile the FA4 schedule
  -> re-measure the matrix / softmax / movement / synchronization balance
```

The useful question is not “is Rubin faster?” It is which previously hidden
stage becomes exposed after the matrix, exponential, HBM, and fabric paths
scale by different amounts.

## 3. Opportunity And Challenge Matrix

| Rubin mechanism | Attention opportunity | Main challenge / non-claim |
|---|---|---|
| 22 TB/s HBM4 | Strongest direct benefit is for decode and KV-cache traffic; training paths also get cheaper Q/K/V, output, LSE, and backward workspace traffic | FlashAttention already removes the quadratic score/probability HBM round trip, so training speedup should not be estimated from HBM bandwidth alone |
| Doubled K-dimension processing per clock | Fewer K-loop iterations can reduce iteration-level waits and improve large-head or long-K tile processing | Fixed softmax, epilogue, SMEM/TMEM movement, and barriers become a larger fraction; exact tile/instruction shapes are not public |
| Faster `EX2` | Native exponentials can reclaim work that FA4 handled with partial software emulation; BF16/FP16 paths may have a different optimum from FP32 | End-to-end softmax still includes max/sum reductions, conversion, rescaling, TMEM/RMEM movement, and dependency waits |
| Inline TMA descriptor update | Same-layout/different-address K/V pages or SplitKV segments can update base pointer/stride with less descriptor overhead | Paged layouts, boundary tiles, masking, and descriptor-lifetime rules still need a concrete implementation audit |
| Fine tile-level dependent-kernel triggering | Makes split-stage SplitKV, paged/decode preprocessing, distributed attention, or output reduction more composable without whole-kernel launch boundaries | A monolithic FA kernel may gain little; extra stages add launch, cache-locality, memory-ordering, and determinism costs |
| Counted NVLink writes | Distributed/context attention can publish completion for a tile/range instead of waiting for an entire communication phase | Completion does not specify floating-point reduction order; counters need generation, reuse, visibility, and hotspot analysis |
| NVLink 6 / NVL72 all-to-all | More head/context parallel traffic can stay inside a fast rack-scale domain | A faster fabric does not erase topology, load imbalance, synchronization, or cross-node RDMA boundaries |
| Activation 2:4 compression | Opens a separate compressed-score or model-co-designed attention research track | Dropping/compressing score activations generally changes the softmax denominator and output; “dense output format” does not prove exact dense-softmax semantics |
| 3-bit LUT matrix-B | Can accelerate surrounding QKV/output projection or inference-specific weight paths | It is not an obvious fit for both `QK^T` and `PV`, whose operands are dynamic activations, and is not a transparent training replacement |

## 4. Two Tracks Must Stay Separate

### 4.1 Exact dense attention

The exact track preserves:

```text
for every unmasked score s_ij:
  contribute exp(s_ij - row_max) to the denominator
  contribute exp(s_ij - row_max) * V_j to the numerator
```

Rubin opportunities on this track are schedule and representation changes:
native-versus-emulated `EX2`, tile shape, TMA setup, pipeline depth,
workspace strategy, distributed tile readiness, and precision choices whose
error stays inside an agreed numerical envelope.

### 4.2 Compressed or activation-sparse attention

If a 2:4 mechanism removes or approximates score elements before softmax, the
dependency graph is different:

```text
dense scores -> compression/selection -> sparse softmax -> output
```

This may be valuable, especially for inference, but it needs its own model
quality, training recipe, mask interaction, backward, and determinism contract.
It should not be reported as exact FlashAttention merely because the final
tensor has dense layout.

## 5. Forward Questions

### 5.1 Re-search exponential strategy

FA4's software-assisted exponential is a response to Blackwell's measured
balance, not a timeless rule. On Rubin compare at least:

- native FP32 `EX2`;
- lower-precision `EX2` with FP32 row statistics;
- FA4-style partial software emulation;
- mixed strategies chosen by head dimension, mask density, and tile shape.

Measure the entire softmax worker timeline. Faster `EX2` can simply expose row
max/sum reductions, TMEM-to-register movement, conversion, or readiness waits.

### 5.2 Re-search tile and stage boundaries

Doubling K work per clock reduces the number of matrix iterations, but does
not reveal the best Q/K tile. Larger tiles may improve reuse while increasing
TMEM/SMEM footprint and tail waste. Fine-grained dependent triggering is most
interesting when a workload already has natural stages:

```text
KV/page gather -> attention tile -> SplitKV partial -> fixed-order reduce
```

For ordinary dense training attention, keeping these operations in one
persistent kernel may still preserve locality and avoid publication overhead.

## 6. Backward And Determinism

Rubin does not remove the FA backward ownership conflict:

```text
one K/V-owner work item -> complete dK and dV tile
many K/V-owner work items -> partial contributions to one dQ tile
```

Counted writes can express **when** remote or split contributors have arrived.
They do not define **in which floating-point order** those contributors are
combined. Keep separate:

- fast arrival-ordered reduce-add;
- semaphore or ticket-ordered writers;
- per-contributor partial buffers plus fixed-order reduction;
- hierarchical fixed-order reduction for multi-GPU attention.

HBM4 makes partial-buffer approaches relatively cheaper, while faster compute
can make their extra writes relatively more visible. Only a shape-specific
profile can choose between them.

## 7. Bottleneck Migration Checklist

For every candidate Rubin kernel, account separately for:

1. Tensor Core cycles for `QK^T` and `PV`;
2. row max, `EX2`, row sum, and rescale work;
3. TMA/SMEM/TMEM/RMEM movement;
4. producer-consumer waits and tile publication;
5. HBM bytes for inputs, outputs, LSE, and workspaces;
6. NVLink/network bytes and topology for distributed modes;
7. tail tiles, masking, varlen imbalance, and persistent-worker utilization;
8. deterministic ordering and its workspace/synchronization cost.

A Rubin port is incomplete if it only reports Tensor Core utilization.

## 8. Validation Matrix

| Question | Minimum evidence |
|---|---|
| Does native `EX2` beat FA4 software emulation? | Same shape/dtype/mask, pinned binary, SASS, Nsight Compute timeline, repeated kernel timing |
| Did the bottleneck move to SMEM/TMEM or barriers? | Per-stage cycle model plus profiler counters/stalls; do not infer from peak ratios alone |
| Does HBM4 help the target path? | Measured bytes, cache behavior, sequence length, decode/training mode, achieved bandwidth |
| Is 2:4 attention exact? | Mathematical equivalence or explicit approximation contract plus quality/error evaluation |
| Are dependent stages worthwhile? | End-to-end comparison including launch/publication overhead and cache-locality changes |
| Is distributed output deterministic? | Fixed topology and binary, repeated bitwise probes, documented contributor order and counter generations |

## 9. Durable Conclusions

- Rubin weakens the specific Blackwell assumption that exponential throughput
  remains flat while matrix throughput rises; FA4's softmax strategy must be
  re-tuned, not canonized.
- HBM4 is a clearer opportunity for decode/KV-heavy attention than for dense
  training FlashAttention, which already avoids quadratic HBM intermediates.
- Fine-grained triggering and counted writes create a larger design space for
  staged and distributed attention, but readiness is not reduction order.
- Activation sparsity belongs to a distinct approximate/model-co-designed
  track until exact dense-softmax semantics are proved.
- Public Rubin material is not yet enough to claim a concrete kernel schedule,
  occupancy, determinism result, or speedup.
