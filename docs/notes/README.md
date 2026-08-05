# Reading Guide

These notes explain FlashAttention by following values and ownership from the
attention equation down to GPU instructions and back up to observable
determinism. Markdown is the source of truth; the HTML decks are alternate
reading surfaces, not independent specifications.

## Recommended Paths

### First encounter

1. [FA1 to FA4 high-level tile spec](tile-spec.md)
2. [FA1 to FA4 tile and computation mental model](evolution.md)
3. [FA1 forward visual map](../slides/fa1-forward.html)
4. [FA1 backward visual map](../slides/fa1-backward.html)
5. [FA1 one-page checkpoint](fa1-checkpoint.md)

The spec gives every generation the same $(i,j)$ vocabulary and separates the
mathematical tile, logical owner, and physical lowering. The evolution note
adds the causal story; the visual maps then introduce CUDA vocabulary and
ground that model in the common `B=1, H=32, N=8192, d=128` example.

### Full source-backed path

1. [FA1 foundations](fa1-foundations.md)
2. [FA2 forward delta](fa2-forward.md)
3. [FA2 backward delta](fa2-backward.md)
4. [FA3 Hopper deep dive](fa3.md)
5. [FA4 Blackwell deep dive](fa4.md)
6. [Current implementation and determinism audit](current-implementation-and-determinism.md)
7. [Rubin Attention opportunity and challenge map](rubin-attention-projection.md)

The later-generation documents assume the earlier ownership and memory model.
They intentionally explain deltas instead of repeating all of FA1.

### Determinism-only re-entry

1. Read the forward/backward ownership boxes in
   [FA1 checkpoint](fa1-checkpoint.md).
2. Read [current implementation and determinism](current-implementation-and-determinism.md).
3. Use the FA2/FA3/FA4 notes only to resolve the selected backend and its
   combine mechanism.
4. Use the FA3 follow-on DASH subsection and the FA4 equal-`4x4` causal toy
   only after separating fixed arithmetic order from efficient writer
   scheduling.

## The Questions Every Generation Must Answer

For one output or gradient tile:

1. Who owns it?
2. Which operands are tiled, and which dimension is a reduction dimension at
   this exact stage?
3. Where do inputs, online-softmax state, intermediates, and accumulators live?
4. Which values have multiple contributors, and in what order are they
   combined?
5. Which movement and compute instructions are used, and where are the
   synchronization boundaries?
6. What bottleneck motivated the schedule, and what counterfactual became
   worse?
7. Which parts are fixed by the algorithm, compile artifact, host dispatch,
   runtime scheduling, and hardware generation?
8. What source or experiment supports the claim?

## Evidence Contract

The notes distinguish:

- mathematical identities and dependency graphs;
- facts visible in pinned paper/source snapshots;
- performance mechanisms inferred from those facts;
- measurements reported by papers;
- checks still pending on A100, H100, B200-class, or Rubin hardware.

“FlashAttention is deterministic” is intentionally rejected as an unscoped
statement. A useful claim names forward or backward, the resolved backend,
shape/mode, deterministic protocol, compiler artifact, device, and RNG state.

## Scope Boundary

This repository currently owns ordinary FA1–FA4 training-attention
foundations. Generic CTA/warp/SM, memory hierarchy, clocks, pipelines, and
Tensor Core vocabulary live in
[gpu-hardware-notes](https://github.com/zyeric/gpu-hardware-notes).

Model-specific large-head-dimension, FlashMLA, inference, batch-invariance,
and linear-attention investigations remain separate until they have enough
coherent material to extend this reading graph without weakening its scope.

The Rubin note is an explicit architecture projection rather than a fifth
FlashAttention generation. It keeps exact dense Attention separate from
activation-sparse/model-co-designed alternatives and records what still needs
a public implementation or GPU measurement.
