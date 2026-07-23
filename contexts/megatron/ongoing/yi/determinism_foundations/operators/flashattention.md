# FlashAttention Forward And Backward

Date: 2026-07-23

Status: first source pass complete; GPU validation pending

## Scope And Source

Pinned source:

- FlashAttention commit
  `b54df166ebb69b896892826014759d09b9c3c9c6` (2026-07-22):
  `https://github.com/Dao-AILab/flash-attention/tree/b54df166ebb69b896892826014759d09b9c3c9c6`

This pass covers ordinary dense/causal/local attention, fixed and variable
lengths, and MHA/MQA/GQA far enough to understand the determinism mechanism.
It uses the older CUDA path under `csrc/flash_attn` as the clearest mechanical
example and checks how the current CuTe path generalizes it.

It does not claim one verdict for every implementation in the repository.
ROCm/Triton/CK backends, inference KV-cache mutation, paged attention,
score/mask modifications, and block-sparse variants need separate dispatch
records.

Source landmarks within the pinned tree:

- `README.md` and `flash_attn/flash_attn_interface.py`: public deterministic
  flag contract;
- `csrc/flash_attn/flash_api.cpp`: split-workspace allocation and GQA
  post-reduction;
- `csrc/flash_attn/src/flash_bwd_kernel.h`: sequence-K `atomicAdd` destination;
- `csrc/flash_attn/src/flash_bwd_preprocess_kernel.h`: fixed split-reduction
  loop;
- `csrc/flash_attn/src/flash_bwd_launch_template.h`: deterministic grid and
  conversion launch;
- `flash_attn/cute/interface.py`, `flash_bwd_sm90.py`, and
  `flash_bwd_sm100.py`: current semaphore/write-order paths and restrictions;
- `tests/test_flash_attn.py` and `tests/cute/test_mask_mod.py`: repeated-output
  test evidence.

## The Physical Ownership Problem

### Forward

FlashAttention does not materialize the full score matrix. A CTA owns an
output row tile, walks K/V column tiles, and maintains online-softmax state:
the running maximum, running normalization sum, and output accumulator.

For a fixed kernel and shape, the K/V tile traversal and the reductions inside
the owning CTA have a fixed order. Ordinary forward therefore avoids the main
source of run-to-run nondeterminism: independent CTAs do not race to
floating-point-add partial results into the same output tile.

Changing tile size or kernel generation can change the association and hence
the bits. Fixed-artifact repeatability is not cross-artifact equality.

The public API explicitly states that forward is deterministic. With dropout,
that statement must still be read as "given the same RNG state." Replaying a
training step also requires identical Philox seed/offset consumption.

### Backward

Backward changes the ownership geometry:

- a K/V column tile can own its `dK` and `dV` work while iterating Q tiles;
- the same Q row contributes to `dQ` from many K/V column tiles;
- with GQA, several Q heads also contribute to the same K/V-head gradients.

Parallelizing the column tiles therefore creates multiple writers to a
gradient buffer. Floating-point atomic addition is mathematically
commutative, but its realized order follows CTA arrival and is not
associative. A deterministic backward must either give one worker complete
ownership or impose a fixed combine order.

## FA2 CUDA Mechanism

The standard CUDA backward in `csrc/flash_attn` makes the tradeoff explicit.

In the normal path:

- sequence-K CTAs all point at the same FP32 `dq_accum`;
- partial `dQ` fragments use `atomicAdd`;
- the winning inter-CTA order is not fixed.

With `deterministic=True`:

- the host allocates zeroed
  `[nsplits, batch, seqlen_q_rounded, heads, head_dim_rounded]` FP32 storage;
- `blockIdx.x` selects a different split buffer, so separate sequence-K CTAs
  no longer contend for the same destination;
- a later conversion kernel loops `s = 0 .. nsplits-1` and sums the split
  buffers in that fixed order before converting to FP16/BF16.

`nsplits` depends on SM count and `batch * heads`. This preserves repeated
execution on one fixed setup but makes GPU topology and launch geometry part
of the exact numerical contract.

For MQA/GQA, the legacy host path first produces expanded per-Q-head `dK/dV`
and then calls an ATen reduction over the head group. That backend reduction
is an additional contract to pin and probe; the split-buffer argument for
`dQ` alone does not prove the entire call.

## Current CuTe Mechanism

The newer CuTe backward uses a different implementation, but solves the same
multi-writer problem.

- Its fast path uses bulk FP32 reduce-add operations into global accumulation
  buffers.
- Deterministic mode allocates semaphores and makes writers wait for a
  prescribed lock value before each reduce-add, then increments the semaphore
  for the next writer.
- It also serializes GQA contributions to shared `dK/dV` when multiple Q heads
  map to one K/V head.
- Deterministic scheduling changes tile scheduling/head swizzling in addition
  to adding synchronization.

The support matrix is conditional. At the pinned revision:

- deterministic CuTe backward is rejected on SM120;
- deterministic block-sparse backward requires explicit `dq_write_order`
  metadata and related sparse traversal metadata;
- some specialized head-dimension/2-CTA and sparse paths reject deterministic
  mode.

This is why the API flag must be checked after actual backend dispatch rather
than treated as a repository-wide boolean.

## Compiler, Dispatch, And State Boundaries

Exact replay requires pinning at least:

- FlashAttention commit and selected FA2/FA3/FA4/CuTe/backend path;
- GPU architecture and SM count;
- shape, dtype, head dimension, MHA/MQA/GQA ratio, varlen metadata, causal or
  local mask, and specialized tile configuration;
- compiler/toolchain, generated artifact, CUDA driver/runtime, and PyTorch;
- stream discipline and every workspace/semaphore buffer;
- dropout RNG seed, offset, and consumption order.

The current repository contains architecture-specific kernels and unsupported
corners. "FlashAttention deterministic" is too broad; the useful claim is a
specific forward/backward entry point under a resolved dispatch.

## Existing Test Signal

The legacy race-condition test repeats a broad shape set 250 times and checks
forward output/LSE exactly. It exact-checks `dK/dV` and checks `dQ` with a very
tight arithmetic-noise allowance. The CuTe tests include explicit repeated
`torch.equal` checks for some deterministic sparse paths.

These tests are valuable evidence that the maintainers care about race
conditions and bitwise behavior, but they do not form a complete matrix over
all backends, architectures, masks, GQA ratios, and deterministic
restrictions.

## Current Verdict

- Ordinary forward is documented deterministic and its visible single-owner
  tile structure supports that contract, provided RNG state is replayed when
  dropout is active.
- Default backward is not safe to assume deterministic wherever independent
  CTAs reduce-add into a shared gradient buffer.
- A supported `deterministic=True` path has a visible fixed-order mechanism:
  split workspace plus ordered reduction in the legacy CUDA path, or
  semaphore-serialized reduce-add in current CuTe kernels.
- Bitwise equality is scoped to one resolved implementation and environment;
  it is not expected across different tile plans, architectures, or compiler
  artifacts.

The source-level label is:

> **Deterministic under a supported deterministic-backward dispatch and fixed
> implementation envelope; default backward and unsupported specialized
> paths must not inherit that label.**

## Smallest Future GPU Validation

1. Log the public entry point, resolved backend/generation, architecture,
   kernel specialization, and whether deterministic mode was accepted.
2. Repeat forward and backward at least 100 times against iteration zero with
   `torch.equal` and byte hashes for output, LSE, `dQ`, `dK`, and `dV`.
3. Run both `deterministic=False` and `True`; include a shape with many
   sequence-K tiles so an atomic race is able to appear.
4. Cover dense causal, local/SWA, varlen/packed, and MQA/GQA separately.
5. For dropout, restore the exact RNG state before every call and separately
   verify RNG-offset progression in an end-to-end replay.
6. Repeat cold/warm, fresh-process, and clean-build-cache cases while
   preserving the kernel or binary hash.
7. Treat every unsupported assertion as a failed qualification for that
   configuration, not as permission to fall back silently.

## Open Questions

- Which FlashAttention generation and entry point will each selected model
  actually dispatch to on the target Hopper/Blackwell Docker?
- Does the ATen MQA/GQA head reduction used by the legacy path have an exact
  deterministic contract on the selected PyTorch build?
- What are the performance/workspace costs of split-buffer versus
  semaphore-serialized deterministic backward for representative training
  shapes?
- Which current CuTe dense/varlen combinations have exact repeat tests, rather
  than only numerical-correctness tests?
