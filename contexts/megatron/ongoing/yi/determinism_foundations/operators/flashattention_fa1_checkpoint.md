# FlashAttention-1 One-Page Checkpoint

Date: 2026-07-26

Status: compact re-entry guide derived from the completed FA1 paper and
historical CUDA source study; GPU validation remains pending

This is the short reading surface for
[the full FA1 foundations and source audit](flashattention_fa1_foundations.md).
It intentionally omits most derivations, layouts, dispatch tables, and source
proof. Use it to rebuild the ownership and determinism model before reading a
later FlashAttention generation.

## 1. The One-Screen Model

```text
mathematics:
  O = softmax(Q K^T) V
  online softmax keeps only row max, exponential sum, and output state

FA1 v1.0.9 forward:
  one CTA owns several Q-row tiles and their O rows
  K/V tile is the outer loop and is reused across those Q tiles
  four warps split K/V columns
  -> four partial-O values per output element
  -> fixed CTA-local shared-memory combine

FA1 v1.0.9 backward, num_splits == 1:
  one CTA per (batch, head) visits K/V tiles in a fixed order
  -> complete dQ, dK, and dV have one CTA owner

FA1 v1.0.9 backward, num_splits > 1:
  one CTA per K/V tile owns final dK_J and dV_J
  every K/V-tile CTA contributes to the same dQ rows
  -> global FP32 atomicAdd
  -> unspecified cross-CTA addition order
```

The determinism rule learned from the example is:

> Find every many-to-one floating-point sum. Determine whether one worker owns
> the complete ordered sum or independent workers publish partial values to a
> shared destination.

## 2. Evidence Boundary

- Paper:
  [FlashAttention, arXiv:2205.14135v2](https://arxiv.org/abs/2205.14135v2).
- Historical source:
  [FlashAttention v1.0.9, commit `6d48e14a6c2f551db96f0badc658a6279a929df3`](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3).

Do not collapse the two:

| Evidence | What it establishes |
|---|---|
| paper | real-arithmetic equivalence, online-softmax invariant, asymptotic IO argument |
| v1.0.9 source | actual grid, tile sizes, warp partition, memory residence, atomics, and fixed local orders |

The conclusions below describe this historical source, not every modern
FlashAttention backend.

## 3. Forward Ownership And Order

For one `(batch, head)`, the v1.0.9 forward grid uses `R=num_splits` CTAs:

```text
CTA r owns Q-tile indices:
  r, r + R, r + 2R, ...
```

For four 16-row Q tiles and `R=2`:

```text
CTA 0 owns Q tiles I0 and I2 -> final O_I0 and O_I2
CTA 1 owns Q tiles I1 and I3 -> final O_I1 and O_I3
```

Each CTA executes the conceptual order:

```text
for K/V tile J:
    load and retain K_J, V_J
    for every Q tile I owned by this CTA:
        load Q_I and prior running O_I / LSE_I
        compute Q_I K_J^T
        update online softmax
        compute the current P_IJ V_J contribution
        store updated running state
```

Consequences:

- one CTA is the only writer of every final O row it owns;
- different CTAs reread K/V because shared memory is CTA-private;
- a Q row has a fixed logical owner, but it is reloaded on later K/V steps;
- running output/normalization state can live in HBM between K/V steps;
- CTA scheduling changes completion time, not the contributor order inside
  one O owner.

### Four-warp split-K inside the CTA

For the representative `B_r=16`, `B_c=128` forward tile:

```text
warp 0 -> score columns J0
warp 1 -> score columns J1
warp 2 -> score columns J2
warp 3 -> score columns J3
```

The same partition becomes a reduction split for `PV`:

```text
partial_O_w = P[:, Jw] @ V[Jw, :]
O_tile contribution = partial_O_0 + partial_O_1
                    + partial_O_2 + partial_O_3
```

All four warps produce partial values for the same O elements. The source
stores four FP32 partial-O slices in shared memory and reloads/adds them in a
fixed index order. This is a deterministic CTA-local reduction, not an atomic
cross-CTA race.

## 4. Backward Ownership

The gradient reductions are:

```text
dQ_I = sum over K/V tiles J of dS_IJ K_J
dK_J = sum over Q tiles I of dS_IJ^T Q_I
dV_J = sum over Q tiles I of P_IJ^T dO_I
```

The implementation chooses K/V-tile ownership:

```text
work for J:
  owns final dK_J and dV_J
  contributes partial dQ_I_from_J to every visited Q tile I
```

With `num_splits == 1`, one CTA serializes all J work for a `(batch, head)`.
It therefore owns the complete `dQ` combine as well as all final `dK/dV`
tiles.

With `num_splits > 1`, the launch contains one CTA per K/V tile:

```text
CTA_J0 -- partial dQ from J0 --\
CTA_J1 -- partial dQ from J1 ---- atomicAdd -> FP32 dq_tmp
CTA_J2 -- partial dQ from J2 --/
```

`dK_J` and `dV_J` remain disjoint single-owner outputs. `dQ` becomes a
cross-CTA many-to-one sum whose atomic arrival order is not specified.

Inside each backward CTA, eight warps also produce eight local `dQ` partials.
Those are staged in `smem_dq` and added in a fixed loop. Keep this local fixed
reduction separate from the optional global atomic reduction.

## 5. Physical-State Checkpoint

| State or mechanism | Historical FA1 v1.0.9 path |
|---|---|
| forward CTA | four warps / 128 threads |
| backward CTA | eight warps / 256 threads |
| Q | global vector load through per-thread registers; double-buffered shared tile |
| K/V | global load through registers and shared memory; fragments distributed across CTA registers |
| score/P/O fragments | distributed accumulator/value registers |
| forward partial O | four warp-local FP32 copies, then shared-memory fixed combine |
| running O/LSE between K/V steps | global FP32 buffers when the CTA revisits multiple Q tiles |
| backward local dQ partials | eight FP32 shared-memory slices, fixed local combine |
| sequence-parallel dQ | global FP32 workspace updated with atomic add |
| copy model | ordinary load/store pipeline; no TMA or dedicated producer warp |
| synchronization | scoreboards for register dependencies; warp/CTA barriers for shared-buffer handoffs |

Tensor Core accumulator dependencies and fixed local reduction trees determine
the arithmetic order. Warp issue order may change timing, but cannot reverse a
true accumulator dependency. A missing barrier or unsafe shared-buffer reuse
would instead be a correctness race.

## 6. Scoped Verdict

| Path | Source-level conclusion |
|---|---|
| forward, fixed binary/dispatch/input/RNG | repeatable: one CTA owns each O row and uses fixed local orders |
| backward, `num_splits == 1` | repeatable with respect to the audited ownership path |
| backward, `num_splits > 1` | not bitwise deterministic: cross-CTA FP32 atomic addition into `dQ` |
| dropout | requires identical Philox seed, offset, and logical-coordinate replay |
| different tile plan, compiler, or architecture | may be individually repeatable but need not return the same bits |

`deterministic=True` maps to `num_splits=1` in this historical implementation.
It disables one known nondeterministic path; it is not a proof that an entire
training stack is deterministic.

## 7. Questions To Carry Into FA2

Use these axes rather than rereading FA2 from elementary attention:

1. Does a CTA still own complete O rows?
2. Is K/V or Q the outer retained state?
3. Does running O remain on chip for the complete K/V traversal?
4. Do warps split an output axis or a reduction axis?
5. Are partial O values combined, or do warp results concatenate?
6. Does memory movement use ordinary loads, `cp.async`, TMA, or dedicated
   producers?
7. Which contributor order changes, and what does that imply for bitwise
   comparison with FA1?
