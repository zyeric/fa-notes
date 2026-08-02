# Project Status

Date: 2026-08-02

The first source-based learning pass over FlashAttention 1 through 4 is
complete. The repository is ready for reader review and incremental
refinement; it is not a claim that every current backend or deployment path
has been qualified on hardware.

## Completed First Pass

| Area | Learning narrative | Source audit | Visual surface | GPU validation |
| --- | --- | --- | --- | --- |
| FA1 forward | complete | v1.0.9 pinned | complete | pending |
| FA1 backward | complete | v1.0.9 pinned | complete | pending |
| FA2 forward | complete delta | v2.0.0 pinned | complete | pending |
| FA2 backward | complete delta | v2.0.0 pinned | complete | pending |
| FA3 forward/backward | complete first pass | early SM90 source pinned | complete | pending |
| FA4 forward/backward | complete first pass | paper plus current SM100 source pinned | complete | pending |
| Current determinism mechanism | complete first pass | legacy CUDA and current CuTe paths | long-form note | pending |

## Evidence Labels

- **Mathematical**: follows from the attention/gradient equations.
- **Source-backed**: visible in a pinned implementation and dispatch path.
- **Paper-reported**: measured by the paper authors, not locally reproduced.
- **Inferred**: a hardware/performance consequence that still needs profiling.
- **Locally measured**: currently absent because this work was done on a
  CPU-only development box.

## Deferred Qualification

- H100/B200 SASS and profiler checks for the described pipelines;
- repeated exact-output probes for forward, `dQ`, `dK`, and `dV` under resolved
  deterministic and default dispatches;
- current-main refreshes after source or compiler changes;
- FP8/FP4 and specialized head dimensions;
- FlashMLA, paged/decode attention, inference-engine scheduling, batch
  invariance, sparse attention, and linear attention;
- model-specific `d=512` and MQA/GQA dispatches.

These are future extensions, not blockers for using the current notes as a
mental model and source-reading guide.
