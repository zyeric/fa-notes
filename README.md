# FlashAttention Notes

Source-backed learning notes for FlashAttention 1 through 4: the attention
math, tile and worker ownership, memory movement, GPU execution primitives,
performance rationale, and determinism boundaries.

The material is organized as a progressive lowering path:

```text
attention math
  -> online softmax and IO-aware tiling
  -> CTA / warp ownership
  -> registers / SMEM / TMEM / HBM
  -> MMA, copy, barrier, and scheduling primitives
  -> performance and determinism consequences
```

## Published Site

Configure GitHub Pages to publish from:

```text
branch: main
folder: /docs
```

The site will then be available at:

```text
https://zyeric.github.io/fa-notes/
```

## Start Here

- [Public landing page](docs/index.html) - choose a reading path and open the
  visual decks.
- [FA1 forward slides](docs/slides/fa1-forward.html) - the beginner-oriented
  starting point, including the necessary A100 and CUDA execution background.
- [FA1 checkpoint](docs/notes/fa1-checkpoint.md) - compact re-entry after the
  first full pass.
- [FA1 to FA4 evolution](docs/notes/evolution.md) - the top-down explanation of
  what each generation changed and why.
- [Current implementation and determinism audit](docs/notes/current-implementation-and-determinism.md) -
  the scoped answer to which forward/backward paths can repeat bitwise.

## Reading Surfaces

The Markdown files under `docs/notes/` are the source of truth. They pin
papers, source revisions, implementation landmarks, inference boundaries, and
future GPU checks.

The standalone HTML decks under `docs/slides/` are visual reading surfaces:

| Generation | Forward | Backward | Markdown source |
| --- | --- | --- | --- |
| FA1 / Ampere | [slides](docs/slides/fa1-forward.html) | [slides](docs/slides/fa1-backward.html) | [foundations](docs/notes/fa1-foundations.md) |
| FA2 / Ampere | [slides](docs/slides/fa2-forward.html) | [slides](docs/slides/fa2-backward.html) | [forward](docs/notes/fa2-forward.md), [backward](docs/notes/fa2-backward.md) |
| FA3 / Hopper | [combined slides](docs/slides/fa3.html) | same deck | [deep dive](docs/notes/fa3.md) |
| FA4 / Blackwell | [combined slides](docs/slides/fa4.html) | same deck | [deep dive](docs/notes/fa4.md) |

`docs/notes.html` is a generated long-form HTML view for readers who prefer a
browser over GitHub Markdown.

## Repository Layout

```text
docs/
  index.html                 # GitHub Pages landing page
  notes.html                 # generated long-form reading surface
  render_notes.py            # dependency-free renderer
  notes/                     # Markdown source of truth
  slides/                    # standalone visual decks
STATUS.md                    # completion and validation boundary
PROVENANCE.md                # extraction and source-history record
```

## Scope

The first pass covers ordinary training attention and the historical evolution
from FA1 through FA4. It deliberately does not merge in FlashMLA, paged decode,
inference-engine scheduling, batch invariance, linear attention, or the
DeepSeek/Gemma large-head-dimension dispatch investigation. Those paths may
reuse this foundation, but require their own resolved implementation records.

Many physical and performance conclusions are source-backed but still
CPU-only. The documents label where SASS inspection, profiling, or repeated GPU
execution remains necessary.

## Regenerating The Long-Form Page

```bash
python3 docs/render_notes.py
```

The renderer uses only the Python standard library. The standalone slides are
maintained directly and are not generated from Markdown.

## Provenance

This repository was split from the FlashAttention learning context originally
maintained in `axis-training-dev-tools`. Relevant commit history was retained;
see [PROVENANCE.md](PROVENANCE.md).

## License

License is not selected yet. Choose an explicit content/code license before
promoting broad reuse.
