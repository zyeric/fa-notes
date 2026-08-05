# Provenance

This repository was extracted on 2026-08-02 from:

```text
repository: rStar-RL/axis-training-dev-tools
branch: master
source commit: 556f7f15c46ec19e0c0beed60a2fd9282a758acb
source directory: contexts/megatron/ongoing/yi/determinism_foundations/operators/
```

The extraction selected `flashattention.md` and the FA1 through FA4 Markdown
and HTML files. It intentionally excluded DeepGEMM, FlashQLA, model-selection
records, and the separate large-head-dimension attention investigation.

The selected path history was exported before the files were reorganized into
`docs/notes/` and `docs/slides/`. Commit messages, authorship, timestamps, and
the relevant progression were retained, but object IDs were rewritten by the
path-filtered export. The original hashes remain available in the source
repository's history.

After the migration, `fa-notes` is the canonical home for the FA1–FA4 learning
notes and visual decks. `axis-training-dev-tools` retains a pointer and the
model/training qualification work that consumes these foundations.

On 2026-08-05, the coherent inference-forward portion of the previously
excluded large-head-dimension investigation was rewritten into
`docs/notes/large-head-dimension-inference.md` and a new visual comparison.
The new files preserve the narrower evidence boundary: public reference and
engine/backend sources explain forward ownership and dispatch, while selected
production binaries, GPU timelines, and large-d training backward remain
unqualified.
