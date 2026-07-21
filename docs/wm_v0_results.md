# WM-v0 results

## Current status

The WM-v0 data, collection, frozen-feature, model, baseline, training/resume,
evaluation, and conservative-probe contracts are implemented. Formal model
training is not authorized because the committed repository contains zero
candidate-specific simulator future trajectories. The accepted M3A/M4A records
cannot be reinterpreted as this supervision.

The live result authority is `artifacts/wm_v0/dataset_manifest.json`. Its audit
snapshot records zero samples and an explicitly failed formal gate. The matching
`evaluation_summary.json` contains no metrics and answers the central question
"does future-latent prediction improve action selection?" with `unanswered`.

Any later small real collection and short optimization run must be labelled
`non_formal_smoke`. A formal direct-versus-outcome-only-versus-WM comparison may
be added here only after strict reload proves at least 1,000 real-future samples,
split-group isolation, both terminal classes, and observed nonterminal targets.

## Local validation

- Complete CPU suite: 1,602 passed, 3 skipped. The skips are existing Windows
  directory-symbolic-link capability checks.
- WM-v0 focused suite: 12 passed, including exact restore, serialization,
  temporal alignment, group isolation, direct baseline, forward/loss,
  candidate comparison, checkpoint, and resume coverage.
- `ruff check .`, `ruff format --check .`, and `mypy src`: passed.
- Collection CLI dry-run: one declared anchor inspected, zero simulator work.
