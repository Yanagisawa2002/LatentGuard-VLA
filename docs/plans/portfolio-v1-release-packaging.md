# Portfolio v1 release packaging plan

## Goal

Create a stable, resume-ready presentation of the frozen LatentGuard-VLA
PickCube research line without changing accepted results or running a new
experiment.

## Scope

- move the complete stable history onto `main` and make it the default branch;
- update the repository homepage with current audited results and limitations;
- build a 60-90 second deterministic demo covering exact replay, verifier
  learning, one-shot success, closed-loop failure, and conservative redesign;
- publish the MP4 outside Git as a stable GitHub Release asset;
- point career material to that immutable release rather than a development
  branch.

## Assumptions

- `portfolio-v1` is the stable release identity for this packaging pass;
- the existing PickCube release evidence and frozen release manifest remain the
  source of truth;
- public repository visibility and license choice are separate governance
  decisions because the repository is currently private and has no license.

## Exclusions

- no simulator execution, training, checkpoint loading, data generation, or
  model/result changes;
- no raw dataset, RGB, state, cache, checkpoint, or video committed to Git;
- no LangMani performance claim and no M6B work;
- no server startup or lifecycle change.

## Validation

- generate and inspect the poster and representative video frames;
- verify MP4 duration, codec, dimensions, and stream integrity;
- run `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy src`,
  `latentguard audit-release --strict`, and `latentguard portfolio-smoke`;
- review all tracked changes, staged contents, and final local/upstream/default
  branch identities before publishing the release.
