# RET-1 Robot Episode Toolkit Core

## Start-state audit

- Branch: `codex/m2c-maniskill-pickcube`, tracking
  `origin/codex/m2c-maniskill-pickcube`.
- Working tree: clean before RET-1 changes.
- Existing milestones: M0 data contract, M1 corruption engine, M2A evidence and
  resumable runner, M2B generic exact-state paired replay, and M2C ManiSkill
  PickCube reference adapter.
- Planned files: new `latentguard.toolkit` audit, offline replay, metrics, and
  reporting modules; public exports; existing CLI; focused CPU-only tests;
  README, architecture/data-contract notes, and this plan.
- Compatibility risks: RET-1 offline iteration must remain distinct from M2B
  exact-state simulator replay; audit findings must not weaken strict M0
  validation; reports must not change the M0 schema or bundle format; candidate
  metrics must not be presented as episode outcomes.
- Explicit exclusions: no LangMani/ManiSkill/LeRobot/ROS imports in the toolkit,
  simulator execution, training, GPU, network/SSH, database, multiprocessing,
  UI, video, remote workflow changes, schema migration, or input repair.

## Implementation plan

1. Add immutable audit models/configuration and deterministic read-only checks.
2. Add exact-index offline episode replay without simulator semantics.
3. Add deterministic candidate-denominator dataset and grouped metrics.
4. Reuse strict bundle loading and add protected transactional JSON/JSONL output.
5. Extend the existing CLI with `audit-data`, `summarize-data`, and
   `replay-episode` while preserving all commands.
6. Add CPU-only API/CLI tests and document contracts and future adapter bounds.

## Acceptance and exclusions

Run the complete required validation suite and RET-1 smoke commands twice for
determinism. Per the milestone-specific request, do not commit, push, or create
a pull request; leave the reviewed working tree available for human inspection.
