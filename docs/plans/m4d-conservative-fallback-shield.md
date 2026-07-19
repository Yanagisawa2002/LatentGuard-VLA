# M4D Conservative Fallback-Aware Shield

## Plan

1. Preserve the accepted M4C eight-candidate pool, exact-state runner, stride,
   tail behavior, rendering, ledger, and recovery semantics.
2. Add a content-bound clean/fault nominal schedule and a conjunctive
   `ConservativeOverrideGateV1` that receives only candidate risks,
   uncertainties, and the upstream nominal candidate ID.
3. Freeze exactly three profiles from accepted M4B validation-derived threshold
   records, run the new 12-source development matrix once, and publish one
   immutable profile selection per scorer.
4. Run the exact 1,560-episode final matrix on 60 additional untouched sources,
   verify a zero-work resume, evaluate paired source-trajectory intervals, and
   preserve negative results without final-set tuning.

## Frozen assumptions

- The local starting revision is
  `6f3d3420180d8cebd4eb7bcba7db4438b1deaf2f`.
- Fixed primary is accepted M3C candidate definition ordinal 0. The exact source
  action remains excluded.
- Fault scheduling is 70 percent fixed primary, 15 percent accepted
  moderate/shifted definitions, and 15 percent accepted severe definitions.
- Gate profile risk quantiles use the accepted direct-visual M4B per-seed
  validation maximum-balanced-accuracy thresholds. Margin quantiles use the
  absolute gap to the matching validation target-recall threshold. Uncertainty
  cutoffs use quantiles of per-seed target-recall threshold deviation from its
  median. All three source validation-prediction digests are content-bound.
- Direct visual is primary because it outperformed distilled visual in accepted
  M4C closed-loop results. Distilled remains the ablation.
- Development visual policies use canonical rendering only. Final visual
  policies use canonical, strong-camera, and strong-lighting domains.

## Exclusions

- No training, cache rebuild, VLM, LangMani, distributed execution, multi-GPU,
  real-robot claim, new corruption strength, or additional model seed.
- No M4C or M4D final outcome contributes to initial profiles.
- No raw images, states, candidates, checkpoints, feature tensors, or simulator
  streams enter Git.
- Remote tracked source is never edited. The restarted server remains powered
  on after M4D unless the user explicitly authorizes shutdown in this task.

## Validation

Run `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy src`,
all eight M4D CLI dry-run/inspection smokes, and `git diff --check`. On the
accepted final execution SHA, run one combined Linux validation suite after the
artifact gate, integration smoke, development/freeze, final benchmark, and
strict zero-work resume.
