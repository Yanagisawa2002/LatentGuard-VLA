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

## Completion evidence

The artifact gate loaded and content-bound all accepted scorers, selectors,
checkpoints, camera/render definitions, candidate pool, and compatibility
identity. The 12-source development matrix completed 384/384 episodes and froze
balanced gates for action-only, direct visual, and distilled visual, plus a
conservative privileged gate. The 60-source final set was accepted in 60
attempts, passed disjointness against seven prior archives, and completed the
fixed 1,560-episode matrix on
`0f24befbfc84a502c1af60e06bbeb8cd734a7863`.

The final benchmark recorded 1,393 successes and 167 horizon exhaustions, with
zero task failures, unsafe outcomes, or execution errors. Gated direct preserved
1.0000 clean success with zero false overrides and improved injected-fault
success from 0.7333 for accept-nominal to 0.8000 while intervening on 0.0117 of
boundaries. It nevertheless failed four predeclared targets: the required 0.10
success gain, 40% relative unsuccessful reduction, success within 0.05 of
always fallback, and 0.60 fault-override recall. The negative result is retained
without tuning.

A native exit 139 occurred after 864 complete episodes. The preserved audit
found no partial episode; transactional resume reused all 864 and executed the
remaining 696. The final strict resume executed zero work for all 1,560
identities. Linux validation passed with 1,546 tests, Ruff, formatting, and mypy
over 172 source files. Compact sanitized evidence is stored under
`reports/m4d/20260719T035540Z_m4d-full_252b4f5_seed271828/`. The server remains
online and SSH-ready.
