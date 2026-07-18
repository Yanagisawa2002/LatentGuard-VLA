# M4C plan: receding-horizon visual action shield

## Objective

Evaluate the already frozen M3B and M4B verifiers as outcome-blind, repeated
candidate selectors in an exact-state ManiSkill PickCube control loop. M4C does
not train, tune, calibrate, or select a model. It does not claim arbitrary
replanning, policy learning, task transfer, or real-robot safety.

## Fixed protocol

- Collect two new official-planner source sets: a six-trajectory mechanics
  smoke and an untouched 60-trajectory full benchmark. The checked-in seed
  schedule fixes their disjoint ranges and bounded attempt counts.
- Independently replay every accepted source plan, then reject any trajectory,
  split group, reset seed, source action, or complete state-tree overlap with
  the accepted M3A or M3C archives.
- At each decision boundary, create the same eight accepted M3C candidates from
  the fixed nominal plan. The source action is not selectable. Candidate
  horizon is 16, execution stride is 4, and the final tail uses the last full
  window followed by the untouched nominal residual.
- Compare fixed primary, deterministic random, five-seed action-only, three-seed
  direct visual, three-seed distilled visual, and five-seed privileged
  structured selectors. The two visual selectors run canonical, strong camera
  shift, and strong lighting shift domains.
- Construct every visual backbone call as exactly 128 images: the three real
  camera slots followed by round-robin repeats. Consume only feature rows zero
  through two.
- Persist the candidate pool and finalized decision before executing any action.
  Crash recovery restores the last committed full state and reuses the exact
  persisted selection without rescoring.
- Run exactly ten episodes per full source: four nonvisual and two visual
  selectors across three domains, for 600 full benchmark episodes.
- Keep success, task failure, unsafe termination, horizon exhaustion, and
  execution error separate. Report paired source-trajectory bootstrap intervals
  with 2,000 fixed resamples, domain robustness, intervention diagnostics,
  selector agreement, and measured wall time.

## Execution gates

1. Validate local configuration, unit tests, CLI dry runs, formatting, lint, and
   typing without a simulator or network.
2. Commit and push the implementation. Synchronize the remote checkout to the
   exact pushed SHA without editing tracked files remotely.
3. Run one six-source complete selector/domain matrix smoke and validate its
   transactional identities, strict resume, and evaluation report.
4. Run one 60-source/600-episode full benchmark. Do not launch an equivalent
   duplicate full benchmark.
5. Require a zero-work completed resume and combined Linux validation.
6. Retrieve only compact sanitized manifests, summaries, metrics, inspection,
   environment, and validation evidence. Keep states, RGB, checkpoints, and
   runtime ledgers outside Git.
7. Validate the retrieved evidence locally, commit and push it, and prove local,
   upstream, and remote SHA equality.
8. Per the user's current-task lifecycle instruction, shut down the remote
   server only after all preceding gates pass. Do not begin M4D or any later
   milestone.

## Assumptions and exclusions

The trusted `maniskill_pickcube_v1` tolerance-bound full-state restoration
contract remains unchanged at `1e-6`; archived bytes and digests remain exact.
The official nominal planner is fixed for the whole episode. Candidate
generation is deterministic and selector-independent. Runtime paths, hostnames,
timestamps, and wall time are evidence only and never semantic identities.
Raw visual observations, state archives, checkpoints, and credentials are
excluded from Git. No VLM, LangMani, distributed execution, multi-GPU execution,
or new training is authorized.

## Completion evidence

The six-source smoke accepted six plans in six attempts and completed 60/60
episodes with zero execution errors. The full set accepted 60 plans in 60
attempts, passed the M3A/M3C/smoke disjointness gates, and completed the fixed
600/600 matrix: 501 success, 98 horizon exhaustion, one unsafe termination,
zero task failure, and zero execution error. Strict resume executed zero work
for all 600 identities.

The primary distilled selector improved over random but failed the fixed
primary, action-only, direct-visual, and privileged-structured comparisons, so
the overall research acceptance targets were not met. The negative result is
retained without tuning. Local validation passed with 1522 tests and three
Windows symlink-permission skips; the final Linux SHA passed all 1525 tests,
ruff, format, and mypy. Compact result JSON and the honest result review are
stored under `reports/m4c/20260718T204934Z_m4c-full_2864b30_seed420000/`.

Final Git parity and the explicitly requested server shutdown remain the last
two lifecycle gates; no later milestone is authorized.
