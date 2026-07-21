# WM-v0 D1 collection report

## Decision

**Result B: the pilot mechanics are valid, but neither the pilot expansion gate
nor the formal-training gate is authorized.** The bounded real-simulator pilot
completed at commit `782f1318dcd1278fc4d58821090ed7b388f1615f` on branch
`codex/wm-v0-d1-policy-dataset`. It committed 120 futures from 30 independent
source episodes and anchors with zero rejected samples, zero duplicate
candidates, no restoration mismatch, and complete future observations.

The decisive blockers are scientific rather than operational: the execution
server has no accepted real-policy provider or bound checkpoint, so all 120
candidates retain the correct `synthetic_corruption` provenance. The pilot also
contains 11 terminal successes, 109 nonterminal/horizon-exhausted futures, and
zero terminal failures. A nonterminal result was not relabeled as failure.

No formal expansion, feature construction, model training, evaluation, or
closed-loop control was started.

## Reproducible run

- Run ID: `20260721T070938Z_wm-v0-d1-pilot_782f131_seed271828`
- Seed: `271828`
- Execution: one RTX 5090, real PickCube simulator restoration, synthetic
  candidate pilot only
- Collection interval: `2026-07-21T07:11:59Z` to
  `2026-07-21T07:16:16Z`
- Remote source gate: exact commit match and clean tracked tree before and after
- Linux validation: 25 focused tests passed; Ruff check and format check passed;
  mypy passed for 198 source files
- Compact retrieval archive SHA-256:
  `b0b747664c67eaadadd5c8b4c1288cd8f217ad9027f8323406fa93f351eb97a3`

The execution sequence was equivalent to:

```bash
python scripts/prepare_wm_v0_d1_pilot.py --config configs/wm_v0_d1/pilot.yaml
python scripts/collect_wm_v0.py --config <resolved-pilot-config> --max-samples 120
python scripts/collect_wm_v0.py --config <resolved-pilot-config> --validate-only
python scripts/validate_wm_v0_dataset.py --config <resolved-validation-config>
python scripts/collect_wm_v0.py --config <resolved-pilot-config> \
  --max-samples 120 --resume
python scripts/check_wm_v0_training_gate.py \
  --manifest <collection>/dataset-manifest.json --allow-failed
```

The resumed collection reported `zero_work_resume=true`,
`newly_completed_anchor_count=0`, and 120 unchanged samples. The payload,
completion-marker, collection-state, manifest, and quality-report digests were
identical before and after resume.

## Collection scale and diversity

| Measure | Observed |
| --- | ---: |
| Valid / rejected futures | 120 / 0 |
| Independent source episodes | 30 |
| Independent anchors | 30 |
| Candidates per anchor | 4 / 4 / 4 min/mean/max |
| Train / validation / test | 96 / 12 / 12 |
| Future-frame completeness | 100% |
| Policy-generated candidates | 0 (0%) |
| Synthetic-corruption candidates | 120 (100%) |
| Duplicate-action rejection rate | 0% |
| Policy inference / simulator failure rate | 0% / 0% |

All 30 anchors use distinct PickCube seed scene groups. Episode-phase coverage is
initial 6, approach 4, pre-grasp 6, grasp-attempt 6, transport 4, and pre-place
4. The frozen evidence did not support reliable post-grasp/lift, release, or
recovery-sensitive anchors, so those phases were not fabricated and remain a
coverage gap.

Candidate diversity is limited to one frozen M3C family:
`m3c_frozen_corruption`. It contains 60 additive-Gaussian-noise, 30
constant-bias, and 30 segment-hold candidates. Every checkpoint field is
`not_applicable`. This diversity is useful for collector mechanics, but none of
it satisfies the real-policy quota.

## Outcomes, supervision, and quality

Terminal outcomes are 11 success, 0 failure, and 109 nonterminal. Progress is
present for every future (minimum 0.1437, mean 0.5353, maximum 1.0).
State-derived events are present at all four future boundaries for grasped,
dropped, object displacement, and task success. Collision, timeout, workspace
violation, and wrong-object contact are unavailable from this adapter and remain
explicitly unobserved; consequently the aggregate event-field missing ratio is
0.5 rather than silently inferred.

Every accepted item passed restoration, action shape, finite-value, mask,
alignment, future-count, identity, provenance, and metadata checks. The rejection
log is empty because no candidate failed; the collector still retains explicit
reason codes for future failures. Source-group leakage, duplicate identity, and
restoration mismatch are all zero.

Collection throughput was 220.73 aggregate anchor-seconds: 7.36 seconds per
anchor, 1.84 seconds per future, or 0.544 futures per second. End-to-end wall time
was approximately 257 seconds. The GPU returned to 0 MiB and 0% utilization
after collection.

## Gate results

The mechanics parts of the pilot passed: at least 100 samples, at least 25
anchors, four candidates per anchor, complete metadata and futures, exact
restoration, no duplicate identity, and no split leakage. The pilot gate failed
only:

1. `both_terminal_classes_present`
2. `policy_generated_ratio_at_least_70_percent`

The formal-training gate failed:

1. `sample_count_at_least_1000`
2. `anchor_count_at_least_250`
3. `source_episode_count_at_least_100`
4. `success_count_at_least_100`
5. `failure_count_at_least_100`
6. `policy_generated_ratio_at_least_70_percent`

All remaining formal checks passed. The project therefore must not claim formal
WM-v0 training, policy-generated-only evaluation, future-latent advantage, or
readiness for a selective closed-loop probe.

## Repair path

1. Restore an accepted LangMani registry, runtime-selection record, real
   controller checkpoint, and matching processor artifacts.
2. Content-bind those assets locally, commit and push the binding, then make the
   execution server pull that exact revision.
3. Run a reset-only provider probe and verify current-observation inference,
   action/mask contract, checkpoint SHA-256, sampling metadata, and failure
   reporting.
4. Rerun the bounded 120-future pilot with at least 70% genuine
   `policy_generated` candidates and deliberately cover terminal failures and
   the missing task phases using legal checkpoints, sampling, horizons, or
   fallback policies.
5. Expand toward 1,000 samples only after every pilot gate passes; then build
   features and run the frozen offline comparisons. Closed-loop work remains a
   later milestone.
