# Canonical supported-results table

This is the single human-readable table for portfolio metrics. Machine-readable values and exact JSON pointers live in [results.json](../release/results.json); public wording is governed by [claims.json](../release/claims.json). Positive, partial, negative, and unsupported findings are intentionally interleaved rather than ranked across incompatible tasks.

| Milestone | Experiment | Metric | Result | Baseline | Interpretation | Evidence source | Claim status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2C | Trusted paired replay | Strong simulator-verified outcomes | 12/12 | Valid baseline required | Physical replay gates passed for the pinned adapter. | [m2c-strong-replays](../../reports/m2c/20260715T080910Z_m2c-pickcube-replay12_aafe838_seed271828/replay-summary.json) <!-- LG-RESULT:m2c-strong-replays --> | Supported |
| M3A | State-indexed dataset | Evaluated corruptions | 2,880 | 60 trajectories, 360 anchors | Thousands of strong counterfactual outcomes with no split leakage. | [m3a-verified-outcomes](../../reports/m3a/20260715T140743Z_m3a-full_cc01a01_seed0/acceptance-summary.json) <!-- LG-RESULT:m3a-verified-outcomes --> | Supported |
| M3B | Selected temporal verifier | Untouched-test failure AUPRC | 0.8986 | Prevalence 0.1790 | Structured failure prediction was learnable in-scope. | [m3b-test-auprc](../../reports/m3b/20260716T072051Z_m3b-full_46f15d8/benchmark-summary.json) <!-- LG-RESULT:m3b-test-auprc --> | Supported |
| M3B | Calibrated coverage-risk | Failure rate at requested 80% coverage | 3.38% | Full-coverage failure 17.90% | Rejection reduced observed risk at reduced coverage. | `m3b-coverage-risk-80` | Supported |
| M3C | Temporal one-shot selector | Selected success | 354/360 (98.33%) | Random 320/360 (88.89%) | +9.44 pp; trajectory-bootstrap 95% CI [6.39, 12.50] pp. | [m3c-temporal-success](../../reports/m3c/20260716T152012Z_m3c-full_a632a70_seed271828/full/result/candidate-selection-report.json) <!-- LG-RESULT:m3c-temporal-success --> | Supported |
| M3C | Joint MLP one-shot selector | Selected success | 358/360 (99.44%) | Temporal 354/360 | Compact joint MLP beat the validation-selected temporal model in this downstream task. | `m3c-joint-success` | Supported limitation |
| M4A | Development visual dataset | Packets / images / derived samples | 1,080 / 3,240 / 9,720 | State-preserving, three views | Content-bound images with exact state integrity. | `m4a-development-packets` | Supported |
| M4A | External visual dataset | Packets / images / derived samples | 1,080 / 3,240 / 8,640 | Training prohibited | External data remained evaluation-only. | `m4a-external-packets` | Supported |
| M4B | Canonical visual one-shot | Selected success | 98.89% | Random lower by 10.24 pp | Visual selection strongly beat random. | [m4b-external-visual-success](../../reports/m4b/20260718T_m4b-full_61f05e5_seed0/result-summary.json) <!-- LG-RESULT:m4b-external-visual-success --> | Supported |
| M4B | Visual vs action-only | Success-rate difference | +0.28 pp | 95% CI [-0.83, 1.39] pp | Added visual value over actions alone was not established. | `m4b-visual-vs-action-estimate` | Partial |
| M4B | Strong camera / lighting shifts | Selected success | 98.06% / 98.06% | Canonical 98.89% | Robust to the fixed rendered shifts. | `m4b-strong-camera-success`, `m4b-strong-lighting-success` | Supported |
| M4C | Distilled visual closed loop | Canonical success | 73.33% | Fixed primary 100% | -26.67 pp: the primary repeated-intervention target failed. | [m4c-distilled-vs-fixed](../../reports/m4c/20260718T204934Z_m4c-full_2864b30_seed420000/execution-summary.json) <!-- LG-RESULT:m4c-distilled-vs-fixed --> | Unsupported success claim / accepted negative |
| M4C | Distilled visual closed loop | Non-primary intervention | 93.48% | Fixed primary 0% | One-shot ranking did not transfer safely to repeated use. | `m4c-distilled-intervention-rate` | Negative |
| M4C | Full benchmark | Outcomes | 501 success, 98 horizon, 1 unsafe, 0 execution error | 600 episodes | Negative result was not an execution-error artifact. | `m4c-horizon-exhausted` | Supported accounting |
| M4D | Clean conservative gate | Success / intervention | 100% / 0% | Accept nominal | Clean nominal behavior was preserved. | `m4d-clean-success` | Supported |
| M4D | Fault-injected conservative gate | Success | 80.00% | Accept 73.33%; ungated 90%; fallback 100% | Measurable but limited +6.67 pp over accept nominal. | [m4d-fault-gated-success](../../reports/m4d/20260719T035540Z_m4d-full_252b4f5_seed271828/result-summary.json) <!-- LG-RESULT:m4d-fault-gated-success --> | Partial |
| M4D | Fault-injected conservative gate | Intervention / fallback | 1.17% / 0.32% | Ungated intervention 90.52% | Conservatism worked operationally, at a success cost. | `m4d-intervention-rate` | Partial |
| M4D | Fault interception | Override recall | 4.03% | Target at least 60% | Central unresolved limitation; no general fault-detection claim. | [m4d-override-recall](../../reports/m4d/20260719T035540Z_m4d-full_252b4f5_seed271828/result-summary.json) <!-- LG-RESULT:m4d-override-recall --> | Unsupported target |
| M4D | Acceptance review | Passed targets | 10/14 | Full success requires 14/14 | Partial engineering success; research targets not met. | `m4d-passed-targets` | Partial |

## Reading rule

Offline classification, one-shot selection, repeated closed-loop control, and controlled fault injection answer different questions. A high AUPRC cannot be substituted for intervention success; one-shot success cannot be substituted for repeated control; and low intervention cannot be substituted for high fault recall.
