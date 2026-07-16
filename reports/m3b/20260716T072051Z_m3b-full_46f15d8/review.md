# M3B remote result review

## Bound identity and gates

- Branch: `codex/m3b-direct-verifier`
- Implementation commit: `87280afc294c6c9eda038ad4c5d7b1972249fa37`
- Accepted-action dtype fix: `91a8910c572d843ac739be35ffd93d9b821d3395`
- Remote execution commit: `46f15d8cf50ac45027b62b0d73a4f2f49ac30224`
- Dataset digest: `sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`
- Split digest: `sha256:173ca40a072a1977cc65dbcccedb4684ae573e97f3984c176e3f91bba09f940a`
- Full benchmark report digest: `sha256:9e569db6cbca9626380373bea51af31bf66b3e73bc6fa2af22fb9d1dc9396bce`
- Local validation: 1,099 passed and three Windows-symlink-privilege skips; remote Linux validation: 1,102 passed with no skips. Ruff, formatting, mypy, dataset dry-run, GPU smoke, and strict completed-output reload all passed.

The full run used one NVIDIA GeForce RTX 5090 with driver 580.76.05, CUDA 12.8, PyTorch 2.8.0+cu128, Python 3.11.15, NumPy 1.26.4, and deterministic algorithms with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The smoke and full output roots were independent. The smoke executed four bounded learned runs without test evaluation; all four tiny-overfit gates reached 1.0 classification accuracy and passed checkpoint save, reload, and exact resume.

## Training matrix

All architectures used seeds `[0, 1, 2, 3, 4]` and the same fixed training protocol.

| Architecture | Parameters | Best epochs by seed | Validation corrupted-only failure AUPRC, mean +/- SD (95% t interval) | Brier, mean | Pairwise concordance, mean |
|---|---:|---|---:|---:|---:|
| `state_only_mlp` | 38,913 | 11, 1, 20, 16, 19 | 0.386841 +/- 0.008826 (0.375882, 0.397800) | 0.212039 | 0.522245 |
| `action_only_mlp` | 50,433 | 17, 24, 3, 5, 3 | 0.730439 +/- 0.023069 (0.701795, 0.759083) | 0.122112 | 0.941348 |
| `state_action_mlp` | 61,249 | 7, 11, 11, 4, 6 | 0.908771 +/- 0.010455 (0.895790, 0.921752) | 0.078351 | 0.979826 |
| `temporal_state_action_verifier` | 312,897 | 10, 15, 9, 10, 21 | 0.924928 +/- 0.015729 (0.905398, 0.944458) | 0.075742 | 0.986650 |

The 20 attempts used 153.508 seconds of measured training-attempt time and 148.810 seconds of cumulative epoch time. Individual attempts ranged from 2.878 to 20.649 seconds. Peak recorded GPU allocation was 116,602,368 bytes. Validation-only selection chose `temporal_state_action_verifier` on the primary aggregate AUPRC; no test metric participated in selection.

## Untouched selected-model test result

The selected five checkpoints were evaluated only after the selection record, per-seed validation temperatures, and both validation threshold policies were frozen.

| Candidate-level metric | Uncalibrated mean | Calibrated mean |
|---|---:|---:|
| Failure prevalence | 0.179012 | 0.179012 |
| Failure ROC AUC | 0.971208 | 0.971208 |
| Failure AUPRC | 0.894784 | 0.894784 |
| Success AUPRC | 0.993643 | 0.993643 |
| Negative log likelihood | 0.217343 | 0.199730 |
| Brier score | 0.063828 | 0.060384 |
| Balanced accuracy | 0.901089 | 0.901089 |
| Failure precision / recall / F1 | 0.723115 / 0.875862 / 0.791146 | 0.723115 / 0.875862 / 0.791146 |
| Specificity / MCC | 0.926316 / 0.746354 | 0.926316 / 0.746354 |
| Expected calibration error | 0.058537 | 0.061362 |

Temperature scaling preserved ranking, improved NLL and Brier score, and increased ECE by 0.002825, which is inside the fixed maximum allowed degradation of 0.01. Every selected seed met the 0.90 target recall on validation before its threshold was frozen. On test, the maximum-validation-balanced-accuracy policy averaged threshold 0.342721, balanced accuracy 0.907363, precision 0.691835, and recall 0.903448. The target-validation-recall policy averaged threshold 0.735542, balanced accuracy 0.872803, precision 0.811422, and recall 0.786207; the lower test recall is reported without retuning.

## Ranking, selective execution, and non-learned baselines

The selected calibrated scores achieved corrupted-only top-1 success 1.0 versus random-choice expectation 0.798611, pairwise success-over-failure concordance 0.986349, top-1 failure rate 0, mean reciprocal rank 1.0, and source-margin diagnostic 0.737604.

| Requested coverage | Observed coverage | Retained failure rate | Rejected failure recall |
|---:|---:|---:|---:|
| 1.00 | 1.000000 | 0.179012 | 0.000000 |
| 0.90 | 0.901852 | 0.095142 | 0.520690 |
| 0.80 | 0.802469 | 0.033846 | 0.848276 |
| 0.70 | 0.700617 | 0.010573 | 0.958621 |
| 0.50 | 0.501852 | 0.000000 | 1.000000 |

The majority, failure-prevalence, and action-magnitude test baselines had failure AUPRC 0.179012, 0.179012, and 0.294021; ROC AUC 0.5, 0.5, and 0.551076; and Brier score 0.179012, 0.146979, and 0.239746, respectively.

## Slices and trajectory bootstrap

Selected calibrated failure AUPRC was 0.604094 for additive Gaussian noise, 0.826709 for constant bias, 0.734740 for segment hold, and 1.0 for segment zeroing. It ranged from 0.811118 near placement to 0.973781 in the approach phase across defined anchor-reason slices. Source candidates, early-trajectory anchors, temporal field shift, local temporal permutation, and multiple mild/moderate severities were single-class; their ranking metrics remain explicitly undefined rather than being filled or dropped. All per-seed, per-corruption, severity, anchor, candidate-type, trajectory, and split records remain in the content-bound evaluation reports.

All paired intervals used 2,000 valid original-source-trajectory bootstrap resamples:

| Selected minus baseline | Failure AUPRC difference (95% interval) | Brier difference (95% interval) | Corrupted top-1 success difference (95% interval) |
|---|---:|---:|---:|
| `action_only_mlp` | +0.214283 (+0.120982, +0.319349) | -0.044373 (-0.053200, -0.032281) | 0.000000 (0, 0) |
| `state_action_mlp` | +0.006306 (-0.014578, +0.021488) | -0.001186 (-0.007069, +0.006934) | 0.000000 (0, 0) |
| `state_only_mlp` | +0.582589 (+0.546160, +0.623174) | -0.134819 (-0.149749, -0.121272) | +0.138889 (+0.083333, +0.166667) |

The selected temporal model did not establish a clear test advantage over the joint MLP: their trajectory-bootstrap AUPRC and Brier intervals include zero, and the joint MLP's calibrated candidate-level test AUPRC mean (0.896159) was slightly above the selected temporal mean (0.894784). This does not alter the frozen validation-only selection and is retained as a limitation.

## Acceptance and scope

Three learned architectures passed all fixed test targets; `state_only_mlp` did not, chiefly because its Brier score (0.188652) exceeded the prevalence predictor (0.146979) and its pairwise concordance was only 0.526587. The selected validation aggregate strictly exceeded both state-only and action-only aggregates, so the narrow joint state-action learnability gate passed.

These results apply only to fixed PickCube privileged structured state and one content-bound continuation policy. They do not establish visual or language understanding, cross-task transfer, causal diagnosis, general safety, or real-robot validity. No VLM, LLM, LangMani, pretrained model, simulator rollout during training, distributed training, or multi-GPU execution was used.
