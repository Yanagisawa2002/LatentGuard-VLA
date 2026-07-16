# M3B Direct Action Verifier Baseline Results

## Identity and execution

- Branch: `codex/m3b-direct-verifier`
- Git SHA: `46f15d8cf50ac45027b62b0d73a4f2f49ac30224`
- Dataset digest: `sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`
- Acceptance-report digest: `sha256:f3d747daffc8dfe3528350de4e193bc2e6d2a2954fe1d4476c3a2797c317c06f`
- Split digest: `sha256:173ca40a072a1977cc65dbcccedb4684ae573e97f3984c176e3f91bba09f940a`
- Matrix: four learned architectures by five fixed seeds (20 runs)
- GPU: `NVIDIA GeForce RTX 5090`
- Driver/CUDA/PyTorch: `580.76.05` / `12.8` / `2.8.0+cu128`
- Python/NumPy: `3.11.15` / `1.26.4`
- Total epoch time: 148.810 s; mean per run: 7.440 s
- Selected epochs across all runs: [17, 24, 3, 5, 3, 7, 11, 11, 4, 6, 11, 1, 20, 16, 19, 10, 15, 9, 10, 21]
- Checkpoint reload and exact resume: passed for every smoke gate; completed-run inventories were revalidated before reuse.

## Validation-only architecture selection

| Architecture | Parameters | Failure AUPRC mean | Brier mean | Pairwise concordance mean |
|---|---:|---:|---:|---:|
| state_only_mlp | 38913 | 0.386841 | 0.212039 | 0.522245 |
| action_only_mlp | 50433 | 0.730439 | 0.122112 | 0.941348 |
| state_action_mlp | 61249 | 0.908771 | 0.078351 | 0.979826 |
| temporal_state_action_verifier | 312897 | 0.924928 | 0.075742 | 0.986650 |

Selected architecture: `temporal_state_action_verifier` using the fixed validation corrupted-only failure-AUPRC ordering and documented tie breakers.

## Untouched test metrics for the selected five-seed architecture

| Metric | Uncalibrated mean | Calibrated mean |
|---|---:|---:|
| Failure prevalence | 0.179012 | 0.179012 |
| Failure ROC AUC | 0.971208 | 0.971208 |
| Failure AUPRC | 0.894784 | 0.894784 |
| Success AUPRC | 0.993643 | 0.993643 |
| Negative log likelihood | 0.217343 | 0.199730 |
| Brier score | 0.063828 | 0.060384 |
| Balanced accuracy | 0.901089 | 0.901089 |
| Failure precision | 0.723115 | 0.723115 |
| Failure recall | 0.875862 | 0.875862 |
| Failure F1 | 0.791146 | 0.791146 |
| Specificity | 0.926316 | 0.926316 |
| Matthews correlation | 0.746354 | 0.746354 |
| Expected calibration error | 0.058537 | 0.061362 |

## Group ranking

| Metric | Five-seed mean |
|---|---:|
| Corrupted-only top-1 success | 1.000000 |
| Random-choice expectation | 0.798611 |
| Pairwise success-over-failure concordance | 0.986349 |
| Corrupted-only top-1 failure rate | 0.000000 |
| Mean reciprocal rank | 1.000000 |
| Source-margin diagnostic | 0.737604 |

## Coverage-risk (calibrated candidate scores)

| Requested coverage | Observed coverage mean | Retained failure rate mean | Rejected failure recall mean |
|---:|---:|---:|---:|
| 1.00 | 1.000000 | 0.179012 | 0.000000 |
| 0.90 | 0.901852 | 0.095142 | 0.520690 |
| 0.80 | 0.802469 | 0.033846 | 0.848276 |
| 0.70 | 0.700617 | 0.010573 | 0.958621 |
| 0.50 | 0.501852 | 0.000000 | 1.000000 |

## Non-learned test baselines

| Baseline | Failure AUPRC | ROC AUC | Brier score |
|---|---:|---:|---:|
| majority | 0.179012 | 0.500000 | 0.179012 |
| failure_prevalence | 0.179012 | 0.500000 | 0.146979 |
| action_magnitude | 0.294021 | 0.551076 | 0.239746 |

## Trajectory-level paired comparisons

All intervals use original source trajectories as the bootstrap unit. Machine-readable observed differences, intervals, valid resamples, and seeds are in `benchmark-summary.json`.
- Selected model versus `action_only_mlp`: recorded for failure AUPRC, Brier score, and corrupted top-1 success.
- Selected model versus `state_action_mlp`: recorded for failure AUPRC, Brier score, and corrupted top-1 success.
- Selected model versus `state_only_mlp`: recorded for failure AUPRC, Brier score, and corrupted top-1 success.

## Acceptance and claim boundary

- At least one learned architecture passed every fixed test target: `True`.
- Selected validation aggregate strictly exceeded state-only and action-only: `True`.
- Per-corruption, severity, anchor-reason, candidate-type, trajectory, and split slices remain in each content-bound `test-evaluation.json`.
- These results are limited to fixed PickCube privileged structured state and one continuation policy. They do not establish visual/language understanding, cross-task transfer, general safety, causal diagnosis, or real-robot validity.
- No VLM, LLM, LangMani, online simulator rollout, pretrained model, or multi-GPU/distributed training was used.
