# M3C blind candidate-selection review

- Mode: `full`
- Git SHA: `a632a702c709edb1fc21e702c83e30964652ff79`
- Source trajectories: 60
- Candidate groups: 360
- Complete candidate outcomes: 2880
- Selection manifest: `sha256:c5aa3798d0d1797a1714f8de6ad76954f6aa1b31835a3aae8d0abbd7e52dcbcc`
- Candidate pool: `sha256:e2982acd30297fb81f000ea53081afdb4e2ca9ad5dc2f16c6aa71d199c8ef1b7`
- Semantic replay evidence: `sha256:c44bb6631243c78b0c8d7db8f573454613ee04f919293c0b9b37d561502cbdd4`
- Full-pool outcome: `sha256:06ac372236586ed7e4dcf3af8fc8d09e05e063f24cdaa687ee6bcdcd8e51d05c`
- CPU latency report: `sha256:64863023fc368c10ab2802ba5db87cf5ddca9a9cf220867ca719025bdee5d584`
- GPU latency report: `sha256:35dc9b55b586c908f97e69858dcac27264da1961c52949e5288f53d07fc9f8b7`

## Selector outcomes

| Selector | Executed | Abstained | Coverage | Success | Failure | Solvable success | Mixed failure | Oracle regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `action_only_ensemble_v1` | 360 | 0 | 1.000000 | 0.986111 | 0.013889 | 0.986111 | 0.040000 | 0.013889 |
| `deterministic_random_v1` | 360 | 0 | 1.000000 | 0.888889 | 0.111111 | 0.888889 | 0.320000 | 0.111111 |
| `frozen_m3b_action_magnitude_v1` | 360 | 0 | 1.000000 | 0.869444 | 0.130556 | 0.869444 | 0.376000 | 0.130556 |
| `oracle_analysis_only_v1` | 360 | 0 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| `state_action_mlp_ensemble_v1` | 360 | 0 | 1.000000 | 0.994444 | 0.005556 | 0.994444 | 0.016000 | 0.005556 |
| `temporal_ensemble_abstention_maximum_validation_balanced_accuracy_v1` | 359 | 1 | 0.997222 | 0.983287 | 0.016713 | 0.983287 | 0.048387 | 0.016713 |
| `temporal_ensemble_abstention_target_validation_coverage_50_v1` | 202 | 158 | 0.561111 | 0.995050 | 0.004950 | 0.995050 | 0.017241 | 0.004950 |
| `temporal_ensemble_abstention_target_validation_coverage_70_v1` | 272 | 88 | 0.755556 | 0.988971 | 0.011029 | 0.988971 | 0.035714 | 0.011029 |
| `temporal_ensemble_abstention_target_validation_coverage_80_v1` | 307 | 53 | 0.852778 | 0.983713 | 0.016287 | 0.983713 | 0.051020 | 0.016287 |
| `temporal_ensemble_abstention_target_validation_coverage_90_v1` | 320 | 40 | 0.888889 | 0.984375 | 0.015625 | 0.984375 | 0.049020 | 0.015625 |
| `temporal_ensemble_abstention_target_validation_failure_recall_v1` | 359 | 1 | 0.997222 | 0.983287 | 0.016713 | 0.983287 | 0.048387 | 0.016713 |
| `temporal_ensemble_v1` | 360 | 0 | 1.000000 | 0.983333 | 0.016667 | 0.983333 | 0.048000 | 0.016667 |

## Predeclared acceptance checks

- `all_replay_evidence_strong_and_simulator_verified`: **passed**; observed 1 == 1.
- `coverage_70_retained_failure_at_most_half_random`: **passed**; observed 0.011029 <= 0.055556.
- `temporal_oracle_success_regret_on_solvable_groups`: **passed**; observed 0.016667 <= 0.100000.
- `temporal_pairwise_success_failure_concordance`: **passed**; observed 0.926502 >= 0.750000.
- `temporal_relative_task_failure_reduction_vs_random`: **passed**; observed 0.850000 >= 0.300000.
- `temporal_solvable_success_improvement_vs_random`: **passed**; observed 0.094444 >= 0.080000.
- `unexplained_execution_error_count`: **passed**; observed 0 == 0.

## Trajectory-level paired bootstrap

- `action_only_ensemble_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.097222, CI [0.069444, 0.127778].
  - `task_failure_rate`: estimate -0.097222, CI [-0.127778, -0.069444].
  - `oracle_success_regret`: estimate -0.097222, CI [-0.127778, -0.069444].
- `state_action_mlp_ensemble_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.105556, CI [0.080556, 0.133333].
  - `task_failure_rate`: estimate -0.105556, CI [-0.133333, -0.080556].
  - `oracle_success_regret`: estimate -0.105556, CI [-0.133333, -0.080556].
- `temporal_ensemble_abstention_maximum_validation_balanced_accuracy_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.094398, CI [0.066597, 0.122222].
  - `task_failure_rate`: estimate -0.094398, CI [-0.122222, -0.066597].
  - `oracle_success_regret`: estimate -0.094398, CI [-0.122222, -0.066597].
- `temporal_ensemble_abstention_target_validation_coverage_50_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.106161, CI [0.079315, 0.136111].
  - `task_failure_rate`: estimate -0.106161, CI [-0.136111, -0.079315].
  - `oracle_success_regret`: estimate -0.106161, CI [-0.136111, -0.079315].
- `temporal_ensemble_abstention_target_validation_coverage_70_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.100082, CI [0.071794, 0.130433].
  - `task_failure_rate`: estimate -0.100082, CI [-0.130433, -0.071794].
  - `oracle_success_regret`: estimate -0.100082, CI [-0.130433, -0.071794].
- `temporal_ensemble_abstention_target_validation_coverage_80_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.094824, CI [0.066503, 0.123658].
  - `task_failure_rate`: estimate -0.094824, CI [-0.123658, -0.066503].
  - `oracle_success_regret`: estimate -0.094824, CI [-0.123658, -0.066503].
- `temporal_ensemble_abstention_target_validation_coverage_90_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.095486, CI [0.067297, 0.124620].
  - `task_failure_rate`: estimate -0.095486, CI [-0.124620, -0.067297].
  - `oracle_success_regret`: estimate -0.095486, CI [-0.124620, -0.067297].
- `temporal_ensemble_abstention_target_validation_failure_recall_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.094398, CI [0.066262, 0.122222].
  - `task_failure_rate`: estimate -0.094398, CI [-0.122222, -0.066262].
  - `oracle_success_regret`: estimate -0.094398, CI [-0.122222, -0.066262].
- `temporal_ensemble_v1` minus `deterministic_random_v1` (2000 resamples):
  - `selected_success_rate`: estimate 0.094444, CI [0.063889, 0.125000].
  - `task_failure_rate`: estimate -0.094444, CI [-0.125000, -0.063889].
  - `oracle_success_regret`: estimate -0.094444, CI [-0.125000, -0.063889].
- `temporal_ensemble_v1` minus `state_action_mlp_ensemble_v1` (2000 resamples):
  - `selected_success_rate`: estimate -0.011111, CI [-0.025000, 0.000000].
  - `task_failure_rate`: estimate 0.011111, CI [0.000000, 0.025000].
  - `oracle_success_regret`: estimate 0.011111, CI [0.000000, 0.025000].

## Interpretation and limitations

- Result disposition: `quality_targets_met_without_full_outcome_tuning`.
- The temporal verifier remained the predeclared primary selector; the joint MLP remained the efficiency challenger.
- Full M3C outcomes were not used for model, threshold, candidate, or selector tuning.
- Abstention is reported as reduced coverage and never converted to success or task failure.
- This fixed post-candidate continuation experiment is not receding-horizon control.
- No VLM, LangMani, image/language model, source-action fallback, multi-GPU path, or real-robot claim was used.
- Claims remain limited to structured-state PickCube simulator selection under the frozen M3B/M3C contracts.
