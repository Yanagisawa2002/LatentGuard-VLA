# LG-R0 VLA-JEPA world-model interface report

## Conclusion

The frozen VLA-JEPA checkpoint contains and can execute a real temporal
V-JEPA predictor. The observed path is an official training-style offline
diagnostic, not the default policy inference path and not an online
candidate-action scorer.

The predictor is conditioned on Qwen special action-token hidden states. It
does **not** consume the generated numeric `[7, 7]` action chunk and exposes no
API for externally supplied numeric candidates. The required probe therefore
returns:

```text
REAL_CANDIDATE_PROBE_NOT_AVAILABLE
```

No synthetic candidate fallback was used.

## Dynamic tensor evidence

Twenty real eight-frame windows were sampled from the accepted recorded
success trajectories. Every window produced:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| Current V-JEPA tokens | `[1, 768, 2048]` | `bfloat16` |
| Predicted future V-JEPA tokens | `[1, 768, 2048]` | `float32` |
| Target future V-JEPA tokens | `[1, 768, 2048]` | `bfloat16` |
| Qwen action-token condition | `[1, 24, 2048]` | `float32` |

Eight frames become four encoded temporal positions. The predictor uses three
context positions with 256 visual tokens per position and predicts the
shifted future-token target. The predicted tensors were non-constant.

Success-window descriptive statistics:

| Metric | Samples | Mean |
| --- | ---: | ---: |
| Prediction-target L1 | 20 | 1.3251728654 |
| Prediction-target cosine distance | 20 | 0.2425866097 |
| Predictor variance | 20 | 5.3758615971 |

These values describe only the sampled successful windows. The four recorded
rollouts contain zero failure trajectories, so the requested failure sample
count is 0/20. There is no success/failure comparison, classifier fit,
threshold selection, calibration, or evidence that any descriptive distance
predicts task failure.

## Score availability

| Candidate signal | Native availability | Claim boundary |
| --- | --- | --- |
| Training L1 to real future target | Yes | Offline only; requires future frames |
| Risk score | No | Must not be inferred from L1 |
| Success probability | No | Requires a separately trained head |
| Progress score | No | Requires labels and a separate head |
| RGB reconstruction score | No | Predictor emits V-JEPA tokens, not RGB |
| External numeric candidate response | No | Official predictor has no such input |
| Native multi-candidate ranking | No | No candidate sampler or ranking API |

The observed L1 is not an online risk score. Target future tokens are
unavailable at decision time and must remain training/evaluation targets.

## Recommended LG-R1 test

LG-R1 should test whether the frozen representations add value to a small,
explicit LatentGuard failure head. The minimum defensible comparison is:

1. direct baseline: current representation plus the generated numeric action
   chunk;
2. VLA-JEPA head: current tokens, predicted future tokens, Qwen action-token
   representation, and the generated numeric action chunk as a separate
   feature.

The numeric action chunk must enter the new head directly because the official
predictor cannot condition on it. Target future tokens may supervise or audit
training but must never be a deployable input.

This recommendation does not select a large architecture. The first
falsification experiment should use frozen backbones and the smallest head
that can express the comparison. It should stop if the VLA-JEPA representation
does not improve validation ranking/calibration over the direct baseline or
if the improvement disappears on an untouched task/trajectory split.

## Data gate before LG-R1 training

The current four-rollout dataset is not training-ready. A future data
milestone must first produce accepted successful and failed windows with:

- natural simulator outcomes rather than heuristic relabeling;
- exact policy, task, seed, processor, and checkpoint identities;
- trajectory/task-grouped development splits;
- an explicit failure taxonomy and runtime-error separation;
- content-bound raw and public manifests;
- reload, video decode, frame iteration, finite-action, and leakage gates;
- no use of sealed PickCube final seeds or frozen PickCube evidence.

Until that gate passes, LG-R1 is **interface-ready but data-blocked**. No
failure head should be trained from the success-only LG-R0 recordings.

Primary evidence:

- `artifacts/lg_r0/world_model_interface_audit.json`
- `artifacts/lg_r0/real_candidate_probe.json`
- `artifacts/lg_r0/rollout_dataset_manifest.json`
- `artifacts/lg_r0/rollout_dataset_validation.json`
- `docs/lg_r0_vlajepa_architecture.md`
