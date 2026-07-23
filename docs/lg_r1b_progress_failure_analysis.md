# LG-R1b progress-based failure-signal analysis

## Frozen signal result

The analysis uses the pre-registered rules:

```text
stagnation |delta| <= 0.02 for at least 21 sampled steps
regression delta < -0.02
```

No threshold was selected from LG-R1b outcomes, no failure head was trained,
and no intervention threshold was selected.

| Episode-level result | Count |
|---|---:|
| True positive | 27 |
| False negative | 0 |
| False positive | 288 |
| True negative | 5 |

| Metric | Value |
|---|---:|
| Failure-episode recall | 1.0000 |
| Failure-episode precision | 0.0857 |
| False-alarm rate | 0.9829 |
| Mean lead to first structured abnormality | 13.15 sampled steps |
| Mean lead to terminal failure | 346.74 sampled steps |

The signal detects all observed failures only by also flagging almost every
successful episode. The long apparent terminal lead time largely reflects
long plateaus and standard horizon exhaustion, not reliable early safety
detection.

## Interpretation

The frozen SARM has useful descriptive association with progress, but this
thresholded signal is not operationally selective. Its 98.3% false-alarm rate
precludes a detector or intervention claim. The lead values are feature-cache
sample steps, not necessarily raw simulator control steps.

The frozen VLA-JEPA diagnostic used 64 matched anchors from 789 available
anchors. It performed zero optimizer/backward steps and exposed no native
progress score. Correlations with observed progress were:

| Frozen diagnostic | Correlation |
|---|---:|
| Qwen action-token norm | -0.1369 |
| Current latent norm | 0.3283 |
| Current/predicted L1 distance | 0.2682 |
| Predicted latent norm | 0.4376 |
| Predicted temporal change | 0.3097 |
| Predictor variance | 0.4334 |

These associations are descriptive only. They do not establish prediction,
causality, candidate ranking, a failure head, or a safety detector.

Machine-readable evidence:
`artifacts/lg_r1b/progress_failure_analysis.json` and
`artifacts/lg_r1b/vlajepa_descriptive_comparison.json`.
