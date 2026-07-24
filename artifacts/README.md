# Artifact Index

This directory contains compact, reviewable JSON evidence committed to Git.
Each phase preserves its original manifests, metrics, gates, environment
identity, source validation, and remote-execution summary where applicable.

## Final closeout

- [`final/final_summary.json`](final/final_summary.json): canonical LG-F0
  machine-readable result summary, contribution categories, stop statement,
  source paths, source SHA-256 digests, and JSON pointers.

Rebuild or verify it from the frozen phase artifacts:

```bash
python scripts/lg_f0_build_summary.py
python scripts/lg_f0_build_summary.py --check
```

The source-bound figures under [`docs/figures/`](../docs/figures/) read only
this summary and embed its SHA-256 digest in PNG metadata.

## Post-release VLA research

| Directory | Scope |
| --- | --- |
| [`lg_r0/`](lg_r0/) | VLA-JEPA LIBERO task-0 development smoke and interface audit |
| [`lg_r1/`](lg_r1/) | In-domain SARM-style progress baseline |
| [`lg_r1b/`](lg_r1b/) | Natural-failure expansion and held-out-task evaluation |
| [`lg_r1c/`](lg_r1c/) | Frozen task-agnostic reward comparison |
| [`lg_r2a/`](lg_r2a/) | Numeric action-conditioning controls |
| [`lg_r2b0/`](lg_r2b0/) | LIBERO candidate-source and state-restoration validation |
| [`lg_rb0/`](lg_rb0/) | RoboLab environment, recording, and baseline replay study |
| [`lg_rb01/`](lg_rb01/) | RoboLab compatibility-patch and repeated replay validation |

## Historical frozen evidence

The directory also retains compact evidence for the earlier PickCube, WM-v0,
and release lines. Those artifacts remain historical and are not rewritten by
LG-F0.

## Intentionally excluded

Raw rollouts, RGB/depth video, simulator state archives, HDF5 recordings, model
downloads, frozen embeddings, checkpoints, optimizer states, caches, remote run
directories, and large logs stay outside Git. Their absence from this index
must not be interpreted as a zero or failed metric.

Artifacts must not contain credentials, private host details, or
machine-specific absolute dataset paths.
