# LatentGuard-VLA Resume Bullets

Use one group aligned with the target role. Each bullet is deliberately bounded
to verified evidence and makes no closed-loop improvement claim.

## Robot Learning / VLA

- Integrated LeRobot/VLA-JEPA with LIBERO and evaluated a 320-episode,
  8-task VLA dataset with 27 natural failures; showed task-specific progress
  correlation falling from 0.977 in-domain to 0.378 on held-out tasks.
- Designed state-only versus numeric state-plus-action controls with
  episode-grouped evaluation; measured only 0.98% short-horizon MAE improvement
  and demonstrated that the gain disappeared in leave-task-6-out testing.
- Benchmarked frozen task-specific and task-agnostic failure signals; the best
  standalone zero-shot reward baseline reached 0.164 failure AUPRC, motivating
  a documented stop before unsupported candidate ranking or intervention.

## Robotics Simulation / Evaluation

- Built LIBERO and RoboLab replay-fidelity gates that separated mechanical
  completion, terminal agreement, official tolerance, and strict per-step
  identity; detected 20/30 official-tolerance failures despite 30/30 completed
  replays.
- Curated 320 multi-task episodes, 79,326 frames, and 27 natural failures with
  episode-grouped splits, held-out-task metrics, task-macro reporting, and
  explicit failure taxonomy.
- Diagnosed a RoboLab callable-list configuration overlay bug and prepared a
  schema-aware patch with no `eval`/`exec`; resolved the execution blocker while
  preserving the separate 800-step replay-determinism failure.

## ML / Research Engineering

- Ran a gate-driven experiment matrix across 7 research phases, preserving
  frozen Result B/C outcomes and binding final metrics to 19 committed JSON
  evidence sources with SHA-256 identities and JSON pointers.
- Implemented reproducible evaluation and closeout tooling with Python 3.11,
  PyTest, Ruff, MyPy, exact Git revisions, deterministic summary regeneration,
  and source-bound 300-dpi figures.
- Controlled compute scope by promoting only evidence-supported work: stopped
  after a 0.98% action-conditioned MAE gain and 20/30 replay-tolerance failures,
  without launching ranking, intervention, extra models, or a third simulator.
