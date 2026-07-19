# M5 release review

## Outcome

The portfolio package consolidates the frozen M0–M4D PickCube research line without starting training, simulator execution, rendering, outcome generation, threshold tuning, or new seeds. The implementation commit was cleanly pushed and matched its upstream and GitHub branch before this review.

## Evidence and claim review

The strict audit resolved all 57 registered results to exact committed report bytes, full execution commits, run IDs, and JSON pointers. All 22 claims reference registered results and are separated into eight supported, four partially supported, one unsupported, and nine not-tested claims. Public numeric rounding was checked against the unrounded registry values. The M3C trajectory-bootstrap interval resolves to the temporal-versus-random comparison rather than the adjacent abstention report.

The visible positive story is physically gated exact replay, 2,880 strong corrupted outcomes, structured learnability, strong blind one-shot selection, fixed-domain visual robustness, and clean nominal preservation. The visible negative story is equally explicit: visual added value over action-only was not established, repeated distilled visual intervention underperformed the fixed primary, the conservative fault gate remained below ungated and fallback baselines, and override recall was only 4.03%. No general safety, real-robot, cross-task, cross-robot, VLA, language, hardware-fault, arbitrary-replanning, or LangMani claim is made.

## Packaging and sanitization

The README, technical report, case study, architecture diagrams, results table, demo plan, presentation outline, resume variants, interview brief, and English/Chinese summaries were reviewed together. Internal links resolve; key result markers are present on all canonical surfaces; no private infrastructure identifier, credential assignment, raw array/model artifact, checkpoint, cache, or large generated output was added.

## Server preservation

The server audit was read-only. SSH remained available, the execution checkout was clean and matched the release-source research SHA, all M3A–M4D major runtime groups were present, the accepted Python 3.11 environment was present, checkpoints and development/external feature plus teacher caches were present, no GPU computation was running, no shutdown automation was found, and storage usage was low. Remote Git was intentionally not changed because the M5 audit contract prohibits it. The server remained online after the audit.

## Remaining boundary

M5 is a release and evidence-consolidation milestone. Its local smoke is deterministic, CPU-only, and non-physical; it does not reproduce any research metric. The remote checkout remains on the frozen release-source revision instead of the M5 documentation branch, by design of the read-only preservation rule.
