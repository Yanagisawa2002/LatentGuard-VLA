# Authoritative public claims

The machine-readable source is [claims.json](claims.json). Wording below may be reused only on the surfaces allowed by that registry. The full evidence table is [key-results.md](../portfolio/key-results.md).

## Supported

- **C-SUP-EXACT-REPLAY:** Generic exact-state paired replay was implemented and physically validated for the pinned ManiSkill PickCube integration; it does not transfer automatically to another adapter.
- **C-SUP-EVIDENCE-SCALE:** The accepted PickCube dataset contains 2,880 strong simulator-verified corrupted outcomes from 60 source trajectories and 360 anchors.
- **C-SUP-STRUCTURED-LEARNING:** Structured state-plus-action failure prediction was learnable in this fixed scope (selected-model untouched-test failure AUPRC 0.8986).
- **C-SUP-ONE-SHOT:** Blind verifier-guided one-shot selection outperformed deterministic random selection on the untouched M3C groups.
- **C-SUP-VISUAL-ROBUSTNESS:** Frozen visual one-shot selection remained strong under the fixed camera and lighting shifts and outperformed random selection.
- **C-SUP-RESUME:** Content-bound transactional publication and zero-work resume worked across the accepted large data and benchmark runs.
- **C-SUP-CLEAN-PRESERVATION:** Conservative M4D gating preserved clean nominal performance with no clean intervention.
- **C-SUP-LIMITED-FAULT-BENEFIT:** Under controlled injected faults, conservative gating improved success from 0.7333 to 0.8000 while intervening on 1.17% of decisions, but recall and fallback-relative performance remained weak.

## Partially supported

- **C-PART-VISUAL-ADDED-VALUE:** Visual one-shot selection was strong, but visual-minus-action-only had a 95% interval containing zero.
- **C-PART-DISTILLATION:** Distillation produced useful prediction/one-shot evidence, but the distilled selector did not beat privileged or fixed baselines under repeated intervention.
- **C-PART-CLOSED-LOOP:** Some verifier-guided selectors beat random or accept-nominal controls, but M4C overall targets failed and M4D passed only 10 of 14 targets.
- **C-PART-APPROACHES-FALLBACK:** M4D used far fewer fallback/intervention decisions, but remained 0.20 below always fallback under injected faults.

## Unsupported

- **C-UNSUP-DISTILLED-BEATS-FIXED:** The distilled visual shield did not outperform the fixed primary in M4C; its success was 0.2667 lower with 93.48% non-primary intervention.

## Not tested

The release does not claim general robot safety, real-robot robustness, cross-task or cross-robot transfer, VLA generalization, arbitrary motion replanning, hardware-fault detection, language-conditioned verification, or LangMani policy improvement. These are explicit future evidence boundaries, not implied results.
