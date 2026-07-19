# Resume bullet variants

All metrics below are defined in the [canonical results table](../portfolio/key-results.md) and restricted to ManiSkill PickCube.

## Concise two-bullet version

- **Built** a typed exact-state counterfactual replay and evidence pipeline for robot actions, producing 2,880 strong simulator-verified PickCube outcomes with trajectory-level leakage controls and zero-work resume.
- **Developed and stress-tested** structured/visual action verifiers: achieved 98.33% blind one-shot success versus 88.89% random, then diagnosed a 73.33%-versus-100% closed-loop regression and redesigned a clean-preserving 1.17%-intervention gate.

## Detailed three-bullet version

- **Architected** a simulator-independent Python framework for content-bound state/action identities, paired baseline/candidate replay, typed evidence, transactional publication, crash recovery, and exact-SHA remote execution.
- **Trained and evaluated** four structured verifier architectures across five seeds, selecting only on validation and reaching 0.8986 untouched-test failure AUPRC; extended the pipeline to 3,240 multi-view images per development/external set with external-training prohibition.
- **Converted** a negative deployment result into a measured redesign: preserved 100% clean success with zero false intervention and improved injected-fault success from 73.33% to 80.00%, while reporting the unresolved 4.03% override recall and 10/14 target outcome.

## Research-oriented version

- **Formulated** action verification as baseline-gated exact-state counterfactual evaluation, separating corruption, evidence, offline prediction, blind selection, repeated control, and controlled fault injection.
- **Demonstrated** positive one-shot selection (temporal 354/360 vs random 320/360; 95% improvement CI [6.39, 12.50] percentage points) and preserved the contradictory closed-loop finding (distilled 73.33% vs fixed 100%, 93.48% intervention).
- **Designed** a conservative validation-bound fallback gate that removed clean over-intervention and reduced injected-fault intervention to 1.17%, while explicitly rejecting general-safety claims because override recall remained 4.03%.
