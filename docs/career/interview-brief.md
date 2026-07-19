# Interview brief

Use the [canonical results table](../portfolio/key-results.md) for exact values and [claims registry](../release/claims.md) for wording boundaries.

## 30-second explanation

I built LatentGuard-VLA to test robot action chunks before execution. The key was not just a predictor: I first created exact-state paired replay and typed evidence so every label represented a baseline-valid counterfactual. On PickCube, blind one-shot selection improved from 88.89% random to 98.33%, but repeated visual intervention failed against the fixed primary. I then redesigned it as a conservative gate that preserved clean success and intervened only 1.17% under injected faults, while honestly reporting that recall remained 4.03%.

## 2-minute explanation

Robot-policy failures are hard to attribute because two actions usually start from different states. LatentGuard archives a complete state, independently restores a baseline and candidate session, and only emits strong evidence if the baseline, restoration, action contract, full execution, and terminal task checks all pass. That produced 2,880 strong corrupted outcomes from 60 PickCube trajectories with trajectory-safe splits.

I trained four structured models across five seeds and selected only on validation; the chosen temporal model reached 0.8986 untouched-test failure AUPRC. In blind one-shot selection, temporal achieved 354/360 successful choices versus 320/360 random, while a smaller joint MLP achieved 358/360. A visual extension remained near 98% across fixed camera and lighting shifts, but did not prove improvement over action-only.

The important research turn came in closed loop. Repeated distilled visual selection intervened on 93.48% of decisions and dropped to 73.33% success versus 100% for the fixed primary. I diagnosed induced distribution shift and displacement of a reliable nominal controller, then implemented a gate that accepts nominal by default. It preserved 100% clean success and raised injected-fault success to 80% from 73.33% at 1.17% intervention. Because recall was only 4.03% and fallback still reached 100%, I call that partial recovery, not safety.

## 5-minute technical explanation

### 1. Evidence before models

- Immutable episode/provenance schemas and semantic action layouts.
- Seeded corruptions that modify only declared windows and remain unlabeled.
- Content-bound replay cases and independent baseline/candidate sessions.
- Exact archived bytes plus adapter-bound complete runtime-state comparison.
- Typed conclusive, indeterminate, invalid, skipped, and execution-error evidence.

### 2. Data and learning

M3A accepted 60 trajectories, 360 anchors, 2,880 corruptions, and 3,240 total samples. Every derivative stayed with its source trajectory. M3B compared state-only, action-only, joint MLP, and temporal models over five seeds. The temporal model won validation at 0.9249 AUPRC and reached 0.8986 test AUPRC, but the later one-shot benchmark revealed the compact joint MLP as the better selector.

### 3. Blind and visual selection

M3C separated pool generation from blind selection and proved outcomes were unavailable. Temporal beat random by 9.44 percentage points with a source-trajectory bootstrap interval excluding zero. M4A rendered three views without stepping physics; M4B external visual success was 98.89% canonical and 98.06% under each strong shift. The visual-minus-action-only interval crossed zero, so I keep that claim partial.

### 4. Deployment stress test and redesign

M4C applied the verifier repeatedly to fresh trajectories. Distilled visual success was 73.33%, fixed primary 100%, and intervention 93.48%. The system was locally competent but operationally overactive. M4D froze validation-derived conservative gates. Clean success/intervention became 100%/0%; fault success became 80% versus accept 73.33%, ungated 90%, and fallback 100%. With 4.03% recall and 10/14 targets, the redesign improved behavior without completing the research objective.

### 5. Systems engineering

Every semantic object has deterministic identity; run outputs publish transactionally; resume is idempotent; crash recovery restores a committed state and reuses the persisted decision; and remote runs use an exact clean pushed SHA. The release adds machine-readable milestone, result, and claim registries whose report bytes and JSON fields are audited locally.

## Likely questions and concise answers

### Why not train directly from corruptions?

A corruption heuristic is not an outcome. I required baseline-valid simulator replay and complete evidence before projection to a label, preventing manual or heuristic relabeling.

### Is the state restoration exact?

The archive structure, inventory, dtypes, shapes, bytes, and digests are exact. The pinned PickCube runtime round trip uses a reviewed full-state tolerance contract of 1e-6; observed maximum error was 1.1920929e-7 across all 70 components. I do not describe that as byte-exact runtime state.

### Why did one-shot success fail in closed loop?

Repeated selection induced a new state distribution and compounded local ranking errors. The selector also displaced a reliable nominal action on about 93% of decisions. One-shot ranking had not tested either effect.

### Why did conservative gating have low recall?

The frozen nominal-risk scores only weakly separated injected faults (AUROC 0.5946). A strict gate removed false/clean interventions but consequently missed most opportunities. Relaxing it after outcomes would have violated the frozen evaluation.

### Why not always use the fallback?

In this benchmark fallback reached 100%, so it is the stronger operational reference. M4D tested whether similar benefit was possible with fewer overrides; it did not meet the within-0.05 target. The correct conclusion is partial, not that gating replaces fallback.

### What did the visual model add?

It was strong and stable versus random under the fixed rendered shifts. It did not establish superiority over action-only because the confidence interval crossed zero, and closed-loop distilled performance was weaker.

### What is reusable beyond PickCube?

The typed contracts, identity, replay protocols, ledgers, audit, and adapter boundaries are reusable infrastructure. The physical and learned results are not reusable claims until a new adapter/task has independent evidence.

## Positive-result story

I separated outcomes from selection, froze all identities, and ran the same pools through every selector. Temporal one-shot selection reached 354/360 versus random 320/360 with a bootstrap interval excluding zero. The discipline made the result defensible, and the joint MLP's 358/360 result kept the model-selection conclusion honest.

## Negative-result story

The visual system looked strong offline and one-shot, but M4C repeated intervention underperformed the fixed controller by 26.67 points. Instead of tuning on final outcomes, I preserved the failure, diagnosed over-intervention and induced distribution shift, and treated it as a new design constraint.

## Systems-engineering story

Long simulator runs were content-bound to Git SHA, config, data, and evidence identities. Atomic/exclusive writes plus ledgers made retries and resumes idempotent: the largest final resume reused all 1,560 episodes with zero work. Runtime bugs were fixed locally and resynchronized rather than patched on the server.

## Research-methodology story

Each stage tested a narrower claim before the next: label validity, learnability, one-shot utility, visual robustness, repeated control, and controlled faults. When a later stage contradicted the earlier intuition, the earlier result stayed valid at its scope while the broader claim was downgraded.

## Exact limitations

One simulated PickCube task; one robot/controller contract; solver-derived nominal behavior; synthetic action corruptions; fixed eight-candidate pools; three fixed rendered views; controlled injected faults; no real robot, task/robot transfer, VLA proposal distribution, language conditioning, hardware-fault experiment, arbitrary replanning, LangMani integration, or safety certification.
