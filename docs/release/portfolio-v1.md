# LatentGuard-VLA portfolio-v1

`portfolio-v1` is the stable public-facing snapshot of the frozen ManiSkill
PickCube research line. It packages the audited evidence registry, current
results and limitations, and a deterministic 78-second motion-graphics demo.

## What this release demonstrates

- content-bound exact-state paired replay with independently restored baseline
  and candidate sessions;
- 2,880 accepted strong simulator-verified corrupted outcomes across 60 source
  trajectories and 360 replay anchors;
- an untouched-test structured-verifier failure AUPRC of 0.8986;
- blind one-shot temporal selection success of 98.33% versus 88.89% for random;
- the contradictory closed-loop result: 73.33% for the distilled visual
  selector versus 100% for the fixed primary, with 93.48% intervention;
- a conservative redesign with 100% clean success, 0% clean intervention, and
  only 4.03% fault override recall.

## Demo

The attached `latentguard-vla-78s-demo.mp4` is generated from committed audited
metrics by `scripts/build_portfolio_demo.py`. It is explanatory motion graphics,
not robot footage and not a new simulator experiment.

| Property | Value |
| --- | --- |
| Duration | 78.0 seconds |
| Video | H.264 High, 1280x720, 15 fps, yuv420p |
| Size | 773,699 bytes |
| SHA-256 | `d6c6c909878cb0611ca834f2a49314e7bbd4af99d85f631ba8987ccd1ebe00a7` |

## Claim boundary

This release covers one ManiSkill PickCube task, one pinned robot/controller
contract, synthetic action corruptions, fixed candidate pools, fixed visual
domains, and controlled injected faults. It does not establish general safety,
real-robot robustness, cross-task or cross-robot transfer, VLA generalization,
or LangMani performance. M6A/M6A.1 provide contract and bridge scaffolding only;
a real LangMani probe has not passed and M6B has not started.

Raw datasets, RGB images, simulator states, caches, and checkpoints remain
outside Git. Their compact evidence references and public claim boundaries are
audited by `latentguard audit-release --strict`.
