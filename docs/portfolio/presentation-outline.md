# Presentation outline (10 slides)

1. **Problem and thesis** — Verify action chunks before execution; distinguish evidence, prediction, selection, and control.
2. **Contract-first architecture** — Core/adapters boundary, exact identities, allowlisted model inputs.
3. **Exact-state counterfactuals** — Independent sessions, baseline gate, complete-state restoration, strong evidence.
4. **Dataset and reliability** — 60 trajectories, 360 anchors, 2,880 outcomes; trajectory splits and zero-work resume.
5. **Structured verifier** — Four architectures, validation-only selection, test AUPRC 0.8986, calibration/ranking/coverage-risk.
6. **Blind one-shot selection** — Random 88.89%, temporal 98.33%, joint 99.44%; outcome-blind chronology and bootstrap.
7. **Visual verifier** — State-preserving three-view data, external training prohibition, 98.89% canonical and 98.06% shifts; no proven action-only advantage.
8. **Closed-loop negative result** — M4C distilled 73.33% versus fixed 100%; 93.48% intervention; why one-shot evidence did not transfer.
9. **Conservative partial recovery** — M4D clean 100%/0%, fault 80% at 1.17% intervention, but 4.03% recall and 10/14 targets.
10. **Engineering and next evidence** — Transactional recovery, exact-SHA remote execution, release registries; future tasks/robots require new milestones.

Use the [canonical results table](key-results.md) for every numeric slide and the [claims registry](../release/claims.md) for wording. Do not add a general safety, VLA, LangMani, or real-robot conclusion.
