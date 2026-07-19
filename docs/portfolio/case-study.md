# From one-shot success to conservative partial recovery

## Initial hypothesis

If a verifier can rank counterfactual action chunks by failure risk, selecting a low-risk candidate should improve action execution. The project first made that hypothesis testable: exact state identity, independent baseline/corrupted sessions, strong evidence, trajectory-grouped data, and outcome-blind selection were all enforced before model claims.

## Offline and one-shot evidence

Structured prediction was strong on the untouched M3B test split, and M3C blind one-shot selection confirmed downstream utility. The temporal ensemble selected successful actions in 354/360 groups versus 320/360 for random; the trajectory-bootstrap improvement interval excluded zero. The compact joint MLP was even stronger at 358/360. M4B then reached 98.89% external visual one-shot success and 98.06% under each fixed strong camera and lighting shift.

These results supported action verification and one-shot choice—not repeated closed-loop control. The authoritative values are in the [canonical table](key-results.md).

## Deployment assumption that failed

M4C applied the selector at every decision boundary. The distilled visual system intervened on 93.48% of decisions and achieved only 73.33% canonical success, versus 100% for the fixed primary. It beat random but missed the important deployment baselines. <!-- LG-RESULT:m4c-distilled-vs-fixed -->

The failure was informative: training/evaluation candidates were state-local counterfactuals, while repeated selection changed the future state distribution. Small ranking errors accumulated, near-continuous alternative choice displaced a reliable nominal controller, and one-shot calibration did not describe the induced closed-loop distribution.

## Conservative redesign

M4D inverted the default: accept nominal unless a frozen validation-bound gate fires. If it fires, separately record fixed-fallback and other-alternative invocations. Clean canonical success returned to 100% with zero intervention. Under controlled injected faults, gated direct visual improved success from 73.33% to 80.00% while intervening on 1.17% of decisions. <!-- LG-RESULT:m4d-fault-gated-success -->

## Remaining limitation

The redesign solved over-intervention, not fault detection. Override recall was 4.03%, gated success was below ungated direct (90%) and always fallback (100%), and 4 of 14 targets failed. <!-- LG-RESULT:m4d-override-recall --> The project therefore ends with low-intervention partial recovery rather than a safety claim.

## Engineering lesson

The strongest contribution is the evidence discipline around changing conclusions. The same immutable reports and predeclared targets support a positive one-shot result, a negative repeated-control result, and a partial redesign. Exact identities, transactional recovery, and strict release registries make those conclusions reviewable without rerunning expensive experiments or rewriting inconvenient outcomes.
