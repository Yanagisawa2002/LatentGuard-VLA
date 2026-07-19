# Conservative fallback-aware action shielding

M4D evaluates whether a frozen verifier can accept a reliable upstream nominal
action by default and intervene only when a complete, confidence-aware gate
fires. It preserves the accepted M4C physical runner and eight-candidate pool.
The original source action remains excluded.

## Roles and information boundary

At each boundary the nominal candidate is the upstream proposal. Candidate
definition ordinal 0 is the accepted fixed-primary fallback. The other seven
candidates are alternatives. In the clean scenario, nominal is always fixed
primary. In the fault-injected scenario, a content-bound SHA-256 schedule assigns
70 percent fixed-primary, 15 percent moderate/shifted, and 15 percent severe
nominals. This reporting label is never passed to a scorer or gate.

Scorer-free `accept_nominal` and `always_fixed_primary_fallback` references
persist null risk fields with an explicit not-applicable gate decision. Scored
policies persist calibrated ensemble mean risks and population standard
deviations across accepted seeds. This distinction prevents non-probabilistic
reference rankings from being misreported as verifier confidence.

## Gate

`ConservativeOverrideGateV1` accepts nominal unless all four conditions hold:
nominal risk is high enough, improvement over the lowest-risk candidate is
large enough, nominal uncertainty is low enough, and best-candidate uncertainty
is low enough. A passing gate chooses minimum predicted risk with stable
candidate-ID tie breaking. Overrides are reported separately as fixed-fallback
or other-alternative interventions.

The conservative, balanced, and responsive profiles are frozen before M4D
development. Their values are deterministic quantiles of accepted M4B
direct-visual validation-derived threshold records; the three underlying
validation-prediction digests are stored in the profile configuration. M4C and
M4D final outcomes do not contribute to these initial profiles.

## Development, freeze, and final evaluation

The one-time development matrix uses 12 new successful sources, both nominal
scenarios, canonical visual rendering, and exactly three gate profiles per
scorer. `select-fallback-gates` chooses one profile independently for action,
direct visual, distilled visual, and privileged structured scorers. It accepts
only a complete development report and writes an immutable content-digested
record.

Final execution requires that record and verifies its gate-profile digest before
opening any final episode. The 60-source matrix contains four nonvisual policies
and three visual policies across three domains, both scenarios: exactly 1,560
episodes. Evaluation reports success and distinct terminal failures, intervention
categories, fault-detection diagnostics, fixed-fallback usage, timing, and 2,000
source-trajectory paired bootstrap samples. Acceptance targets are reported as
passed or failed, never assumed.

## Operational boundary

M4D performs no training, calibration fitting, feature-cache rebuild, new model
seed, or final-set tuning. Raw simulator states, images, candidates, checkpoints,
and features stay outside Git. A restarted M4D server remains online and
SSH-ready after completion unless the user explicitly authorizes shutdown in
the current task.

## Accepted M4D result

The frozen 60-source benchmark completed all 1,560 episodes: 1,393 success and
167 horizon exhaustion, with zero task failure, unsafe outcome, or execution
error. Balanced gates were frozen for action-only, direct visual, and distilled
visual; privileged structured used the conservative gate.

Gated direct preserved 1.0000 clean success with zero false overrides. Under
canonical injected faults it reached 0.8000 success versus 0.7333 for accepting
the nominal proposal, while invoking any intervention on only 0.0117 of
boundaries. Always fallback reached 1.0000 success with a 0.2907 fallback rate,
and ungated direct reached 0.9000 success with a 0.9052 intervention rate.

The result is therefore a cost/success tradeoff rather than a full acceptance.
Ten of fourteen targets passed, but fault-override recall was only 0.0403, the
success gain over accept-nominal was below 0.10, and gated direct did not come
within 0.05 of always-fallback success or reduce unsuccessful episodes by the
required 40%. Strong-camera and strong-lighting success each declined only
0.0167 from canonical, so the domain-robustness targets passed. These values are
preserved without final-set tuning in the compact M4D report.

The first final process ended with native exit 139 after 864 complete episodes.
Transactional recovery found no partial episode, completed the remaining 696,
and a strict resume subsequently executed zero work for all 1,560 identities.
