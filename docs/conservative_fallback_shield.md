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
