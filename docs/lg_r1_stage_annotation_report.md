# LG-R1 stage-annotation report

## Verdict

The final simulator-grounded annotation set passes the LG-R1 pre-training
gate:

- 160 of 160 rollout episodes were labeled;
- 24,373 current-state steps were labeled;
- 20 episodes were explicitly reviewed, with at least two per task;
- critical annotation errors: 0;
- minor disagreements: 0 (`0.0%`, limit `5%`);
- deterministic relabeling mismatches: 0;
- 119 stage-regression events were retained rather than repaired;
- privileged sidecar fields were used only to produce targets and were absent
  from the model-input allowlist.

The accepted report is content-bound by
`artifacts/lg_r1/stage_annotation_report.json`. The human-readable review
decisions are in `artifacts/lg_r1/stage_review_decisions.json`; their packet
digest is
`52d0c257df8a602656af2b9db28c59dc8d45f60466e840cd2d1fc85408041611`.

## Label semantics

Every label describes the current pre-action state. It is not backfilled from
a future terminal outcome. A successful post-action state may be recorded in
the episode summary, but it is not silently copied into the current-state
training samples. Consequently, a timeout does not set
`terminal_failure=true`, a failed episode is not reset to progress zero, and
progress may decrease when the current task predicates regress.

The common fields are:

- `stage_id` and `stage_name`;
- within-stage completion in `[0, 1]`;
- overall progress in `[0, 1]`;
- current-state terminal success/failure flags;
- evidence fields sufficient to audit the label.

The model-input allowlist is limited to two observations, robot state, and task
text. Object poses, fixture state, contacts, distances, and goal predicates
remain supervision-only.

## Task-specific stage adapters

| Adapter | Stage structure |
| --- | --- |
| `pick_place` | approach, alignment, contact, lift, transport, place, release, success |
| `open_place` | approach/open fixture, approach object, contact, lift, transport, place, success |
| `toggle` | approach control, alignment, contact, activation, success |
| `multi_place` | approach/manipulate/complete first object, then the corresponding sequence for the second object, success |
| `place_close` | access fixture, approach/manipulate/place object, close fixture, success |

The frozen thresholds are 18 cm for approach, 8 cm for alignment, 4.5 cm for
contact-scale geometry, 3.5 cm for lift delta, and 12 cm for target proximity.
Simulator contact and task predicates take precedence over geometric
heuristics.

## QA correction before model training

The first structured review found that a BDDL region such as
`wooden_cabinet_1_top_region` did not itself expose the articulated joint state
of its owning fixture. The adapter was corrected locally to resolve a region
to the owning fixture body, committed, pushed, and synchronized before any
SARM or representation-probe optimizer step. Labels and the review packet were
then regenerated. This was a label-infrastructure correction, not a policy,
rollout, task, seed, or outcome change.

The final natural failure, `libero_goal-task3-seed2017`, illustrates the
accepted semantics: the drawer opened, contact and transport stages were
reached, four regressions occurred, and the episode ended at the 300-step
horizon with the task predicate still false. Its final current state is stage
2 with progress `0.275`; it is not fabricated as a terminal-failure stage.

## Known limitation

Stage 7 has no current pre-action examples in the bounded feature samples:
environment success is observed after the final action, while current-state
targets remain causal. The zero stage-7 F1 reported for the probe and SARM
therefore reflects an unsupported class in these sampled inputs, not evidence
that other stages or terminal task success were misannotated. A future
milestone may add an explicitly modeled post-action target, but must not
retroactively rewrite these labels.
