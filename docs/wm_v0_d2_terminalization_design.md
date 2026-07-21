# WM-v0 D2 Terminalization Design

## Status

This is a frozen design, **not an executed result**. D2 stopped at the accepted
policy asset gate, so no branch was run and no terminal label was produced.

## Two-stage rollout

For every future compatible-policy candidate, the evaluator will:

1. restore the same content-bound full simulator state in an independent session;
2. execute the candidate chunk while recording the short prediction horizon;
3. retain RGB, proprioception, progress, event, and physical-integrity evidence;
4. obtain the post-candidate real observation without replaying an expert suffix;
5. initialize the same accepted primary continuation policy;
6. continue under identical limits until success, explicit task failure,
   environment termination, maximum outcome horizon, or simulator error.

The terminal vocabulary is exactly `SUCCESS`, `FAILURE`,
`UNRESOLVED_HORIZON`, and `SIMULATOR_ERROR`. An exhausted horizon is not failure,
and a simulator exception is not a task outcome.

## Fair continuation rule

The default reset semantic is
`RESET_FROM_CURRENT_OBSERVATION_AND_CLEAR_ACTION_QUEUE`:

- the complete candidate chunk is executed;
- continuation observes the resulting physical state;
- anchor-time temporal aggregation, history, recurrent state, latent sample, and
  queued actions are cleared;
- the continuation checkpoint, processor, inference configuration, seed rule,
  remaining-step limit, and safety limits are identical for every candidate at
  an anchor;
- each branch records candidate and continuation policy IDs and hashes,
  execution counts, reset semantic, and terminal reason.

This isolates the effect of the candidate chunk better than allowing every
candidate to use its own continuation. It also measures recoverability by a
fresh primary policy, not the behavior of a stateful policy whose hidden state
was generated along a different physical history.

## Limitation

Clearing policy state can be conservative for policies whose success depends on
long observation history. A future package may instead authorize
`RESTORE_BRANCH_SPECIFIC_POLICY_STATE`, but only if state serialization is
complete, content-bound, restored independently for every branch, and tested for
resume equivalence. `REUSE_ACTION_QUEUE` is prohibited for the default design
because the queue was computed before the candidate changed the state.

## Candidate diversity thresholds

Exact duplicates are defined by identical action content hash and are retained
once. Near-duplicate and meaningful-distance thresholds must be expressed in the
accepted package's bound normalized action coordinates and checked against its
physical action ranges and gripper convention. No compatible package was found,
so D2 deliberately leaves these numeric thresholds unset rather than invent an
arbitrary dimensionless constant.

## Resume and split invariants

- The immutable key is source episode, anchor, candidate policy identity,
  candidate inference configuration, and action-content digest.
- A committed branch result is never executed twice on resume.
- An incomplete terminalization attempt resumes from its last committed full
  state and never fabricates a terminal label.
- All candidates from one anchor remain in one source-group split.
- Exact restore mismatch invalidates context and cannot become failure.
