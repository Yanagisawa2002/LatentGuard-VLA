# M3A-Data — State-indexed action-chunk replay dataset

## Objective

Build a training-ready, strongly labeled Direct Action Verifier dataset for the
fixed ManiSkill `3.0.1` `PickCube-v1` Panda reference scope. M3A records every
state along successful official trajectories, selects deterministic
intermediate anchors, corrupts only a fixed 16-action candidate window, replays
the complete remaining trajectory under an unchanged source continuation, and
exports compact verifier samples grouped and split by original trajectory.

M3A is data generation and validation only. It does not train a model and does
not use LangMani.

## Fixed contracts

- Environment: ManiSkill `3.0.1`, `PickCube-v1`, Panda, `num_envs=1`,
  `pd_joint_pos`, one GPU, official packaged motion-planning solver.
- Adapter: `maniskill_pickcube_v1`, complete-state numeric restoration semantic
  `tolerance_verified_full_state_v1`, fixed maximum absolute error `1e-6`.
- Candidate horizon: `H=16`; the source candidate contains `a[t:T]`, while the
  verifier input contains only `a[t:t+H]`.
- Continuation: `a[t+H:T]` remains byte-identical for source and corruption
  replay and is bound to the official source-policy identity.
- Unsafe semantic: only `pickcube_cube_center_below_world_zero_v0`.
- Split unit: stable original source trajectory identity, with target counts
  48 train, 6 validation, and 6 test trajectories.

## Implementation slices

1. Add a safe sequence archive that retains the existing M2C initial/terminal
   archive contract while recording all `T+1` complete state trees, structural
   and content digests, component inventories, public task snapshots, seed,
   compatibility identity, and source-action index.
2. Capture every post-step state and task snapshot without changing the
   official solver action stream. Validate every state of a bounded trajectory
   through a fresh public wrapper reset, full set/get round trip, complete
   structure/component comparison, and the fixed tolerance.
3. Define `PickCubeVerifierStateV1` from verified public task/robot interfaces,
   with explicit component names and fixed float32 ordering, and independently
   re-extract it after state restoration.
4. Select up to six anchors per trajectory using public task evidence and
   trajectory-relative positions: early, approach, first grasp transition,
   early transport, late transport, and near placement. Deduplicate collisions
   and fill remaining slots with deterministic evenly spaced eligible indices.
5. Represent each anchor as an independent M0 episode whose sole source
   candidate is the complete remaining source action sequence and whose sole
   observation contains the verified compact state vector.
6. Add backward-compatible M1 schema-1.1 step windows. Every M3A corruption is
   resolved with `window_start=0` and `window_end=16`, and generation verifies
   that the complete continuation remains byte-identical.
7. Build eight checked-in corruption definitions spanning Gaussian noise,
   constant bias, temporal gripper shift, segment hold, segment zeroing, and
   local temporal permutation, with explicit mild/moderate/severe IDs.
8. Reuse the generic paired replay/evidence ledger and the trusted PickCube
   adapter with state-indexed archive references. A failed remaining-trajectory
   baseline invalidates the anchor; completed counterfactual failures remain
   conclusive strong simulator evidence.
9. Export `ActionVerifierSampleV1` and
   `ActionVerifierCandidateGroupV1` using compact safe JSON/NPY storage,
   evidence references, deterministic ordering, strict inventory, transactional
   publication, and trajectory-level split/leakage validation.
10. Add the five bounded CLIs, CPU-only fake tests, compact reports, remote
    six-source smoke, and remote sixty-source single-GPU acceptance run.

## Progress semantic decision

The accepted M2C compatibility contract binds official terminal task keys and
the binary semantic `pickcube_binary_completion_v0`; it does not probe or bind
the implementation identity, callability, scalar normalization, state-only
derivation, or repeated determinism of ManiSkill's normalized dense reward.
Those required gates therefore have not passed. M3A fails closed to binary
completion only: it records official binary completion immediately before the
candidate, after the candidate window, and before the unchanged continuation,
but exports no fractional progress value or invented weighted heuristic.

## Validation

Local validation requires the complete CPU suite, Ruff lint and formatting,
mypy, `git diff --check`, all five CLI help/dry-run smoke paths, archive and
dataset round trips, tamper rejection, continuation immutability, resume
idempotence, and split leakage rejection.

Remote execution begins only from the clean pushed implementation revision.
The six-trajectory smoke must produce at least 24 valid anchors, at least 96
evaluated proposals, both conclusive classes, zero unexplained execution
errors, only strong simulator-verified accepted evidence, and zero duplicate
resume attempts. The subsequent single-RTX-5090 run targets 60 accepted source
trajectories, at least 300 anchors, at least 2,400 conclusive corrupted samples,
at least 500 successes and 500 task failures, exact 48/6/6 trajectory splits,
no leakage, complete restoration coverage within `1e-6`, unchanged
continuations, safe reload, and idempotent resume.

The local workstation does not provide CPython 3.11, so the pre-push CPU suite
was run with its repository virtual environment on CPython 3.12.10. This is a
temporary environment exception, not a compatibility claim. The exact pushed
revision must repeat the complete suite and static checks with the configured
remote CPython 3.11 interpreter before any simulator smoke. The local result is
`1021 passed, 3 skipped`; the three skips are Windows tests that require the
unavailable directory-symlink privilege. Ruff lint, Ruff format checking,
mypy over `src`, all five CLI dry runs, and `git diff --check` pass.

## Storage and source-of-truth policy

Tracked source, configuration, tests, documentation, and compact sanitized
reports are authored locally. Full states, raw actions, evidence ledgers, and
the full verifier dataset remain in the remote run root and outside Git. The
remote checkout is execution-only and must match the exact pushed SHA before
each run. Any runtime defect is fixed locally and rerun under a new run ID.

## Exclusions

No training, neural networks, VLMs, images, video, LangMani, additional tasks,
robots, controllers, simulators, arbitrary continuation policies,
multiprocessing, multi-GPU merge, manual relabeling, generalized safety claims,
or real-robot execution is included.

## Milestone state

- Starting revision: `ca8e6a8b9d4eb305b8b1bfbdaa9f4d29e0efb842`.
- Branch: `codex/m3a-state-indexed-dataset`.
- Implementation and local CPU validation are complete. The implementation
  commit/push, remote CPython 3.11 validation, remote smoke/full acceptance,
  compact report retrieval, and final result revision are pending.
