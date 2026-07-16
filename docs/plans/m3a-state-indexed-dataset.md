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
   compatibility identity, and source-action index. Carry an ordered,
   content-bound full `T+1` state-tree digest sequence for every source
   trajectory into the anchor manifest, then derive its sorted unique leakage
   inventory in the compact split assignment so repeated states retain their
   temporal multiplicity while the leakage gate covers non-anchor states.
2. Capture every post-step state and task snapshot without changing the
   official solver action stream. Validate every state of a bounded trajectory
   through a fresh public wrapper reset, full set/get round trip, complete
   structure/component comparison, and the fixed tolerance.
3. Define `PickCubeVerifierStateV1` from verified public task/robot interfaces,
   with explicit component names and fixed float32 ordering. Its content-bound
   extraction boundary is after a verified fresh-session restoration and before
   any action; it is independently re-extracted at that same boundary.
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
   publication, trajectory-level split/leakage validation, and independent
   rebinding of each full trajectory state inventory to the source archive.
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

## Restored public-projection decision

The first remote smoke exposed one important distinction that is now explicit
in the archive contract. ManiSkill `3.0.1` computes the public `is_grasped`
result from PhysX pairwise contact impulses. `set_state_dict` restores the
complete serialized rigid-body and articulation state and updates articulation
kinematics, but it does not reconstruct that contact-impulse buffer. The only
public refresh found is a physics step, which would change `s[t]` and is invalid
for a pre-action exact-state boundary.

M3A therefore stores two separate, content-bound public projections. The
uninterrupted source-trajectory task snapshot remains immutable event evidence
for grasp-transition anchor scheduling. The verifier vector and a parallel
restored-task snapshot are captured only after
`reset(seed) -> set_state_dict -> complete get_state_dict comparison`, before
any action or physics step. The vector still declares and includes
`task/is_grasped`; it uses the value actually returned at that restored boundary
and never substitutes the source-time value, infers a grasp, or repairs contact
state. Independent fresh sessions must reproduce the entire declared vector
schema and every value within `1e-6`. Source/restored snapshot differences are
reported as compatibility diagnostics, not counted as state-restoration error.

## Validation

Local validation requires the complete CPU suite, Ruff lint and formatting,
mypy, `git diff --check`, all five CLI help/dry-run smoke paths, archive and
dataset round trips, tamper rejection, continuation immutability, resume
idempotence, and split leakage rejection. Compact collection, replay, and final
validation reports include deterministic fixed-bin and nearest-rank restoration
error distributions in addition to maxima and compared-component inventories.

Remote execution begins only from the clean pushed implementation revision.
The six-trajectory smoke must produce at least 24 valid anchors, at least 96
evaluated proposals, both conclusive classes, zero unexplained execution
errors, only strong simulator-verified accepted evidence, and zero duplicate
resume attempts. The subsequent single-RTX-5090 run targets 60 accepted source
trajectories, at least 300 anchors, at least 2,400 conclusive corrupted samples,
at least 500 successes and 500 task failures, exact 48/6/6 trajectory splits,
no leakage, complete restoration coverage within `1e-6`, unchanged
continuations, safe reload, and idempotent resume.

The local workstation does not provide CPython 3.11, so the final CPU suite was
run with its repository virtual environment on CPython 3.12.10. This is a
temporary environment exception, not a compatibility claim. The exact pushed
execution revision repeated the complete suite and static checks with the
configured remote CPython 3.11 interpreter before simulator execution. The
final local result is `1036 passed, 3 skipped`; the three skips are Windows
tests that require the unavailable directory-symlink privilege. The remote
CPython 3.11 result is `1039 passed`. Ruff lint, Ruff format checking, mypy over
`src`, all five M3A CLI help smoke paths, and `git diff --check` pass.

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
- Implementation lineage:
  `fece3b199a05a456d43c9a97ec83fabe8a0e1542`,
  `7795e329eae16ccf6530ffc4b7beb21c4ae77ecf`,
  `b57c287e95a68d64c05272bb81ec6905ec6f0d98`,
  `d745bbe185d537342beecafcf754141c57ee40f2`, and the accepted execution
  revision `cc01a01a22bd8e53f4a442d0a6f7fd561d0ab85a`. The lineage records the
  restored-boundary public projection, exact upstream-warning filter,
  production factory purpose binding, serialized failure-event comparison,
  contract-valid zeroing definition, ordered complete T+1 state inventory,
  and deterministic restoration-error distributions.
- Accepted smoke run `20260715T133653Z_m3a-smoke_cc01a01_seed0` executed the
  clean pushed revision on one RTX 5090. It accepted 6/6 source trajectories,
  archived 468 T+1 states, built 36 valid anchors with zero baseline
  exclusions, and evaluated all 288 proposals. All 288 results were conclusive
  strong simulator evidence: 231 successes and 57 task failures, with zero
  invalid, indeterminate, skipped, or execution-error outcomes. Resume
  evaluated zero attempts and reused all 288 terminal attempts. Export and
  independent reload validation accepted 324 samples in 36 groups with 4/1/1
  trajectory splits, no leakage, and dataset digest
  `sha256:feae1784aec087689e9a7efd36851e12692d70545ce556c6f73249c82379a1c2`.
- Full single-GPU acceptance run
  `20260715T140743Z_m3a-full_cc01a01_seed0` executed the same clean pushed
  revision on one RTX 5090. Collection accepted 60/60 attempts, recorded 4,560
  source actions and 4,620 T+1 states, and published archive digest
  `sha256:cc48c42f1c8395d348f968be72102b857eb2994702c6ca85bf8e6239f2fb36d5`.
  Build produced 360 anchors and 360 successful anchor baselines with zero
  exclusions. Anchor reasons were approach 60, early trajectory 60, evenly
  spaced fill 69, first grasp transition 53, late transport 59, and near
  placement 59.
- Full replay evaluated all 2,880 proposals with 2,880 valid paired baselines:
  2,289 conclusive successes and 591 conclusive task failures, with zero
  invalid, indeterminate, skipped, or execution-error outcomes. Every accepted
  result uses the `exact_simulator` trust tier. The no-op resume evaluated,
  recovered, and retried zero attempts and reused all 2,880 terminal attempts.
  Complete continuation integrity passed for every proposal.
- Full-state restoration validation covers 6,120 comparisons of exactly 70
  components, with maximum absolute error `1.1920928955078125e-7` and no value
  above `1e-6`. Independent `PickCubeVerifierStateV1` restoration covers all
  360 anchors at dimension 38 with zero maximum error. The action dimension is
  8, horizon is 16, and progress remains the binary semantic
  `pickcube_binary_completion_v0`.
- Export and independent `--require-full-target` reload validation accepted
  3,240 samples in 360 groups: 360 source and 2,880 corrupted samples. Exact
  trajectory splits are 48/6/6, with 2,592/324/324 samples. Evidence rebinding,
  resume idempotence, continuation integrity, and full-trajectory state leakage
  validation all pass. Dataset digest is
  `sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`.
- Compact sanitized evidence is stored under
  `reports/m3a/20260715T133653Z_m3a-smoke_cc01a01_seed0/` and
  `reports/m3a/20260715T140743Z_m3a-full_cc01a01_seed0/`. Raw states, actions,
  source bundles, manifests, evidence ledger, and the full verifier dataset
  remain outside Git in the remote run root.
- No training or LangMani execution was performed. The result commit subject is
  `docs(m3a): record state-indexed dataset acceptance`.
