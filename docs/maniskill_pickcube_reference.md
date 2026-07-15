# ManiSkill PickCube reference integration

## Scope and role

M2C uses ManiSkill `3.0.1` `PickCube-v1` with the Panda robot as the first real
simulator implementation of the generic M2B-Core exact-replay protocols. It is
a narrow reference because the task has an official packaged Panda
motion-planning solution and a compact terminal evaluator. It is not a general
ManiSkill adapter and does not depend on LangMani.

The required runtime contract is one environment, GPU simulation,
`control_mode="pd_joint_pos"`, a minimal state-free observation mode whose exact
installed spelling must be probe-verified, no camera observations, and no
video. M2C does not train a model.

## Compatibility before trust

`latentguard probe-maniskill-pickcube` inspects the real installed runtime and
writes a sanitized, versioned report. The report binds the task and official
solver source digests, controller and action contract, public state/evaluator
APIs, reset-state structure, and state round-trip behavior. Operational Python,
PyTorch, CUDA, GPU, SAPIEN, and mplib versions are retained for diagnosis, but
hostnames and installation paths do not enter semantic compatibility identity.

The reviewed schema-1.1 discovery report resolves ManiSkill `3.0.1`, mplib
`0.1.1`, SAPIEN `3.0.3`, observation mode `none`, and the eight-component
controller/action contract. Those values, the source and state identities, and
the compatibility identity are now bound locally rather than guessed from
memory or vector dimension. Trusted collection cannot begin until this binding
is committed, pushed, and matched by a second trusted probe. Discovery observed
maximum full-state round-trip error `1.1920929e-7` across 70 numeric components;
the authorized adapter-bound maximum remains fixed at `1e-6` and cannot be
loosened on the remote machine.

## Official source boundary

Source actions are expected from the installed official module
`mani_skill.examples.motionplanning.panda.solutions.pick_cube` and its `solve`
entry point, the pinned-package equivalent of the official PickCube solution.
The compatibility probe must confirm and content-bind that installed source;
the project neither copies nor modifies it. A narrow environment proxy records
every action passed to `step`, copies it before storage, checks that the solver
did not mutate it, and compares the intercepted count with the environment's
elapsed-step evidence. The proxy records the exact reset-boundary state,
terminal state, task evidence, seed, and action contract without using a private
HDF5 handle or requiring ManiSkill's `RecordEpisode` parser.

A solver-reported success is only generation evidence. A second fresh
environment must restore the recorded state, verify the full round trip, replay
every recorded source action, and independently establish terminal success.
Only then is the trajectory admitted to the runtime archive and M0 dataset.

## State archive

Runtime archives live outside Git. Each state tree is flattened with canonical
typed key paths. A strict JSON manifest records mapping/list/tuple structure
plus the dtype, shape, and file reference of every numeric leaf; leaves use
non-pickle NPY files. Loading rejects traversal, links, unknown versions, object
arrays, non-finite data, missing or extra files, dtype/shape changes, and digest
tampering before reconstructing the tree.

The `maniskill_state_tree_v1` digest binds structure, canonical leaf paths,
dtype, shape, and exact C-order bytes. This digest is the archive-integrity
boundary: loading and replay references still require the archived source state
to match it exactly. Converting leaves back to runtime tensors and devices
occurs only inside the integration. Live `set_state_dict`/`get_state_dict`
verification is a separate numerical gate: the complete structure, paths,
dtypes, and shapes must match, and every value must remain within the checked-in
tolerance. Both raw state digests remain recorded, so tolerated runtime byte
drift is visible without being misclassified as archive-content mutation.
Baseline and corrupted replay references use this same numerical comparison
contract independently. Its versioned name is
`tolerance_verified_full_state_v1`, and its maximum absolute tolerance is
`1e-6`. The compatibility report serializes that exact named semantic and
tolerance in its deterministic identity. The same values and the compatibility
identity are jointly bound into adapter configuration, replay-case identity,
and evidence. Evidence also records the observed maximum error and complete
compared-component count. Structural, inventory, non-finite, or over-tolerance
drift remains unverified. A completed comparison that exceeds the bound does not
authorize action execution.

## M0 import and action corruption

Every accepted source becomes one M0 episode with a single observation at
timestamp zero, no cameras, and one complete official-source action candidate.
The Panda robot-state vector is the concatenation of joint positions followed
by joint velocities in the exact public active-joint name order recorded by the
archive. Joint names and the robot-state semantic version are content-bound;
array position alone is never treated as a name.

The source candidate receives a strong simulator label only after independent
baseline success: success `true`, binary progress `1.0`, and unsafe `false`
under the narrow cube-below-world-zero proxy. M1 then creates unlabeled
proposals using only the verified action layout. Transformations are not
clipped; actions outside the probed action bounds are invalid context and are
not task failures.

## Paired replay and evidence

The explicit adapter ID is `maniskill_pickcube_v1`. Runtime archive and dataset
paths are supplied separately and never enter adapter, replay-case, or evidence
identity. Each replay case uses the standard M2B models and references one
content-bound archived reset state.

Baseline and corrupted actions execute in separate freshly created GPU
environments. Each public Gym wrapper is first reset with the same source seed
bound from the exact archive into the replay-case identity. Only then does each
session restore and read back the same archived state. A missing or archive-
drifted seed fails closed before state restoration or action execution. The
original source action must execute completely and succeed before the
transformed action is interpreted. Simulator exceptions remain execution
errors; restoration or action-contract mismatches are invalid context; a
complete corrupted task failure remains conclusive evidence.

Terminal task evidence uses the official `success`, object-placed,
robot-static, and grasp-status values. Missing, non-scalar, non-boolean, or
non-finite values are never replaced with failure defaults. Progress uses
`pickcube_binary_completion_v0`: `1.0` for official success and `0.0` for a
complete official failure. No fractional progress is claimed.

Unsafe uses `pickcube_cube_center_below_world_zero_v0` and is true only when the
cube center is verified below world `z=0`. It says nothing about collision or
contact force, self-collision, humans, hardware, or general workspace safety.
Still grasping, incomplete placement, and robot motion do not themselves imply
unsafe.

The adapter's exact-simulator/strong descriptor is a ceiling, not a result. A
record becomes strong and simulator-verified only after compatibility binding,
both complete descriptor-bound restorations, complete successful baseline
execution, complete corrupted execution, and complete terminal task evaluation
pass.

## Commands and remote lifecycle

The integration exposes three commands:

```text
latentguard probe-maniskill-pickcube --help
latentguard collect-maniskill-pickcube --help
latentguard replay-maniskill-pickcube --help
```

Collection targets six independently successful trajectories with deterministic
ordered seeds and an explicit attempt limit. Replay reuses the M2A ledger and
supports bounded selection, dry run, resume, execution-error retry, and
fail-fast behavior. Dry run validates static artifacts and deterministic
identities without importing the simulator runtime, creating a session,
stepping an environment, or creating an output directory.

Collection writes the immutable state archive, finalized M0 bundle, and a
compact content-bound reference summary. The standard M2B `ReplayBundle` is
materialized by replay only after the M1 proposal dataset exists; collection
does not invent an alternate proposal-free bundle format.

Every remote gate runs from an exact pushed SHA via a dedicated key-authenticated
`BatchMode=yes` connection whose endpoint details remain outside Git. Large
archives and logs remain under the remote run root outside the checkout. Only
reviewed, sanitized compatibility, collection, replay, evidence, state-statistic,
environment, and resume summaries return under `reports/m2c/<run-id>/`.

## Current status

The local integration, probe-bound expected contract and dependencies, safe
archive, M0 import, adapter boundaries, three CLIs, and CPU fake tests are
implemented. Dedicated key-only access, the remote read-only Git deploy key,
Python 3.11, one RTX 5090, CUDA/Vulkan/PhysX, ManiSkill `3.0.1`, SAPIEN `3.0.3`,
and mplib `0.1.1` have been operationally verified.

Remote discovery established observation mode `none`, eight float32 action
components (seven arm and one gripper), and complete 4-leaf/70-component state
round trips with maximum error `1.1920929e-7`. The historical schema-1.0 probe
remains sanitized diagnostic evidence and cannot authorize trust. The passing
schema-1.1 discovery report
`20260715T073250Z_m2c-pickcube-compat_bb2c35c_seed0_numeric-v1` supplied the
committed dependency and expected-contract binding. A second trusted probe then
passed, and six official source trajectories passed independent baseline replay.
The first 12-proposal paired replay preserved complete restoration but produced
12 `ResetNeeded` execution errors at baseline action step zero because the
public Gym wrapper had not been initialized. No task failure was fabricated.
Adapter version `1.1.1` now binds the archived source seed and requires public
wrapper initialization before state restoration. Trusted rerun
`20260715T080910Z_m2c-pickcube-replay12_aafe838_seed271828` produced 12 valid
baselines, eight conclusive successes, four conclusive task failures, 12 strong
simulator-verified evidence records, and zero invalid, indeterminate, or
execution-error results. Both baseline and corrupted restorations compared all
70 components with maximum absolute error `1.1920929e-7`. Resume reused all 12
completed attempts and left the evaluation manifest byte-for-byte unchanged.
M2C is accepted for this narrow reference scope; no training was performed or
claimed.

## Known limitations

M2C covers one task, robot, controller, backend, reset boundary, and complete
trajectory replay. It has no images, video, subtrajectory replay, vectorized
environments, fractional progress, general safety judgment, policy checkpoint,
or training path.
