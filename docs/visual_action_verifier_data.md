# Multi-View Visual Action-Verifier Data

M4A adds deterministic RGB observations to the accepted PickCube action-verifier
contracts. It is a data-generation and validation milestone only. It does not
train a visual model, fine-tune or extract a visual encoder, compute teacher
logits or frozen embeddings, run large-scale augmentation or normalization
fitting, run a VLM/LLM, use LangMani, or change any accepted M3A or M3C outcome.
M4A ends after visual data generation, strict validation, compact result
retrieval, Git closeout, and paid-server shutdown; it does not begin M4B
automatically.

## Exact render boundary

Every packet is rendered at one exact physical anchor in a compatible session:

```text
reset(bound source seed)
-> set_state_dict(anchor state)
-> verify the complete restored state
-> capture the restored-boundary 38D verifier state and task projection
-> configure exactly one fixed render domain
-> render all three fixed RGB cameras without stepping
-> recapture and verify state, verifier state, and task projection
```

One rendering worker may reuse an initialized environment to reduce startup
cost, but every packet still performs the complete sequence above with an
independent reset and exact state restoration. No state from a previous packet
may substitute for that restoration, camera/domain changes may not step
physics, and cross-packet drift fails closed. The worker closes its environment
when its bounded work completes or fails. If safe reuse fails the smoke
integrity gate, production falls back to a fresh compatible environment per
packet without weakening any check.

The render path has no candidate action and no physics/contact refresh step.
The complete state comparison still covers all 70 numeric components under the
adapter-bound `tolerance_verified_full_state_v1` contract with maximum absolute
error `1e-6`. The verifier vector and restored-boundary task projection must be
exact. A visually plausible image is rejected if any physical-integrity gate
fails.

## Camera rig and domains

`PickCubeMultiViewRigV1` contains three fixed project-owned world cameras:

- `front_oblique`;
- `overhead`;
- `side_oblique`.

Each camera binds its world pose, intrinsics, extrinsics, near/far planes,
field of view, and stable camera configuration identity. Authoritative output
is RGB uint8 with shape `[224, 224, 3]`. M4A has no wrist camera, depth,
segmentation, optical flow, or video input.

The declared vertical field of view and frozen pinhole intrinsics must describe
the same projection at a fixed absolute tolerance of `1e-9`; changing either
side alone is rejected. Render seeds use only the versioned
`sha256_anchor_domain_base_seed_v1` semantic. That semantic is checked in the
domain configuration, every packet job, the complete job inventory, and the
CLI plan rather than being accepted as an arbitrary descriptive string.

The five versioned domains are canonical, mild camera shift, mild lighting
shift, strong camera shift, and strong lighting shift. Domains may change only
camera pose or lighting; they never change geometry, physics, materials,
textures, object/goal pose, collision, task logic, or robot state.

Domain assignment is fixed by the pre-existing trajectory split before any
image is rendered:

| Collection/split | Domains |
|---|---|
| M3A train | canonical, mild camera, mild lighting |
| M3A validation | canonical, mild camera, mild lighting |
| M3A test | canonical, strong camera, strong lighting |
| M3C external | canonical, strong camera, strong lighting |

Strong domains never enter development training or validation. Mild domains
never substitute for held-out M3A test or M3C external evaluation.

## Packet, image, and sample identities

A `VisualObservationPacketV1` binds one anchor/domain render and its three
ordered views. Its semantic packet ID binds the source trajectory, anchor,
state, rig, domain, render seed, ordered camera identities, visual compatibility
identity, and schema version. It excludes output paths, hosts, PIDs, and time.

Each view records both the raw RGB pixel digest and the exact authoritative NPY
file digest. The visual dataset digest binds packet semantic identities and all
exact image digests. NPY loading always uses `allow_pickle=False`; paths are
strict relative POSIX paths, and traversal, links, duplicate references,
tampering, and orphan files are rejected.

The packet also content-binds the observed 38-component verifier comparison,
its exact zero error, the elapsed-step values before and after rendering, and a
successful environment close. Runtime intrinsics and extrinsics dtypes are
observed by the compatibility probe and bound into its identity; calibration is
then compared exactly after casting the frozen planned matrices to those single
declared dtypes. Validation never searches across several dtypes or relaxes the
calibration comparison dynamically.

Images are stored once per anchor/domain/view. Candidate records reference the
three packets assigned to their anchor without copying image or action arrays.
The single-packet visual model examples are a derived expansion:

- M3A has 3,240 physical candidate bindings and 9,720 candidate-domain
  examples;
- M3C external has 2,880 physical candidate bindings and 8,640
  candidate-domain examples.

## Student inputs and privileged metadata

The future visual-student allowlist is limited to ordered RGB views, the
candidate action chunk, its action mask, and optionally the fixed canonical
task text if a later model milestone authorizes text consumption. M4A fixes the
task identity to `maniskill/PickCube-v1` and the exact text to
`Pick up the cube and place it at the goal.`; other task IDs or wording are
rejected rather than normalized.

Domain/camera IDs, collection, split, trajectory, packet/evidence IDs,
corruption metadata, and outcomes are reporting-only. The 38D verifier state,
complete simulator state, object/goal pose, joints, and task snapshot are
privileged teacher information and are absent from visual-only model inputs.

## Development and external datasets

`VisualVerifierDevelopmentDatasetV1` is bound to the accepted M3A dataset and
its exact 48/6/6 trajectory split. All variants of one anchor remain in that
trajectory's split.

`VisualVerifierExternalDatasetV1` is bound to the accepted M3C source set,
candidate pool, outcome-blind manifest, strong replay evidence, and full
outcomes. It has `training_allowed=false` and cannot be opened through a
training-mode loader. External images and outcomes cannot tune rendering,
preprocessing, architecture, hyperparameters, calibration, thresholds, or
checkpoint selection.

Cross-dataset validation rejects overlap in source seeds, trajectories,
split-group IDs, anchors, complete/verifier states, packets, candidates, and
image content. Repeated pixels across different physical states are reported
and investigated rather than silently ignored.

## Determinism, resume, and storage

The compatibility probe first requires exact repeated-render and
fresh-environment pixel equality and reports changed pixels plus maximum and
mean channel error. Any nondeterminism stops trusted generation pending a narrow
review; M4A defines no pixel tolerance.

The probe report identifies the exact source archive, episode, trajectory,
reset seed, state index, state content/state/verifier digests, and episode
content digest used for the check. These path-independent source facts are part
of the visual compatibility identity; runtime archive paths are not.

Rendering uses a content-bound ledger and transactional packet publication.
Interrupted work can be recovered without overwriting completed packet bytes;
configuration/source drift and orphan packet stores are rejected, deterministic
contract violations are non-retryable invalid context, operational errors are
not task failures, and a completed resume must prove zero duplicate packets.

Formal probe output is one absent output root containing exactly
`visual-compatibility-report.json` and `run-manifest.json`; both files are
strictly reloaded in a staging directory before that directory is atomically
published. A failed exact-pixel trust gate still preserves both evidence files
and fails the command. A render root similarly publishes its ledger, complete
render-job inventory, and operational manifest in one staging transaction.

The operational manifest records the clean full Git SHA and attached branch,
rejects tracked and untracked checkout drift, and binds hostname, Python,
PyTorch, CUDA, one RTX 5090 and its driver, ManiSkill, SAPIEN, renderer backend,
visual/rig/domain identities, fixed render seed, exact source variant, the same
canonical source-identity digest as the complete render-job inventory, and the
exact ordered selected-packet digest.
Probe manifests instead bind the complete path-independent archived-state
evidence and carry no render-job inventory. These operational fields never
enter packet, image, or dataset semantic identities. Resume reconstructs the
expected manifest from the current environment and source while retaining the
original start time and sanitized initial command; any drift fails before
staging recovery or ledger mutation. Dry runs collect no operational context
and create no output.

On a complete zero-work resume, the render command immutably persists a report
bound to the unchanged ledger, canonical source identity, rig, and domain
configuration. `validate-visual-verifier-dataset --report-dir <absent-path>`
publishes only a fixed inventory of strict, reloaded JSON reports after every
full-target source, image, integrity, and leakage gate passes; `--allow-partial`
cannot publish acceptance reports. It includes compact camera/domain summaries,
the split-by-domain cross-tab, a digest over the exact ordered image inventory,
the probe's exact pixel determinism, and count/minimum/median/nearest-rank-p95/
maximum/zero-count statistics for complete-state and verifier-state restoration
errors. It includes only a live resume report that still matches the render
root. The external
training-prohibition report comes from an actual rejected training-loader
call. Validation does not invent determinism or resume evidence that was not
observed by those workflows.

Raw NPY datasets, state archives, actions, and full evidence remain outside
Git. Only compact sanitized identity/count/integrity/determinism/leakage/resume
reports may be retrieved. Building and validating these controlled visual
inputs before VLM integration isolates renderer and physical-state errors from
model behavior and prevents an unbounded language/vision stack from obscuring
the provenance boundary.
