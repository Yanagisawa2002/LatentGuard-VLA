# LG-RB0 replay contract

## Frozen identities and exclusions

The executable protocol is
`configs/lg_rb0/protocol.yaml`. The RoboLab source, Python, Isaac Sim, and
Isaac Lab versions are content-bound before any simulator result is accepted.
Every run uses one headless CUDA environment. The LatentGuard checkout must be
clean and exactly match the pushed milestone commit; the RoboLab checkout must
be clean and exactly match the frozen external commit.

No policy, checkpoint, learned scorer, candidate generator, or ranker is loaded.
The A/B suffixes are registered mechanics probes, not deployable policy
candidates. No outcome is used to change an action, anchor, tolerance, task,
seed, or repetition count.

## Recording set

The recording phase generates exactly ten episodes:

- `BananaInBowlTask`, seeds `810000..810004`;
- `RubiksCubeAndBananaTask`, seeds `810000..810004`.

For 40 steps, the controller holds the current seven arm joint positions and
toggles only the legal gripper joint command every ten steps. Recordings may
succeed or fail the task; their purpose is deterministic mechanics validation,
not policy-quality measurement. HDF5 actions, initial state, per-step state,
environment sidecar, terminal/success outcome, and provenance are retained
outside Git and content-bound in the compact recording manifest.

## Faithful full replay

Each of the ten recordings is replayed three times in a fresh simulator
session using its recorded environment configuration, recorded initial state,
and complete action sequence. The official maximum-absolute-error tolerance is
fixed at `0.01`.

The gate requires:

- initial restore failures: zero;
- missing or unexpected compared state paths: zero;
- per-step state failures: zero;
- terminal mismatches: zero;
- success mismatches: zero.

The tolerance may not be relaxed after observing results. The extra ten
repeated final actions used by RoboLab's example are excluded; LG-RB0 compares
exactly the recorded action horizon.

## Prefix reconstruction

For each recording, the early, middle, and late anchor index is the floor of
25%, 50%, and 75% of the recorded action count, clamped to
`1..action_count-1`. Each anchor is reconstructed three times in a fresh
session by restoring the initial state and executing the exact recorded prefix.

The comparison includes the complete relative articulation and rigid-object
state tree, observation digest, task predicates, subtask status, local event
log, termination latch, and success latch. State paths, dtypes, shapes, and
finite values must match exactly; numeric maximum absolute error must be at
most `1e-6`. Every semantic field must be byte-stable after canonical JSON
encoding. Any missing semantic channel makes semantic coverage incomplete and
prevents Result A.

## Fixed branch suffixes

From every reconstructed anchor:

- branch A holds the anchor arm joint positions and commands open gripper;
- branch B holds the anchor arm joint positions and commands closed gripper.

Each branch runs for ten steps and is repeated three times in fresh sessions.
Complete state and semantic snapshots are compared at suffix steps 1, 5, and
10 using the prefix contract. A and B are not required to differ; LG-RB0 tests
same-branch determinism, not action identifiability.

## Isolation

At each anchor, execute fresh-session orders A-B-A and B-A-B. The two A runs
and the two B runs must reproduce their same-branch reference snapshots at
steps 1, 5, and 10. Any mismatch is branch contamination. Complete simulator
recreation is preferred; if the runtime cannot safely recreate fresh sessions,
the takeover gate fails rather than weakening isolation.

## Result and stop logic

- Result A: at least ten valid episodes and zero faithful, prefix, branch, or
  isolation mismatch with complete semantics and valid provenance. Only LG-RB1
  design is authorized.
- Result B: official faithful replay passes, but deterministic intermediate
  takeover is absent, incomplete, or unsafe. RoboLab cannot take over as the
  counterfactual data source.
- Result C: official faithful replay fails under the frozen supported stack.
  Stop the RoboLab route.

Neither successful environment launch, recording generation, initial restore,
low state error, prefix reconstruction alone, deterministic branch execution,
nor a task success can be described as policy success or as authorization for
training.
