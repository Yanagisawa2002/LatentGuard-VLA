# LG-RB0 results

## Decision

LG-RB0 is **Result C**. `LG_RB1_AUTHORIZED=false`.

The frozen RoboLab stack launched and generated ten structurally valid
mechanics recordings, but official recorded-config faithful replay did not
complete for any of the registered 30 attempts. The takeover gates were
therefore not run. This is a negative simulator-infrastructure result, not a
policy result and not evidence that RoboLab task dynamics are nondeterministic
after a successfully completed replay.

LG-R2b0 Result C remains frozen and unchanged. LG-RB0 is a separate RoboLab
feasibility route and does not reinterpret the LIBERO evidence.

## Runtime identity

- RoboLab: `v0.2.1`,
  `0aef241fb088ca21bb4ebd24448940ed56620d17`;
- Python `3.11.15`;
- Isaac Sim distribution `5.1.0.0` (PEP 440-equivalent to `5.1.0`);
- Isaac Lab `2.3.2.post1`;
- PyTorch `2.7.0+cu128`, CUDA runtime `12.8`;
- NVIDIA GeForce RTX 5090, `34190917632` bytes, driver `580.76.05`;
- one headless `cuda:0` environment using `nvidia_icd.json`.

The two one-step GPU smokes passed for `BananaInBowlTask` and
`RubiksCubeAndBananaTask`. This establishes only environment mechanics and
task registration.

## Recording gate

The process-isolated recording gate passed:

- 10/10 valid HDF5 episodes;
- five seeds `810000..810004` for each of the two registered tasks;
- exactly 40 actions per episode;
- deterministic `fixed_joint_hold_gripper_toggle_v1` mechanics controller;
- all HDF5, environment sidecar, and outcome files content-bound in the compact
  manifest;
- 0/10 terminal and 0/10 successful at the recorded horizon.

The last line is expected for the fixed mechanics probe and must not be
interpreted as policy evaluation. No policy or candidate source produced these
actions.

## Faithful replay gate

The registered 10 episodes x 3 repeats all ran through the evidence harness:

| Measure | Result |
| --- | ---: |
| expected replay attempts | 30 |
| fully completed replays | 0 |
| strict initial-restore failures | 30 |
| recorded-config overlay failures | 30 |
| replay execution errors | 30 |
| official/per-step comparisons completed | 0 |
| terminal comparisons completed | 0 |
| success comparisons completed | 0 |

The numeric zero fields for per-step, terminal, and success mismatches have a
zero comparison denominator. They are unavailable metrics, not evidence of
zero mismatch.

All 30 attempts, across both tasks and all seeds, had the same evidence:

1. Every stored articulation and rigid-object leaf matched at initial restore
   with identical dtype and shape and maximum absolute error `0.0`.
2. The strict complete-tree comparison failed because the live state also
   contained `deformable_object/<empty-mapping>` and
   `gripper/<empty-mapping>`, while the recorded tree omitted those empty
   categories.
3. Official recorded-config overlay skipped
   `/_instruction_variants (not in current config)`.
4. Recorded subtask condition partials were serialized as strings. After the
   official value overlay, the first replay action reached the subtask state
   machine and raised `TypeError: 'str' object is not callable`.

The fourth item prevented official per-step validation, terminal comparison,
success comparison, and observation comparison from completing. It is recorded
as an execution error, never relabeled as task failure or state mismatch.

## Takeover gates

Because faithful replay failed, the frozen stop rule applied:

- early/middle/late prefix replay: `not_run`;
- A/B same-suffix determinism: `not_run`;
- A-B-A and B-A-B isolation: `not_run`;
- semantic coverage: unavailable;
- intermediate takeover: not established.

Their mismatch counts are `null`, not zero. No prefix, branch, or isolation
claim is supported.

## Audit and prohibitions

The compact remote audit binds the exact commits, run IDs, exit codes, and log
digests for both task probes, the recording gate, and the faithful gate. Both
the LatentGuard execution checkout and the external RoboLab checkout were clean
at final audit.

No VLA or other policy was integrated. Candidate generation, sampling,
ranking, intervention, optimizer steps, backward passes, training,
checkpoints, PointWorld, ROS 2, LangMani access, and final seeds
`900000..900099` were not used.

Large RoboLab assets, HDF5 recordings, simulator state, caches, and logs remain
outside Git. Only compact reviewed JSON evidence is committed.
