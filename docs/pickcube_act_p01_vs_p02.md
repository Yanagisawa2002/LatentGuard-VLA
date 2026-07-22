# PickCube native ACT: P0.1 versus P0.2

P0.2 preserves P0.1's most important positive result: end-to-end native ACT
actions remain finite and strictly inside the unchanged `pd_joint_pos` bounds
without clipping, projection, fallback, or action replacement. It changes only
the supervision and development-time execution horizon justified by the
diagnostics.

| Dimension | P0.1 | P0.2 |
| --- | --- | --- |
| Policy family | ACT | ACT |
| Observations | RGB + 18D Panda state | unchanged |
| Action | 8D absolute `pd_joint_pos` | unchanged |
| Output transform | `affine_tanh_v1` | unchanged |
| Chunk length | 16 | unchanged |
| Execution horizon | 4 | 2, selected from fixed 1/2/4 audit |
| Supervision | ordinary action L1 | separated arm/gripper/transition losses plus capped train-only phase weights |
| Full steps | 20,000 | 20,000 |
| Complete checkpoints | 15 | 14 |
| Selected checkpoint basis | early action-legal survivor | offline top-3 plus identical 10-seed behavioral screen |
| Development pre-grasp | not instrumented as a passing gate | 28/30 |
| Development valid close | not instrumented as a passing gate | 20/30 |
| Development grasp | 0/30 | 1/30, then dropped |
| Development lift | 0/30 | 0/30 |
| Development success | 0/30 | 0/30 |
| Action violations | 0 | 0 |
| Final sealed 100 | untouched | untouched |
| Package / D2 | none / blocked | none / blocked |

The evidence supports a narrow improvement: P0.2 removed the global gripper
endpoint collapse, improved approach and near-object closing, and produced one
verified grasp. It does **not** support grasp acquisition, lift competence, or
task-level policy improvement. Contact-to-grasp stability remains the dominant
failure (12/30 contact-without-grasp), and the sole grasp failed before lift.

The earlier P0.1 milestone used its own historical Result B label for a legal
but rejected policy. Under P0.2's explicit A/B/C grasp taxonomy, the new result
is Result C because the 18/30 grasp gate was not met. Neither result is an
accepted compatible policy.
