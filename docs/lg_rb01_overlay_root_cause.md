# LG-RB0.1 recorded-config overlay root cause

## Bound evidence

The affected upstream is RoboLab `v0.2.1` at exact commit
`0aef241fb088ca21bb4ebd24448940ed56620d17`. The evidence set is the unchanged
LG-RB0 manifest and its ten recordings: two tasks, seeds `810000..810004`, and
40 fixed mechanical actions per recording.

LG-RB0 completed zero of 30 faithful attempts. Every attempt reached the same
`TypeError: 'str' object is not callable`; per-step, terminal, and success
denominators therefore remained unavailable.

## Callable corruption

The serialized callable leaves occur under:

- `/subtasks[0]/conditions/banana[0][0]`
- `/subtasks[0]/conditions/banana[1][0]`
- `/subtasks[0]/conditions/rubiks_cube[0][0]` on the two-object task
- `/subtasks[0]/conditions/rubiks_cube[1][0]` on the two-object task

The recording stores strings such as
`functools.partial(<function object_grabbed at 0x...>, object='banana')`.
The live config contains `functools.partial` objects at those leaves.

Contrary to the original missing-key hypothesis, the real condition key is
present. RoboLab `_overlay()` sees a recorded list with no dict elements and
takes its plain-list wholesale-assignment branch. That replaces the complete
live list of `(partial, score)` tuples with the lossy JSON list. On the first
subtask update, `ConditionalsStateMachine.check_condition_satisfied()` executes
`conditional_func(**params_with_env)`, where `conditional_func` is now the
recorded string. Missing-key callable injection is a separate unsafe old branch
and remains covered by a regression fixture; it is not presented as the real
recording root cause.

## Instruction metadata

`env_cfg.json` contains both a resolved `/instruction` string and
`/_instruction_variants`. At overlay time, `parse_env_cfg()` has not created
the private runtime field. `create_env()` creates it later from the live
instruction variants before resolving the instruction. Restoring the lossy
private field is neither necessary nor safe. LG-RB0.1 therefore preserves
`/_instruction_variants` as expected runtime metadata while still restoring
the recorded resolved `/instruction`.

## State schema difference

The recorded and live articulation and rigid-object leaves have identical
paths, dtypes, shapes, and initial numeric values. The live tree additionally
contains exactly:

- `/deformable_object` = `{}`
- `/gripper` = `{}`

They are canonicalized only when the exact namespace is allowlisted and the
mapping is strictly empty. Non-empty values and unknown empty namespaces fail.
The post-patch audit covered 1,230 comparisons (initial state plus 40 steps
across 30 attempts). Both live-only mappings were empty and removed in every
comparison, canonicalization was symmetric, and no numeric value changed.

## Root-cause boundary

This is a recorded-config compatibility defect: JSON can restore declarative
values but cannot restore executable code identity. No evidence supports a
physics, solver, action, task-predicate, recording, or tolerance change.

The patch removes this blocker: all 30 overlays and replays complete without a
callable exception. It does not make repeated replay faithful. Repeat zero
matches exactly for all ten recordings, while repeats one and two drift at all
40 checked steps for every recording. This repeat-dependent state drift is a
separate, confirmed blocker; its exact internal source was not localized
because the milestone permits only one compatibility patch.
