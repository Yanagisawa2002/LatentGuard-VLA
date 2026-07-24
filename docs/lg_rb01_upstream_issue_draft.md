# Upstream issue draft — withheld

> **Status:** not ready to submit. LG-RB0.1 passed the focused overlay tests but
> failed the full repeated faithful-replay gate. This text is evidence-preserving
> preparation only; no upstream issue was created.

## Title

Recorded replay overwrites live callable containers with lossy JSON lists

## Affected revision

- RoboLab `v0.2.1`
- Exact commit: `0aef241fb088ca21bb4ebd24448940ed56620d17`

## Minimal reproduction

1. Construct a live replay config whose condition table contains a list of
   `(functools.partial(...), score)` tuples.
2. Overlay a recorded JSON config in which the same list contains serialized
   callable strings.
3. Observe that the plain-list branch assigns the recorded list wholesale.
4. Execute the first condition check.

Observed exception:

```text
TypeError: 'str' object is not callable
```

The call occurs when the condition state machine evaluates
`conditional_func(**params_with_env)`.

## Recorded/live difference

At condition leaves such as
`/subtasks[0]/conditions/banana[0][0]`, recording JSON contains a lossy string
representation of `functools.partial`, while the live config contains the
callable partial object. The key exists in the real config; the destructive
operation is wholesale list replacement, not missing-key insertion.

`/_instruction_variants` is runtime-owned metadata. The recorded resolved
`/instruction` remains useful, but the lossy private metadata should not
replace live runtime objects.

## Root cause

The existing list overlay assumes lists contain declarative data. For lists
containing runtime code, JSON cannot preserve object identity, so wholesale
replacement violates the replay documentation's separation between recorded
values and live code structure.

## Candidate fix

Preserve live callable containers, fail closed on incompatible callable
representations, keep runtime instruction variants live, and allow
recorded-only pure data only under explicit declarative namespaces. Do not use
`eval`, `exec`, or arbitrary string execution.

Focused regression tests pass and the callable exception is eliminated in
30/30 real replay attempts. However, later repeats exhibit unrelated state
drift, so this draft is intentionally withheld rather than presented as a
fully validated upstream fix.
