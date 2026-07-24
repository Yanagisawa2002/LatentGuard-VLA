# Upstream PR draft — withheld

> **Status:** not ready to submit. No upstream branch was pushed and no PR was
> created. The focused compatibility behavior passes, but the complete repeated
> faithful-replay gate does not.

## Title

Preserve live callable code during recorded-config replay overlay

## Problem

Recorded JSON stores callable-bearing condition containers as lossy strings.
The current plain-list overlay replaces live lists of callable tuples
wholesale, and the condition state machine later calls a string.

## Design

- Treat recorded configuration as declarative data, not executable code.
- Preserve a live callable or callable-bearing container.
- Use safe callable resolution only to compare a stable identity; retain the
  live callable object.
- Reject unresolved or incompatible callable values.
- Reject recorded-only runtime-code keys.
- Allow recorded-only pure data only within explicit declarative namespaces.
- Preserve live `/_instruction_variants` while restoring the recorded resolved
  `/instruction`.
- Report preservation, schema drift, and invalid values as distinct classes.
- Use no `eval`, `exec`, arbitrary string execution, or task-specific repair.

## Why live code is preserved

JSON can encode values but cannot faithfully reconstruct Python callable
identity, closures, partial arguments, or code ownership. The current source
tree is the compatible authority for executable objects; the recording is the
authority for serializable replay values.

## Tests

Focused tests cover:

- the old callable-container failure;
- missing condition-key fail-closed behavior;
- existing callable identity preservation;
- safe importable and unimportable callable paths;
- explicit recorded-only pure data;
- runtime namespace preservation;
- resolved instruction consistency;
- repeated patch rejection and unpatched checkout detection.

The focused external test file passes 7/7, and old/patched integration
regressions each pass their seven expected cases. In real replay, overlay,
environment creation, and replay complete 30/30 with zero callable exceptions,
fatal skips, unexpected skips, or execution errors.

## Compatibility boundary

The patch changes only recorded-config overlay behavior and documentation. It
does not change tasks, actions, recordings, physics, solver behavior, or
tolerances.

Full LG-RB0.1 promotion is withheld: repeat zero is exact for ten recordings,
but repeats one and two fail state validation. Prefix and branch gates were
therefore not run. Upstream submission requires separate authorization and a
decision about whether focused overlay correctness is sufficient despite that
independent platform-level replay limitation.
