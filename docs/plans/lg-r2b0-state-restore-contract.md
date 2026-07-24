# LG-R2b0 LIBERO complete state restoration contract

The branch start is valid only when a fresh environment is restored to the
content-bound anchor and every captured component satisfies the registered
comparison.

The snapshot includes:

- the control environment's flattened simulator state;
- complete exposed MuJoCo runtime arrays used by execution;
- numeric controller runtime attributes;
- wrapper, control-environment, and task termination/success/timestep latches;
- Python, NumPy, CPU Torch, CUDA Torch, and exposed environment RNG states;
- a complete array inventory with dtype and shape;
- content and structure SHA-256 identities.

Restoration rejects missing or added components, dtype/shape drift, non-finite
values, unavailable latches/RNG owners, and content corruption. It does not
clip, normalize, infer, or repair state.

The frozen comparison semantic is
`tolerance_verified_complete_runtime_state_v1` with numeric tolerance `1e-6`.
Runtime scalar/RNG JSON must match exactly. Rendered observation bytes, task
predicates, stage identity, termination, and success must match exactly.

The pre-pilot audit uses 10 registered states and 5 repeated 49-step executions
per state. Zero restoration failures, zero predicate mismatches, and zero
terminal mismatches are permitted. This gate is infrastructure evidence only;
it is not task performance evidence.
