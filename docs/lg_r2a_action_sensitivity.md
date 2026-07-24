# LG-R2a action-sensitivity protocol

Model C is evaluated with a deterministic real-action permutation that keeps
the current representation fixed while sourcing an executed action from
another episode of the same task, nearest current-progress bin, and identical
effective horizon. These actions are never training examples for the target
state and are not synthetically perturbed.

A tighter swap surrogate also prefers the same frozen stage and nearest
progress bin. It is an offline sensitivity surrogate, not a true same-state
counterfactual: the substituted action was executed in a different physical
state.

Three structured ablations retain only the first action, repeat the endpoint
action summary, or mask the complete action. Absolute input gradients of the
short-progress output provide a simple per-dimension attribution. All
permutation, swap, ablation, and attribution operations are test-time
diagnostics only and never alter training data.

The final evidence revision will report prediction change, performance
degradation, event-specific AUPRC changes, and whether the frozen sensitivity
gate passes.
