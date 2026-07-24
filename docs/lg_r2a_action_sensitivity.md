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

The action-magnitude diagnostic additionally reports no-fit correlations for
translation, rotation, pose magnitude, and gripper switching. It repeats the
correlations after subtracting same-task/current-progress-bin means. This
cannot prove absence of an action-statistics shortcut; in particular,
Action-only lower MAE with weaker rank correlation is treated as a shortcut
concern rather than state-conditioned consequence evidence.

The final real-action permutation worsened short-progress MAE by 8.58% and
reduced regression AUPRC by 0.0839. The latter passes the sensitivity
alternative, but terminal AUPRC fell by only 0.0075 and failed-placement AUPRC
improved under permutation. The cross-episode swap surrogate changed the
short-progress prediction by mean absolute 0.0164.

Full masking, first-action-only, and endpoint-summary short MAE were 0.06945,
0.05972, and 0.04109 respectively, versus 0.04060 with the full chunk. These
results establish model action use, not stable state-conditioned consequence
value.

The simple magnitude correlations are near zero after task/progress-bin
control, but Action-only's lower MAE and weaker rank correlation leave a
near-zero-delta/action-statistics shortcut concern. The overall sensitivity
gate passes; the full LG-R2b gate does not.
