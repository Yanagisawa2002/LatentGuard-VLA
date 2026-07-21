# WM-v0 design

WM-v0 is a simulator-independent, action-conditioned feature dynamics model.
The integration supplies only verified RGB observations, an allowlisted
proprioceptive vector, action rows, and task labels. Complete simulator state is
used to restore and verify collection boundaries but is never a learned input.

## Data flow

1. Restore one content-bound anchor in a fresh session and verify every state
   component under the adapter's fixed comparison semantic.
2. Render the fixed ordered current views without stepping physics; verify that
   rendering did not change complete state, task facts, or proprioception.
3. Execute one immutable candidate. At the configured stride, capture real
   future RGB, proprioception, progress, and observable event channels.
4. Restore the same anchor again and require the same content identity before
   accepting the sample.
5. Split by source trajectory before derived samples. Unknown event channels
   have a false availability mask; their stored value has no label meaning.
6. Encode current and future images once with the frozen, content-bound M4B
   ResNet-18 and train from the immutable feature cache.

## Model

The model projects each current view, current proprioception, and each masked
action row into a common token dimension. Learned future queries attend over
those inputs through a compact Transformer encoder. Each query predicts the
ordered future view features, progress, event logits, and a non-negative
uncertainty proxy; the final query also predicts terminal success. Prediction
horizon is configurable up to the model's declared maximum.

The loss is normalized feature cosine distance plus masked smooth-L1 progress,
masked class-weightable event BCE, terminal-success BCE, and a small detached
latent-error uncertainty target. The outcome-only ablation uses the identical
backbone and training budget with the future-feature term disabled.

## Fair comparison

- **Direct verifier:** the accepted M3B temporal 38-D state plus 16-by-8 action
  architecture, trained/evaluated on the same new sample groups.
- **Matched outcome-only:** the WM-v0 token backbone and outcome heads without
  future-feature supervision.
- **WM-v0:** the same backbone with real future-feature supervision.

All variants use identical source-group splits, candidate inventories, step
budgets, checkpoint-selection data, and test data. Candidate rankings are
compared only inside one anchor group and are reported separately for
`policy_generated` and `synthetic_corruption` sources.

## Closed-loop boundary

The optional probe is disabled in the checked-in evaluation configuration. If
later authorized by a passing offline gate, the primary action remains the
default. An alternative may be selected only when primary risk exceeds a frozen
threshold, the alternative clears a fixed margin, uncertainty stays below a
fixed ceiling, cooldown has elapsed, and the intervention budget remains.
