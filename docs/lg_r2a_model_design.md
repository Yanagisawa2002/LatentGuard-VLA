# LG-R2a model design

LG-R2a uses one small, fixed probe family. It performs no foundation-model
fine-tuning and no architecture or fusion search.

Model A widens the state-only hidden layer to control trainable parameter count.
Model B receives only the flattened 7x7 numeric action plus its seven mask
bits. Model C separately maps the 2,048-dimensional frozen current
representation and the action input, concatenates them, and applies one
action-derived gate. Model D adds a 16-dimensional embedding of the current
pre-execution 8-dimensional proprioceptive state. Model E is optional and
cannot enter the gate.

The action encoder is `56 -> 64 -> 32` with LayerNorm and GELU. It is far below
the 500,000-parameter cap. A and C must be within 5% total trainable parameter
count; the exact values, per-sample inference latency, peak GPU memory, and
initialization seed are recorded by each run.

The shared output has:

- three progress-delta values for 7/21/49 steps;
- three stagnation logits;
- three regression logits;
- `OBJECT_DROP` and `FAILED_PLACEMENT` logits;
- one terminal-failure logit.

The loss weights are frozen at 1.0 for Huber progress and 0.5 for each binary
group. BCE positive weights are computed from each train fold only and capped
at 50. AdamW uses learning rate 0.0005, weight decay 0.0001, batch size 128,
gradient clipping 1.0, seed 0, and validation-only early stopping. Test
predictions are produced only after checkpoint selection.

The current VLA representation and current proprioception are standardized
using train-fold statistics. Numeric actions retain their frozen executed
scale. No task ID, current progress, stage, outcome, future observation, actual
remaining length, ROBOMETER score, TOPReward score, or SARM output is supplied
to a probe.
