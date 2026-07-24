# LG-R1c reward stack audit

The benchmark pins the LeRobot `v0.6.0` reward integrations at commit
`30da8e687a6dfc617fcd94afc367ac7071c376ce`. Upstream source review is bound
to ROBOMETER commit `5b815254bf31ee1bea3753c3a2da9f9033736d9a`
(Apache-2.0) and TOPReward commit
`4877a0ee5098cbec18485125466631b1dcc4a573` (MIT). Exact model and file
identities are in `artifacts/lg_r1c/reward_stack_manifest.json`.
Integration source hashes use canonical LF bytes so Windows checkout line
endings cannot create a false source-drift failure on Linux.

## ROBOMETER

The LeRobot checkpoint is `lerobot/Robometer-4B` revision
`db167a7c369a3ee59cda801fe33ca9da560b1662`. Its single 8.89 GB safetensors
file is hash-bound. The processor/tokenizer is Qwen3-VL-4B-Instruct revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.

The published port uses ten discrete progress bins and a sigmoid success head.
It contains a preference head but `compute_reward()` does not query it.
Internally the model produces per-frame progress and success logits.
LeRobot's public `compute_reward()` decodes those logits but returns only the
last-frame scalar; for success it additionally thresholds the probability.
LG-R1c therefore uses a read-only adapter around the official private logits
and official decode function. It asserts an eight-element output for every
eight-frame input and never changes the model.

## TOPReward

The default backbone is `Qwen/Qwen3-VL-8B-Instruct` revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; its four weight shards and
processor files are hash-bound. The fixed score is the log probability of the
terminal literal `True` in:

```text
The above video shows a robot manipulation trajectory that completes the following task:
{instruction} Decide whether the above statement is True or not. The answer is: True
```

The zero-shot normalized score is `exp(raw_log_probability)`. TOPReward
produces one scalar per video window, not a per-frame curve. Curves therefore
require repeated rolling-window inference, which is reported as additional
cost rather than misrepresented as a native frame output.

## Compatibility and limitations

Both LeRobot ports are inference-only in this milestone. No ROBOMETER,
TOPReward, Qwen, VLA-JEPA, policy, or processor parameter is trainable.
ROBOMETER consumes at most eight frames. The fair comparison uses eight frames
for both models; TOPReward receives a separately labeled 16-frame native
diagnostic on long anchor windows. Model revision, processor revision,
tokenizer revision, checkpoint hashes, prompt, sampling, and normalization
all fail closed.

An already populated remote snapshot may be reused offline only when every
root file has Hugging Face local-directory metadata bound to the declared full
revision. Weight files then pass the pre-registered size and SHA-256 checks
again before model loading. This permits an identical official ModelScope
weight mirror without weakening the HF revision or content gates.
