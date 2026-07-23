# LG-R0 VLA-JEPA architecture audit

## Policy path

The official checkpoint combines:

1. Qwen3-VL-2B for two-view vision/language fusion;
2. a state encoder for the eight-dimensional LIBERO proprioceptive vector;
3. a DiT-B flow-matching action head;
4. an action chunk of shape `[batch, 7, 7]`;
5. four inference integration steps.

The raw action head output is normalized action space. Official checkpoint
postprocessing then clips normalized values, pre-snaps the gripper,
unnormalizes with checkpoint statistics, and binarizes the gripper. The
LatentGuard adapter returns the raw tensor unchanged; closed-loop execution
uses the official postprocessor and identifies the two forms separately.

## World-model training path

Eight temporal frames from each of two views are encoded by
`facebook/vjepa2-vitl-fpc64-256`. Tubelet compression produces four temporal
positions. The official shift-by-one split uses positions 0–2 as context and
positions 1–3 as target. Each temporal position contains 14×14 visual tokens.
The two views are concatenated in the embedding dimension.

`ActionConditionedVideoPredictor` projects the context visual tokens, interleaves
Qwen action-token conditions, runs 12 causal transformer blocks, and projects
back to the V-JEPA embedding dimension. Its output is future visual-token
predictions, not RGB, reward, success, progress, or risk. Mean L1 against the
shifted target tokens is the only official scalar.

The word “action-conditioned” is narrower than an arbitrary action-candidate
API: the condition is Qwen special-token hidden states produced from the
vision/language prompt. The predictor never receives the generated numeric
action chunk. Consequently:

- default policy inference does not run the world model;
- external numeric actions cannot be scored;
- model-native multi-candidate ranking is unavailable;
- a temporal offline diagnostic can expose current, predicted, and target
  tensors without gradients;
- a later failure head may treat those representations plus the separately
  recorded numeric action chunk as inputs, but that head does not yet exist.

## Checkpoint load result

The model has 2,770,333,319 parameters. `requires_grad` remains true for the
published architecture, but LG-R0 creates no optimizer and calls no backward
pass. The detailed per-module counts and GPU memory are in
`artifacts/lg_r0/model_structure.json`.

Fail-closed loading has zero missing keys. One reviewed non-critical duplicate
is reported by safetensors:

```text
model.qwen.model.model.language_model.embed_tokens.weight
```

The checkpoint also stores `model.qwen.model.lm_head.weight`. LG-R0 verifies
the two `[151936, 2048]` BF16 tensors are exactly equal in chunks and verifies
the runtime parameters are tied to one storage. No other unexpected key is
accepted.

The official checkpoint contains trained weights for both
`model.video_encoder` and `model.video_predictor`; their presence and strict
shape compatibility are proven by the fail-closed state load. “Contains
weights” does not imply that the branch runs at inference time.

## Available score and next representation

There is no public risk, success, progress, or reconstruction score. The
training L1 is computable only when a real future temporal window is available.
For a future supervised LatentGuard failure head, the defensible inputs are:

- current V-JEPA visual tokens;
- predicted future V-JEPA visual tokens;
- Qwen special action-token hidden states;
- the separately recorded generated numeric action chunk;
- target future V-JEPA tokens only as a training/evaluation target, never as an
  online deployable input.
