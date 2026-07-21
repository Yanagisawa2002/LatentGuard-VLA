# PickCube native ACT training report

## Data and training authorization

The native expert gate passed with 95 successes in 100 independent PickCube-v1
seeds and zero simulator errors, non-finite actions, out-of-contract actions,
or workspace violations. The authorized collector then retained 500 complete
successful episodes from 517 attempts. The dataset contains 21,053 frames and
is split by unique episode/scene seed into 400 train, 50 validation, and 50
test episodes. Duplicate episodes, missing observations, non-finite values, and
cross-split leakage are all zero.

The content-bound dataset digest is
`sha256:234b82709359f1f2fde513ab26c015f4ebfb262a16b1f16206c54eddbde44800`.
The normalization digest is
`sha256:8093f5bba3916dff218e73789739b18e05e9eb96cfc200fa18747f010959532a`.
Raw demonstrations and images remain outside Git.

## Model and optimization

The single screened model is standard LeRobot 0.6 ACT. Its deployable inputs
are one 224 x 224 `front_oblique` RGB image and the current 18-dimensional
named Panda joint position/velocity vector. It predicts a 16 x 8 future action
chunk and executes four actions before the next query. Privileged cube pose,
goal pose, expert phase, outcome, provenance, and task labels are not model
inputs.

The model uses a ResNet-18 image encoder without downloaded pretrained weights,
512 hidden dimensions, eight attention heads, four encoder layers, one decoder
layer, a 32-dimensional CVAE latent with four VAE encoder layers, dropout 0.1,
and KL weight 10. Training uses mean/std normalization, AdamW at 1e-5, batch
size 32, bfloat16, deterministic seed 0, validation every 1,000 steps, and a
20,000-step ceiling. Configuration, CPU reload, GPU forward,
forward/backward, short overfit, checkpoint save/load/resume, and the initial
closed-loop mechanism smoke all passed before the full run.

## Formal run

- Run ID: `20260721T104200Z_wm-v0-p0-act-full_ef24b14_seed0`
- Training source: `ef24b14529317ee8fc240307be77fcbe94a0cf31`
- Runtime: Python 3.12.3, LeRobot 0.6.0, PyTorch 2.11.0+cu130
- GPU: NVIDIA GeForce RTX 5090
- Wall time: approximately 33 minutes (18:40:10 to 19:13:14 UTC+8)
- Completed steps: 20,000 / 20,000
- Processed training examples: 638,853
- Peak allocated/reserved GPU memory: 1,813,110,272 / 2,113,929,216 bytes
- Best validation loss: 0.10424631055105817 at step 20,000
- Checkpoints retained: 16 distinct checkpoint directories

The genuine role assignment is early=step 1,000, mid=step 10,000,
best-validation=step 20,000, and final=step 20,000. Best and final refer to the
same actual final checkpoint; no file was copied and renamed to manufacture a
role. The final checkpoint manifest digest is
`sha256:b5df38aef975aad127fd5d5e65cf047b14c5eacbb019a16acdb42cf87d75e54c`
and its 206,413,016-byte model has digest
`sha256:fc05cd5777091852c2e9840eb0a930dedf0da9b4329fee8323bfbd7325b18bcd`.

Checkpoint reconstruction loss is not a deployment claim. Independent
closed-loop checkpoint evaluation and the fail-closed package gate determine
whether this run yields a usable continuation policy.
