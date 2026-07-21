# WM-v0 results

## Current status

WM-v0 now has a physically executed end-to-end smoke path: exact-state
counterfactual collection, frozen ResNet-18 feature construction, GPU
forward/backward, checkpoint resume, and fixed held-out evaluation. The run is
strictly `non_formal_smoke`; it does not alter `portfolio-v1` or any sealed
PickCube result.

The live dataset authority is `artifacts/wm_v0/dataset_manifest.json`. It
contains six real simulator-future samples from three source trajectories and
three exact anchors. Each source group is assigned to exactly one of train,
validation, and test, with two candidates per split. All six candidates are
synthetic corruptions: three additive-noise and three constant-bias chunks. No
policy-generated candidate was present, so the smoke cannot make a learned
policy-proposal claim.

The formal gate correctly remains closed. Six samples are below the required
1,000, and all six terminal outcomes are successful, so the dataset lacks the
required failure class. Real future observations, required splits, source-group
isolation, and nonterminal progress/event supervision did pass.

## Remote execution

Run `20260721T053204Z_wm-v0-smoke_990d8ce_seed271828` collected on the exact
pushed revision `990d8ce1dd9c303e3295bad41320ca1a29c72607`. Its first collection attempt
failed before serialization because the runtime job used an archive episode ID
where the WM schema requires the immutable source-group identity. The invalid
job file and failure log were retained; attempt two changed only that identity
field, not anchors, candidates, or action bytes, and produced all six samples.

The first feature-build attempt then exposed a CLI bug: it tried to construct a
reviewed M4B backbone contract without its required fields. The source was fixed
locally, covered by a regression test, committed and pushed as
`bafe920b777778519b2df2b2c0e10139e059b4e1`, and the clean remote checkout was
fast-forwarded to that exact revision. Feature construction and all later work
ran on the fixed SHA. No tracked source was edited remotely.

The frozen official ResNet-18 produced six authenticated feature records. On a
single RTX 5090, WM-v0 trained for one step and resumed from checkpoint step 1
to step 2. The outcome-only and accepted direct-verifier baselines each ran one
bounded forward/backward step. Peak allocated GPU memory was about 48.4 MB,
43.6 MB, and 23.4 MB respectively. These step counts validate mechanics, not
convergence or fair model selection.

## Non-formal metrics

The test split contains only two successful synthetic candidates in one anchor
group. Consequently terminal AUROC and pairwise ranking accuracy are undefined,
and top-1 success is trivially 1.0 for every scorer. The following values are
recorded for reproducibility only and must not be interpreted as a comparison:

| Model | Steps | Validation loss | Terminal Brier | Terminal ECE |
| --- | ---: | ---: | ---: | ---: |
| Direct verifier | 1 | 0.5750 | 0.1745 | 0.4177 |
| Matched outcome-only | 1 | 0.6869 | 0.0641 | 0.2532 |
| WM-v0 | 2, including resume | 1.2837 | 0.0116 | 0.1076 |

For WM-v0, mean future-latent cosine similarity was 0.1337, mean latent MSE
was 1.6696, progress MAE was 0.5949, and task-success event AUPRC was 0.8167
over eight observed horizon labels. The support is far too small for a research
claim. The central question, "does future-latent prediction improve action
selection?", therefore remains **unanswered**.

The offline gate did not pass, so no closed-loop probe executed. This preserves
the conservative default rather than using a six-sample smoke to authorize an
intervention.

## Reproduction commands

The real integration requires private runtime paths supplied only in the
resolved run configuration. The reviewed public entry points are:

```bash
python scripts/collect_wm_v0.py \
  --config <resolved-collect-config.json> \
  --jobs <frozen-jobs.json> \
  --adapter-factory latentguard.integrations.maniskill_pickcube.world_model:build_pickcube_world_model_adapter \
  --output <raw-output>

python scripts/build_wm_dataset.py \
  --config configs/wm_v0/dataset.yaml \
  --samples <raw-output>/samples \
  --output <feature-output> \
  --backbone-config configs/training/m4b/backbone-resnet18-imagenet1k-v1.json \
  --backbone-manifest <authenticated-backbone-manifest.json> \
  --device cuda

python scripts/train_wm_v0.py \
  --config configs/wm_v0/train.yaml \
  --features <feature-output>/features \
  --dataset-manifest <feature-output>/dataset-manifest.json \
  --model wm_v0 --max-steps 1 --device cuda \
  --allow-insufficient-data-smoke

python scripts/evaluate_wm_v0.py \
  --train-config configs/wm_v0/train.yaml \
  --eval-config configs/wm_v0/eval.yaml \
  --features <feature-output>/features \
  --checkpoint <checkpoint.pt> \
  --model wm_v0 --split test --output <evaluation.json> \
  --device cuda --non-formal-smoke
```

## Validation

- Local complete suite: 1,603 passed, 3 existing Windows symbolic-link skips.
- Local focused WM-v0 suite: 13 passed.
- Local `python -m ruff check .`, `python -m ruff format --check .`, and
  `mypy src`: passed.
- Remote Linux focused WM-v0 suite: 13 passed; focused Ruff checks, format
  checks, and `mypy src` passed.
- Retrieved compact artifact SHA-256 and size inventory: passed locally.
- Raw RGB, states, feature caches, and checkpoints remain outside Git.

## Remaining work

A formal experiment needs at least 1,000 real-future samples, both terminal
classes, more source groups, and equal optimization/model-selection budgets. It
also needs genuine `policy_generated` candidates reported separately from
synthetic corruptions. Only after that offline comparison passes should
uncertainty calibration or selective closed-loop intervention be considered.
