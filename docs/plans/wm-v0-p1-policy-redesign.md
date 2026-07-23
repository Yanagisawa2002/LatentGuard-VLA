# WM-v0 P1 evidence-grounded policy redesign

**Status:** `PAUSED_ARCHIVED`. The design is complete but is not an active
roadmap. It was paused before implementation on 2026-07-23 while a new
user-directed technical route is prepared. No training, ManiSkill rollout, data
collection, checkpoint generation, final-seed access, SSH, or remote execution
was performed. See
[`wm-v0-p1-policy-redesign-archive.md`](wm-v0-p1-policy-redesign-archive.md)
for the authoritative pause and handoff record.

**Date:** 2026-07-23

**Baseline branch:** `codex/wm-v0-p02-act-grasp`

**Baseline commit:** `91b46ea9cecb759e48d6d568be2d8bc957ec7809`

**Planning branch:** `codex/wm-v0-p1-policy-redesign`

## 1. Decision summary

P0.2 did not fail because it could not emit legal actions, reach the cube, or
issue a close command. It failed after those coarse behaviors emerged. The
strongest committed evidence is:

- 28/30 episodes entered pre-grasp;
- 20/30 issued a valid close near the cube;
- 12/30 reached contact without a verified grasp;
- 1/30 produced a verified grasp, which dropped before lift;
- 0/30 lifted and 0/30 completed the task;
- all 30 episodes timed out with zero action-contract, simulator, workspace, or
  release-integrity failures.

The most defensible root-cause statement is therefore:

> The current single-observation, 16-step ACT imitation formulation learned
> coarse approach and close timing after phase/gripper reweighting, but did not
> learn repeatable contact-conditioned alignment, stable grasp retention, or the
> grasp-to-lift transition.

This is a failure-localization conclusion, not yet a causal proof that any one
architectural feature is responsible. The archive has no independent contact or
lift labels, so contact observability, short-term history, action representation,
and conditional action multimodality remain hypotheses to test.

The preferred next direction is a **compact object-centric,
contact-aware hierarchical reactive policy**. The low-cost backup is a
**history-conditioned one-step recurrent behavior-cloning policy**. A compact
short-horizon diffusion policy is third and is authorized for training only if a
pre-training ambiguity audit establishes conditional action multimodality.

No direction is authorized to train by this plan alone. Each later execution
phase requires a separately committed implementation/data plan, a pushed exact
revision, and explicit authorization to start the currently powered-off remote
server.

## 2. Scope, claims, and anti-claims

### 2.1 Primary claims

| ID | Claim | Present status | Minimum convincing future evidence |
| --- | --- | --- | --- |
| C1 | P0.2's remaining bottleneck is contact-to-grasp stability and grasp-to-lift retention, not action legality, timestamp alignment, or coarse approach. | Supported as failure localization by committed development evidence. | Preserve the P0.2 failure taxonomy and reproduce its integrity facts without changing frozen P0/P0.1/P0.2 artifacts. |
| C2 | Explicitly reactive, contact-aware structure can turn valid approach/close behavior into stable grasp and lift. | Untested hypothesis. | A validation-selected, frozen seed-0 policy must pass the unchanged 30-seed grasp/lift/task gates on a new development manifest; seeds 1 and 2 must then pass the same gates without configuration changes. |

### 2.2 Anti-claims to rule out

- A lower scalar validation loss alone is not evidence of physical progress.
- A new result must not be attributed merely to more parameters, more training
  steps, a larger dataset, a new final-seed search, or a relaxed task/action
  contract.
- Pre-grasp entry, a close command, contact, a single grasp, smooth actions, or
  action legality must not be described as task success.
- Privileged cube pose, goal pose, contact graph, expert phase, success, or
  outcome fields must not become deployable policy inputs.
- P0, P0.1, and P0.2 remain immutable historical results. P1 may compare against
  them but must not rewrite their labels, metrics, reports, or registry entries.

### 2.3 Main experimental story

The must-run story has four blocks:

1. bind and validate the data needed to observe the contact/lift bottleneck;
2. test one compact preferred policy at seed 0 with validation-only checkpoint
   selection;
3. compare the frozen candidate with the frozen P0.2 direct baseline on a new,
   precommitted development manifest;
4. only after the unchanged physical gates pass, repeat the frozen configuration
   at seeds 1 and 2 and consider sealed-final authorization.

Direction 2 is a predeclared fallback, not a hyperparameter continuation of
Direction 1. Direction 3 is conditional and should be cut without training if
its ambiguity audit fails.

## 3. Bound evidence reviewed

### 3.1 Evidence identities

- Demonstration dataset:
  `sha256:234b82709359f1f2fde513ab26c015f4ebfb262a16b1f16206c54eddbde44800`
- Normalization:
  `sha256:8093f5bba3916dff218e73789739b18e05e9eb96cfc200fa18747f010959532a`
- P0/P0.1/P0.2 observation:
  one `224x224` `front_oblique` RGB frame plus the current 18D named Panda
  position/velocity state
- Native action:
  eight-dimensional absolute `pd_joint_pos` at 20 Hz
- Episode/data split:
  500 successful episodes, 21,053 frames, split by source scene seed into
  400 train / 50 validation / 50 offline test episodes
- Historical development seeds:
  `800000..800029`
- Still-sealed final seeds:
  `900000..900099`

The evidence review used the committed training, audit, screening, rollout, and
result artifacts under:

- `artifacts/pickcube_act/`;
- `artifacts/pickcube_act_bounded/`;
- `artifacts/pickcube_act_p02/`;
- the corresponding `docs/pickcube_act*.md` reports.

### 3.2 P0: unbounded native ACT

| Evidence | Observation | Interpretation |
| --- | --- | --- |
| Dataset audit | 500 successful episodes; zero missing/non-finite/duplicate/leakage events; every target legal. | The demonstration archive did not cause P0's action-bound failure. |
| Training | Standard ACT, seed 0, 20,000 steps, best scalar validation loss at step 20,000. | Training completed, but reconstruction loss did not encode action legality. |
| Checkpoint evaluation | 16 checkpoints × 30 development seeds = 480/480 pre-execution action rejections. | No P0 checkpoint was physically evaluable. |
| Failure localization | All audited violations were gripper-dimension overshoots from an unbounded decoder plus inverse normalization. | Intrinsic bounded output was required; clipping or tolerance relaxation was not. |
| Final/package | 0 final seeds, no package, zero accepted policies. | Frozen Result B remains valid. |

### 3.3 P0.1: bounded ACT

| Evidence | Observation | Interpretation |
| --- | --- | --- |
| Controlled change | Same data, inputs, ACT, optimizer, seed, 20,000 steps, chunk 16; only `affine_tanh_v1` changed output semantics. | P0's action legality confounder was isolated. |
| Target audit | Zero target violations; gripper targets were approximately balanced between `+1` open and `-1` closed. | Dataset sign, scale, and class frequency were not inverted. |
| Static screen | All 15 checkpoints legal on 2,000 anchors; only steps 1k–4k survived the full entry screen; later checkpoints collapsed globally near closed. | Bounded output fixed legality but ordinary L1 permitted an always-closed solution. |
| Development rollouts | Four early checkpoints scored 0/10; selected step 1k scored 0/30, zero grasps, all timeouts. | Legal actions were insufficient for closed-loop competence. |
| Final/package | 0 final seeds, no package, zero accepted policies. | Historical Result B remains valid. |

### 3.4 P0.2: phase/gripper-aware ACT

| Evidence | Observation | Interpretation |
| --- | --- | --- |
| Temporal audit | The collector binds `observation[t] -> action[t]`; offset 0 alone preserved all phase boundaries. | No one-step label-shift repair was justified. |
| Gripper audit | `+1` open, `-1` closed; normalization preserved endpoints; the target is commanded, not measured; finger qpos/qvel are already in the 18D state. | Sign/scale confusion and total absence of gripper proprioception are ruled out. |
| Phase audit | 63.0456% of 16-step chunks crossed a phase transition; the archive has expert phases but no independent contact/lift labels. | Narrow transitions were diluted, but true contact/lift supervision was unavailable. |
| Candidate B | Separated arm, gripper, and close-transition losses; capped phase weights; retained the same ACT family and bounded native action. | P0.1's always-closed collapse was directly addressed without privileged policy input. |
| Checkpoint selection | Scalar-loss minimum was step 8k; phase-aware offline top three were steps 6k, 7k, 18k; step 18k was selected because it alone grasped in the frozen 10-seed screen. | Scalar validation loss was not predictive of the decisive physical event. |
| Horizon audit | H=1: 0 grasps; H=2: 1 grasp; H=4: 0 grasps; all had 0 lifts/successes. | H=4 was not uniquely responsible; H=2 was the least-bad fixed choice. |
| Fixed 30 | 28 pre-grasp, 20 valid closes, 12 contact-without-grasp, 1 grasp then drop, 0 lifts, 0 successes. | The bottleneck moved past approach/close and remained at contact, retention, and lift. |
| Integrity/final | Zero action/simulator/workspace failures; 0 final seeds; no package; zero accepted policies. | Frozen Result C and blocked D2 remain valid. |

## 4. Excluded, confirmed, and unverified explanations

### 4.1 Excluded by committed evidence

The following explanations must not be reopened without new contradictory
evidence:

- illegal dataset actions;
- P0's unbounded output head as the remaining P0.2 defect;
- runtime clipping, action repair, or a widened action bound;
- gripper sign or scale inversion;
- `observation[t]` / `action[t]` timestamp misalignment;
- a missing one-step offset in the dataset;
- global always-closed checkpoint collapse in P0.2;
- execution horizon 4 as the sole cause;
- inability to enter pre-grasp as the dominant P0.2 failure;
- action-contract, simulator, workspace, or release failure as the source of the
  0/30 success result.

### 4.2 Confirmed failure facts

- Contact does not reliably become a verified grasp: 12/30 episodes are
  `contact_without_grasp`.
- Close timing is still imperfect: 8/30 are `close_command_too_early`, and
  20/30 misses the frozen 21/30 close gate by one episode.
- A grasp does not reliably persist into lift: the sole grasp dropped before
  lift.
- No learned evidence exists for lift or task completion: both are 0/30.
- The dataset contains expert phase proxies but not independent first-contact or
  first-lift labels.
- One current observation conditions a 16-action prediction, and two actions
  are executed before replanning in the selected P0.2 runtime.
- Average offline loss and the physical checkpoint ranking disagree.

### 4.3 Unverified causal hypotheses

| Hypothesis | Why plausible | Why not yet established | Decisive test |
| --- | --- | --- | --- |
| H1: contact-relevant geometry is not represented precisely enough by one camera/current state. | Contact without grasp dominates, and the policy never receives cube/goal/contact fields. | No object-pose or occlusion error was measured. | Train-only object/fingertip geometry targets plus an object-centric bottleneck; measure held-out geometry error and physical grasp. |
| H2: short observation history and explicit contact/hold state are needed. | Contact is discontinuous and P0.2 executes two open-loop actions; one grasp is not retained. | H=1 also failed, so query frequency alone is insufficient. | Compare the preferred history-aware policy to its otherwise identical no-history ablation. |
| H3: the 16-step ACT/CVAE objective averages or aliases different actions around transitions. | 63.05% of chunks cross phase transitions and scalar loss selected the wrong physical checkpoint. | Conditional action multimodality has not been measured. | Run a train/validation-only nearest-neighbor ambiguity audit before any diffusion training. |
| H4: absolute joint targets are a poor representation for millimeter-scale contact correction. | Coarse approach works while stable contact fails. | The expert succeeds with the same controller, and no Cartesian representation comparison exists. | Future deletion study comparing one-step native joint targets with a versioned Cartesian-to-native adapter while holding observations/data fixed. |
| H5: the demonstrations lack sufficient recovery and near-miss support. | All training trajectories are successful expert executions; learned rollouts visit off-expert contact states. | The archive's local state/action support around failure states has not been quantified. | Measure candidate-state distance to train support; authorize recovery data only if the precommitted coverage test fails. |

## 5. Why P0.2 approaches and closes but does not grasp or lift

Candidate B gave narrow transition frames more weight and separated the gripper
loss from the seven arm dimensions. That directly explains the recovered
approach and close behavior: the global closed-endpoint collapse disappeared,
both gripper endpoints were predicted, 28/30 trajectories reached pre-grasp, and
20/30 issued a near-object close. H=2 also shortened the open-loop interval
relative to P0.1.

Those repairs do not teach the policy what happens at physical contact. The
training archive supplies phase names such as `GRIPPER_CLOSING` and
`TRANSPORT_OR_COMPLETION`, but it does not supply first-contact, stable-grasp, or
first-lift labels. The phase proxy therefore cannot distinguish a well-centered
bilateral grasp from a close that merely touches the cube, nor can it distinguish
a retained grasp from an immediate drop. The 12 contact-without-grasp outcomes
and the one grasp-before-drop are exactly the physical distinctions missing from
the offline supervision.

The policy also makes a 16-step prediction from one current RGB/state sample.
After the first contact, feedback is not represented in the already-predicted
remainder until the next query. That may worsen a narrow contact transition, but
H=1's zero-grasp result means shorter execution alone is not a demonstrated fix.
Likewise, a single front-oblique view may be insufficient for contact-scale
geometry, but no committed pose-error audit proves that yet.

Accordingly, P1 should not repeat ACT with more steps or a different learning
rate. It should expose and test the missing contact/retention mechanism.

## 6. Candidate directions and ranking

| Rank | Direction | Data status | Seed-0 minimum remote GPU estimate | Decision |
| ---: | --- | --- | ---: | --- |
| 1 | Object-centric contact-aware hierarchical reactive policy | Existing demonstrations are useful, but accepted contact/lift/geometry labels are absent; conservative default is a new content-bound training collection. | 3.0–5.5 GPU-hours including data gate and development rollouts | Preferred |
| 2 | History-conditioned one-step recurrent behavior cloning | Existing 500 episodes and phase metadata are sufficient. | 1.25–2.5 GPU-hours including development rollouts | Low-cost backup |
| 3 | Compact short-horizon diffusion policy | Existing data are sufficient only if an ambiguity audit justifies a multimodal generator. | 2.5–4.5 GPU-hours after the audit | Conditional/deferred |

Estimates assume the previously evidenced RTX 5090 class runtime. P0/P0.1
20,000-step ACT runs took about 33–35 minutes, but P1 models, relabeling, and
simulator throughput are not yet benchmarked; the ranges include setup and
validation margin rather than claiming exact runtime.

### 6.1 Direction 1: object-centric contact-aware hierarchical reactive policy

#### Core hypothesis

A compact policy that infers object-relative geometry and an internal
approach/align/close-hold/lift mode from a short allowlisted observation history,
then emits one bounded native action at a time, will correct contact errors and
retain the grasp better than a monolithic 16-step ACT chunk.

#### Change relative to P0.2

- replace ACT/CVAE chunk prediction with a deterministic, compact hierarchical
  reactive controller;
- use a short fixed history (initial proposal: four RGB/state observations) and
  execute one action per query;
- use train-only auxiliary targets for cube/fingertip geometry, contact,
  stable-grasp, and lift onset;
- infer deployment mode from allowlisted observations/history; never feed
  expert phase, contact graph, cube pose, goal pose, success, or outcome as a
  policy input;
- retain fail-closed bounded eight-dimensional native `pd_joint_pos` output;
- keep the camera, controller, 20 Hz control frequency, native 50-step horizon,
  task, and success definition unchanged.

The initial implementation must remain compact. A shared encoder, a small
history module, one internal mode head, and mode-conditioned action heads are
enough. A larger backbone, language model, VLM, or pretrained foundation model
is not authorized by this plan.

#### Data availability and collection

Already available:

- the immutable 500 successful RGB/state/action episodes;
- exact source scene seeds and split groups;
- expert phase proxies and close transitions;
- legal native action and controller identities.

Not present in the committed dataset:

- complete simulator state archives proving frame-identical replay;
- cube/goal/fingertip geometry targets;
- contact-side/force evidence;
- first stable grasp and first lift labels.

The first future data action is a read-only inventory of any retained remote
state archives. A seeded rerun must not be joined to old RGB frames merely
because the scene seed matches.

- If complete content-bound source states exist, replay the exact source actions
  under the accepted runtime, verify full state and action identity at every
  required boundary, and create a derived annotation dataset in the original
  source split.
- If exact source-state identity cannot be proven, do not attach new labels to
  the old frames. Collect a new P1 training dataset in which RGB/state/action,
  geometry, contact, grasp, and lift annotations are captured together. Split
  source episodes before deriving samples and keep every derivative with its
  source group.

The conservative budget assumes the second path. Final seeds and all prior
development seeds are prohibited as collection seeds.

#### Minimal falsifiable experiment

1. Freeze an accepted P1 dataset manifest and annotation semantic.
2. Train one seed-0 compact policy on the full accepted training split with
   ordinary validation-only early stopping.
3. Select exactly one checkpoint from validation metrics before any P1
   development outcome is opened.
4. Freeze the policy, checkpoint, runtime, and the new development manifest.
5. Run the frozen P0.2 step-18k/H=2 baseline and the P1 policy on the identical
   new 30-seed development schedule.
6. Stop Direction 1 if it fails any hard integrity gate or fails to reach
   24/30 pre-grasp, 21/30 valid close, 18/30 grasp, 15/30 lift, and 23/30 task
   success. Partial stage gains are reported but do not promote the policy.

The technically essential ablation removes observation history while retaining
the same data, supervision, encoder, parameter count envelope, and one-step
execution. It runs only after the seed-0 preferred configuration passes its
offline gate and is one of the at-most-three configurations allowed to advance
under the cost-aware screening rule.

#### Cost, risk, and stop condition

- Seed-0 data replay/collection and data gate: 1.0–1.5 GPU-hours.
- Seed-0 train/screen/checkpoint-resume: 1.5–3.0 GPU-hours.
- Paired P0.2/P1 30-seed development evaluation: 0.5–1.0 GPU-hours.
- Main risks: privileged-label leakage, replay labels that are not state-bound,
  mode collapse, and changing too many components at once.
- Stop before training if label provenance or source-state identity fails.
- Stop after seed 0 if the fixed physical gates fail; do not tune on the same
  30 outcomes.

### 6.2 Direction 2: history-conditioned one-step recurrent behavior cloning

#### Core hypothesis

The decisive defect may be ACT's long chunk/CVAE transition aliasing rather
than missing privileged training targets. A small recurrent policy over the
last four allowlisted RGB/state observations, predicting one bounded native
action per query with a separate continuous gripper head, may preserve feedback
through close and grasp using the existing data alone.

#### Change relative to P0.2

- no ACT, CVAE latent, or 16-action chunk;
- deterministic one-step action prediction;
- fixed short observation history;
- separate arm and continuous-gripper heads with the existing bounded action
  contract;
- expert phase may control train-only sampling/loss weights, but is not an
  input or an inferred ground-truth deployment state.

#### Data availability and collection

The existing 400/50/50 episode split contains aligned sequences, action targets,
and phase metadata, so no new collection is required. It still lacks true
contact/lift labels; this limitation is the point of the low-cost test.

#### Minimal falsifiable experiment

- Train one seed-0 model on all 400 train episodes.
- Select one checkpoint using only the 50 validation episodes.
- Require zero action violations, no constant output, both gripper modes, and
  non-inferior phase-conditioned arm/gripper/close metrics to P0.2 before any
  rollout.
- Use a direction-specific untouched 30-seed development manifest.
- Apply the same 24/21/18/15/23 physical gate. If grasp or lift remains below
  the gate, conclude that removing ACT chunking/history aliasing is insufficient
  without contact-aware data; do not increase steps or model size.

#### Cost, risk, and stop condition

- Seed-0 training and static gates: 0.75–1.5 GPU-hours.
- Frozen baseline/candidate development pair: 0.5–1.0 GPU-hours.
- Main risk: it may reproduce behavior cloning's off-distribution contact
  failure because only successful expert states are present.
- Stop if validation transition metrics regress, output collapses, or the
  physical gate fails. This direction is the low-cost backup, not a bridge to a
  recurrent architecture sweep.

### 6.3 Direction 3: compact short-horizon diffusion policy

#### Core hypothesis

If visually/proprioceptively similar transition states genuinely have multiple
valid future action modes, a compact diffusion action generator may preserve
those modes better than ACT's reconstruction objective.

#### Change relative to P0.2

- replace ACT/CVAE reconstruction with a compact one-dimensional conditional
  diffusion head;
- use a fixed short horizon (proposal: eight actions, execute one or two);
- retain the same allowlisted inputs, bounded native action map, task, and
  controller;
- cap the encoder and total parameter envelope before training; no large
  pretrained model is introduced.

#### Data availability and collection

The existing data can support this direction; no new collection is required.
However, training is authorized only after a split-safe nearest-neighbor
ambiguity audit shows that close/contact-near observations have multiple
well-separated legal action continuations beyond sensor noise.

#### Minimal falsifiable experiment

1. Run the ambiguity audit on train/validation only. Use the frozen P0.2 image
   encoder plus normalized 18D state, exclude same-source episodes from
   neighbors, and retrieve the 32 nearest train frames for every validation
   close-transition anchor. In native-range-normalized coordinates, a local
   neighborhood is multimodal only when a deterministic two-cluster split has
   at least eight members per cluster, centroid L2 distance at least `0.10`, and
   silhouette score at least `0.25`. The audit passes only if at least 20% of
   validation transition anchors meet all three conditions and the conclusion
   is unchanged in five fixed source-episode bootstrap replicates.
2. Cut Direction 3 with zero GPU training if the multimodality criterion fails.
3. If it passes, train one seed-0 compact diffusion candidate and one
   deterministic matched-capacity deletion baseline.
4. Select checkpoints on validation only and evaluate the predeclared winner on
   a direction-specific untouched 30-seed manifest.
5. Apply the unchanged 24/21/18/15/23 physical gate.

#### Cost, risk, and stop condition

- Ambiguity audit: CPU or at most 0.25 GPU-hours if a frozen image embedding is
  required.
- Seed-0 training/static gates: 2.0–3.5 GPU-hours.
- Frozen development pair: 0.5–1.0 GPU-hours.
- Main risks: no demonstrated multimodality, sampling latency, and adding
  stochasticity without contact feedback.
- Stop before training if the ambiguity audit fails; stop after seed 0 if the
  fixed physical gate fails. Do not rescue the result by adding denoising steps,
  a larger backbone, or more seeds.

## 7. Data and leakage protocol

### 7.1 Immutable historical data

- The existing dataset digest, episode split, P0/P0.1/P0.2 artifacts, and
  historical development outcomes are read-only.
- Existing test episodes remain unavailable to training, preprocessing fitting,
  threshold fitting, checkpoint selection, and architecture selection.
- Any P1 derivative records source episode, scene seed, source policy/task,
  transformation/annotation semantic, parameters, split group, schema version,
  and content digests.

### 7.2 P1 data acceptance gate

Training is prohibited until all of the following pass:

1. exact episode/source inventory and digest validation;
2. trajectory-level 400/50/50 or newly declared split before derivation;
3. no source or derivative leakage across splits;
4. complete finite shape/dtype validation;
5. exact action/observation/frame alignment;
6. explicit train-only versus reporting-only field allowlists;
7. no missing contact/lift value silently replaced with false;
8. exact replay identity if annotations are joined to historical frames;
9. accepted runtime/task/controller/source identities;
10. reload from the final serialized dataset and reproduce all digests.

Unavailable or indeterminate labels remain unavailable/indeterminate. They do
not become negative examples.

### 7.3 Deployable input allowlist

The default P1 allowlist is:

- `observation.images.front_oblique`;
- named 18D Panda qpos/qvel state;
- a fixed, explicitly serialized history of those two fields;
- deterministic policy-internal recurrent/mode state if its save/resume
  contract is complete and content-bound.

Cube pose, goal pose, expert phase, contact graph/force, candidate type,
provenance, success, and outcome are train-only targets or reporting metadata,
never policy inputs. Any proposed expansion of this allowlist requires a
separate reviewed contract before collection or training.

## 8. Checkpoint selection and execution order

### 8.1 Run order

| Stage | Goal | Required result | Remote execution |
| --- | --- | --- | --- |
| P1-0 | Local contracts, schemas, config validation, CPU loader/tests | All local validation passes; no result files changed | No |
| P1-1 | Data inventory and acceptance | Accepted content-bound manifest or explicit data blocker | Later authorization required |
| P1-2 | Config/GPU mechanics gates | Dry-run, one-batch forward, forward/backward, tiny overfit, save/load/resume, peak memory | Later authorization required |
| P1-3 | Seed-0 full-data screen | Exactly one validation-selected checkpoint, zero integrity failures | Later authorization required |
| P1-4 | Frozen paired development | P0.2 baseline and P1 candidate on one new 30-seed manifest | Later authorization required |
| P1-5 | Stability | Frozen seeds 1 and 2, same config and checkpoint rule | Only if P1-4 passes |
| P1-6 | Sealed final | One predeclared deployable seed/checkpoint on `900000..900099` | Separate explicit authorization only |

### 8.2 Checkpoint selection

- Seed 0 screens each authorized architecture once on the full training data.
- Ordinary early stopping and checkpoint ranking use validation data only.
- Preprocessing, class weights, auxiliary-loss weights, calibration, and
  thresholds are fit without test, development, or final outcomes.
- Checkpoints must first pass finite/shape/digest, native action, non-collapse,
  deterministic query, save/load, and exact resume gates.
- The validation ranking is lexicographic and must be frozen before training:
  1. hard integrity;
  2. no gripper or internal-mode collapse;
  3. contact/hold/lift auxiliary metrics when accepted labels exist;
  4. close-transition F1 and timing;
  5. phase-conditioned arm/gripper errors;
  6. scalar aggregate loss only as the final tie-breaker.
- No development rollout selects a checkpoint. P0.2's behavioral checkpoint
  selection remains historical evidence but is not reused as P1 protocol.
- At most the best validation model, the frozen P0.2 direct baseline, and one
  technically essential ablation advance to seeds 0/1/2.
- Seeds 3 and 4 are prohibited unless a later plan documents one of the
  repository's explicit variance/ranking/collapse triggers.

### 8.3 New development manifests

The previous `800000..800029` schedule has been observed repeatedly and is not
an untouched P1 selection set. Before any new outcome:

- preferred Direction 1 reserves a new disjoint 30-seed manifest, proposed
  range `820000..820029`;
- backup Direction 2 reserves `821000..821029`;
- conditional Direction 3 reserves `822000..822029`;
- exact ordered lists, configs, code revision, candidate/baseline identities,
  and hashes are committed before a simulator is started;
- a collision audit must verify disjointness from expert, collection,
  historical development, smoke, and final schedules.

These numeric ranges are proposed identifiers, not permission to execute. If a
range is found to collide, a replacement must be committed before any outcome
exists.

Mechanics-only smoke seeds must be separate and may validate reset, query,
action, instrumentation, and resume. Observed smoke quality cannot select a
model or threshold.

If a frozen 30-seed result causes a policy/config change, the changed revision
must use a new untouched development manifest. The old outcomes remain negative
or partial evidence and are never overwritten.

## 9. Physical gates, promotion, and stop rules

### 9.1 Hard integrity gates

Every evaluated episode requires:

- finite, exactly shaped actions;
- zero native/canonical action violations and no clipping/projection/fallback;
- exact policy/checkpoint/processor/config identity;
- zero simulator and workspace integrity failures;
- complete runtime and task evidence;
- deterministic fixed-seed reproduction;
- conclusive task metrics kept separate from execution errors.

Any hard-integrity failure blocks promotion. It is invalid context or execution
error, not task failure.

### 9.2 Unchanged component and promotion gates

For each seed-0 30-episode development run:

| Gate | Requirement |
| --- | ---: |
| Enter pre-grasp | at least 24/30 |
| Valid close near cube | at least 21/30 |
| Verified grasp | at least 18/30 |
| Lift | at least 15/30 |
| Native task success | at least 23/30 (at least 75%) |
| Action/simulator/workspace integrity | zero failures |

All gates must pass. A component-stage improvement that fails a later gate is
reported as partial evidence and does not promote the policy.

If seed 0 passes, seeds 1 and 2 train under the identical frozen configuration
and validation-only checkpoint rule. Each of the three seeds must independently
pass the full table. A collapsed seed, ranking change, or high variance is
reported; it is not repaired by selecting a lucky seed from development
outcomes.

The predeclared deployable candidate for sealed-final evaluation is seed 0.
Seeds 1 and 2 are stability evidence and cannot replace seed 0 based on
development success.

### 9.3 Stop rules

Stop the direction and preserve its evidence when any of the following occurs:

- the final dataset reload/integrity/leakage gate fails;
- a privileged or outcome-derived field reaches the deployable input path;
- training produces non-finite gradients, actions, or incomplete checkpoints;
- the bounded action or resume contract fails;
- the seed-0 validation/offline screen fails;
- any seed-0 physical gate fails;
- the preferred/backup hypothesis is falsified on its frozen development
  manifest;
- the exact remote revision cannot be synchronized cleanly;
- source code would need a remote-only fix;
- final seeds would be needed to choose a model, checkpoint, threshold, or
  architecture.

No failed gate is repaired by more steps, a larger model, relaxed thresholds,
or reuse of the same development outcomes.

## 10. Failure taxonomy

The evaluator records the earliest decisive physical failure while also
retaining all observed progress events:

1. `never_enters_pregrasp`;
2. `pregrasp_misaligned`;
3. `close_command_too_early`;
4. `close_command_too_late`;
5. `contact_without_grasp`;
6. `grasp_unstable_drop_before_lift`;
7. `grasp_without_lift`;
8. `lift_then_drop`;
9. `lift_without_goal_approach`;
10. `goal_approach_without_valid_release`;
11. `release_without_placement`;
12. `placed_but_robot_not_static`;
13. `timeout_after_progress`;
14. `task_success`.

Where the adapter can prove contact side/force without changing policy inputs,
the evidence may additionally distinguish unilateral and bilateral contact.
Missing contact evidence must not be inferred from cube motion.

The following remain separate non-task categories:

- dataset or evidence invalidity;
- restoration/source/action identity mismatch;
- action-contract violation;
- simulator error;
- workspace integrity failure;
- policy query/runtime error;
- incomplete or indeterminate task evidence.

Resume is idempotent. A committed episode is never rerun, and an incomplete
episode never receives a fabricated failure label.

## 11. Metrics and claim language

### 11.1 Metrics that can support component progress

- pre-grasp entry count/rate;
- valid-close count/rate and timing;
- contact count, side, and force when directly evidenced;
- verified grasp count/rate;
- stable-grasp duration;
- post-grasp drop count/rate;
- lift count/rate and lift-onset timing;
- placement and robot-static subconditions;
- stage-conditioned Wilson intervals;
- paired candidate-minus-P0.2 development differences with a fixed
  2,000-resample source-seed bootstrap;
- action integrity, determinism, resume, and runtime reliability.

Offline auxiliary/event metrics can support representation or checkpoint
readiness, but they are not physical results.

An "improved over P0.2" component claim additionally requires the paired 95%
bootstrap interval for the relevant count/rate difference to exclude zero.
Absolute promotion still requires every fixed gate; a favorable paired interval
cannot replace a failed grasp, lift, or task-success gate.

### 11.2 Metrics that cannot be called task success

- lower training or validation loss;
- arm/gripper MAE, event F1, timing error, or image/pose auxiliary accuracy;
- legal actions or zero simulator errors;
- checkpoint diversity;
- pre-grasp entry;
- a close command;
- contact;
- grasp without lift;
- lift without placement;
- placement without the task's robot-static success condition;
- smoothness, latency, query count, or GPU memory;
- aggregate progress scores that merge different physical stages.

Only the unchanged ManiSkill `PickCube-v1` terminal condition
`is_obj_placed and is_robot_static` within the native 50-step horizon is task
success.

## 12. Sealed-final protocol

Final seeds `900000..900099` remain sealed until all of the following are true:

1. the data, policy, checkpoint rule, seed-0 checkpoint, processors, thresholds,
   evaluator, failure taxonomy, and final config are committed and pushed;
2. local `HEAD` is clean and exactly matches upstream;
3. the remote tracked checkout is clean and exactly matches the pushed full SHA;
4. data and development manifests reproduce their digests;
5. seeds 0, 1, and 2 each pass every development gate;
6. seed 0 passes package identity, deterministic query, action-contract,
   save/load, and resume validation;
7. the final schedule/access decision is content-bound before final outcomes
   are loaded;
8. a separate task explicitly authorizes server start and final evaluation.

The final run evaluates the predeclared seed-0 package once on exactly 100
seeds. No final outcome may change the policy, checkpoint, preprocessing,
threshold, execution horizon, failure classifier, or seed schedule. Poor final
performance does not trigger extra seeds, model selection, or test tuning.

Final results are reported with exact counts, Wilson intervals, failure
taxonomy, integrity failures, and the distinction between partial stage
progress and task success. A negative final result remains public evidence.

## 13. Compute budget and ETA

The current design milestone consumes **0 remote GPU-hours**.

Future estimates begin only after explicit execution authorization:

| Phase | Work | GPU-hours | Wall-clock ETA after authorization |
| --- | --- | ---: | --- |
| Local P1 implementation | contracts, loaders, tests, configs, manifests | 0 | 1–2 working days |
| Preferred data gate | state-archive audit and exact relabel or new annotated collection | 1.0–1.5 | 0.5–1 day |
| Preferred seed-0 minimum | mechanics gates, full train, validation selection | 1.5–3.0 | 0.5–1 day |
| Preferred paired development | frozen P0.2 and P1 runs on 30 new seeds | 0.5–1.0 | 0.5 day |
| Seeds 1 and 2 | identical frozen training and development gates | 3.0–5.0 | 1–2 days |
| Sealed final, if separately authorized | one seed-0 package, 100 sealed seeds | 1.0–1.5 | 0.5–1 day |

Preferred-direction estimate:

- through seed-0 falsification: **3.0–5.5 GPU-hours**, about **2–4 working
  days** including local implementation;
- through three-seed development promotion: **6.0–10.5 GPU-hours**, about
  **4–7 working days**;
- including a separately authorized sealed final: **7.0–12.0 GPU-hours**,
  about **5–8 working days**.

If Direction 1 stops, Direction 2 adds approximately **1.25–2.5 GPU-hours** for
seed 0 and **3–5 GPU-hours** through three seeds. Direction 3 is not included in
the main budget and adds approximately **2.5–4.5 GPU-hours** for seed 0 only
after its ambiguity gate.

These are planning ranges, not reservations. They exclude server provisioning
delay and any separately authorized new-data expansion. Earlier gate failure
reduces cost by stopping later phases.

## 14. Compliance review

| Constraint | Plan treatment | Status |
| --- | --- | --- |
| Release freeze | P0/P0.1/P0.2 artifacts, labels, claims, registries, and reports remain unchanged; P1 is a new post-release research phase. | Compliant |
| Negative evidence | Result C, 0/30 success, one grasp then drop, 0 lifts, and blocked D2 are quoted exactly. | Compliant |
| Final-seed seal | `900000..900099` are excluded from data, development, selection, and debugging; opening requires a later explicit authorization. | Compliant |
| Data isolation | Split by source before derivation; no privileged input leakage; no fabricated missing labels; test/development/final outcomes excluded from fitting. | Compliant |
| Remote source of truth | All tracked changes occur locally, are validated, committed, and pushed before any remote execution. | Compliant |
| Current server state | No SSH, server start, training, rollout, or remote simulation occurs in this milestone. | Compliant |
| LangMani boundary | No LangMani worktree/source is entered, copied, modified, or used. | Compliant |
| Cost-aware screening | One seed-0 screen per direction, full training data, validation-only checkpoint selection, at most one baseline and one essential ablation advance. | Compliant |
| Claims | Component progress and task success remain separate; all future claims require committed evidence. | Compliant |

### 14.1 Local validation environment note

The repository's existing `.venv` is Python 3.12.10, matching the documented
LeRobot-era exception, and passes the complete local validation stack. An
additional isolated Python 3.11.9 audit passed `ruff` and `mypy` and reached
1,701 passed / 3 skipped tests, but one existing ACT test failed because
`Enum.__contains__` raises a `DeprecationWarning` on Python 3.11 and repository
warnings are errors. The design-only diff does not touch that source or test.
Before P1 implementation or training, the local P1-0 gate must restore a fully
green Python 3.11 suite with a separately reviewed compatibility fix; it must
not suppress the warning or treat the Python 3.12 pass as evidence that the
Python 3.11 contract is satisfied.

## 15. Current milestone completion criteria

This design-only milestone is complete when:

- this plan and its tracker are locally validated;
- all changes are reviewed and limited to new planning documents;
- the target branch is committed and pushed;
- the working tree is clean and `HEAD` equals upstream;
- the final report records that remote training, rollout, collection, SSH, and
  final-seed access did not occur.

After completion, work pauses for the next explicit authorization.
