# LG-R1 multi-task rollout and SARM progress plan

## Objective

LG-R1 expands the frozen LG-R0 VLA-JEPA policy from a task-0-only smoke into
an outcome-complete, multi-task LIBERO development dataset and tests whether a
stage-aware progress model can describe advancement, stagnation, and
regression. It does not train a LatentGuard failure head, rank action
candidates, calibrate an intervention threshold, or change policy execution.

## Immutable starting point

- local baseline: `04b2a253a4c2ac7d3328df8861dfb9f5f0c32e80`
- LG-R0 remote execution revision:
  `9e9e7276685e0e4052ee307f9e379d1278f904f1`
- VLA-JEPA checkpoint and processor:
  `735d9f692981e286ade093b5046627eda876e5d0`
- LeRobot: `0.6.0` at
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- action horizon 7, relative control, native scale and gripper convention

LG-R0 artifacts are read-only inputs. PickCube evidence and sealed seeds are
out of scope. LangMani, RoboLab, PointWorld, LingBot-VA, ROS 2, candidate
ranking, intervention, policy fine-tuning, world-model training, and synthetic
failure generation are excluded.

## Frozen task and seed design

The registry freezes eight tasks before any new policy outcomes are observed:
two tasks each from `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`. The selection covers ordinary pick-place, articulated access,
toggle activation, two-object placement, and placement followed by closure.

Pilot seeds are `2000..2009`. Extension seeds are `2010..2019`; both groups are
frozen now, rather than selected after pilot outcomes. They do not overlap
LG-R0 development seeds `1000..1009` or sealed seeds `900000..900099`.

## Execution gates

1. Audit the SARM implementation, dependencies, checkpoint availability, and
   exact source identities.
2. Validate configs and all CPU-only contracts locally.
3. Commit and push the implementation before remote execution.
4. On the server, source `/etc/network_turbo`, use Aliyun mirrors by default,
   confirm an idle GPU, and fast-forward to the exact pushed revision.
5. Run one config-only validation, one CPU loader smoke, one GPU forward smoke,
   and one recorded rollout before the 80-episode pilot.
6. Record all pilot episodes. Do not alter action execution or choose seeds
   from observed outcomes.
7. Because LG-R2 requires at least 100 valid rollouts, run the already-frozen
   extension to 160 total after pilot integrity passes. The extension is also
   mandatory when pilot natural failures are below 10.
8. Reload and iterate every dataset; reject non-finite actions, identity
   incompleteness, infrastructure errors, and split leakage.
9. Generate stage labels from privileged sidecars. Privileged fields are
   supervision-only and are explicitly excluded from model inputs.
10. Pass stage QA before any SARM or representation-probe optimizer step.
11. Freeze validation thresholds and model selection before test evaluation.
12. Retrieve only compact manifests, metrics, and reports.

## Stage semantics

Pick-place adapters use approach, alignment, contact, lift, transport, place,
release, and success stages. Toggle tasks use approach, alignment, contact,
activation, and success. Multi-object and articulated tasks use task-specific
subgoal stages while mapping completion to `[0, 1]`.

Labels use BDDL goal predicates, simulator object/fixture poses, contact pairs,
gripper state, and distance changes. Step fraction is retained only as the
explicit time baseline. Failure does not force progress to zero; timeout does
not become a failure stage; labels can regress; terminal outcome is not
backfilled into earlier frames.

## Model comparison

- Time baseline: `t / T_max`, plus offline-only `t / T_episode`.
- Current-representation probe: frozen VLA-JEPA current visual tokens with a
  linear head; a small MLP is allowed only if frozen before training.
- SARM: the official LeRobot SARM module at the frozen LeRobot commit. No
  compatible official LIBERO checkpoint is assumed. If none is found, the run
  is reported as a SARM-style small baseline with frozen CLIP and reduced
  stage/progress transformers, not an official checkpoint reproduction.
- VLA-JEPA predictor comparison: descriptive statistics only.

The SARM/probe may optimize only their declared progress heads. VLA-JEPA,
Qwen, CLIP, policy processors, and action policy remain frozen.

## Promotion and stop conditions

LG-R2 remains unauthorized unless all of the following pass: at least 100
valid episodes, four tasks, two suites, ten natural failures, 100 failure
windows, 100 success windows, stage QA, completed SARM evaluation, zero
episode/seed leakage, and complete checkpoint/processor identity.

SARM training stops before its first optimizer step if stage QA fails. A model
run stops on non-finite loss, frozen-backbone mutation, identity drift, resume
failure, or invalid split use. SARM has no promotion value unless it exceeds
the online-available time baseline on validation and retains the improvement
on the untouched test split.

## Expected remote cost

- rollout pilot: approximately 1.5–3 GPU-hours
- rollout extension: approximately 1.5–3 GPU-hours
- feature extraction and stage dataset build: approximately 1–2 GPU-hours
- representation probe: below 1 GPU-hour
- SARM-style baseline and checkpoint/resume validation: approximately
  2–5 GPU-hours
- final evaluation and descriptive VLA-JEPA comparison: approximately
  1–2 GPU-hours

The total estimate is 8–16 RTX 5090 GPU-hours and roughly one to two elapsed
days, depending on long-horizon failures and video encoding. These are planning
estimates, not measured results.
