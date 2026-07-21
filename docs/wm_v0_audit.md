# WM-v0 repository and data audit

## Decision

The existing accepted datasets do **not** authorize formal world-model
training. They contain enough candidate-outcome records to train direct
verifiers, but they do not contain candidate-specific post-execution RGB and
proprioceptive sequences. WM-v0 must first collect new real counterfactual
rollouts. Until the formal gate passes, all model results are training smoke
evidence rather than performance claims.

## Existing verifier contract

The accepted structured direct verifier consumes a 38-component PickCube
verifier state, a masked 16-by-8 action chunk, and no images. The selected
temporal model produces one failure logit. M3B also retains state-only,
action-only, and joint-MLP baselines. M4B consumes three current-boundary RGB
views through a frozen official torchvision ResNet-18 (`IMAGENET1K_V1`,
224-by-224 input, 512 output features), combines those features with the same
masked action chunk, and produces one failure logit. Neither accepted model
predicts future state or features.

## Existing data inventory

- M3A has 3,240 samples in 360 anchor groups from 60 source trajectories:
  360 source candidates and 2,880 synthetic corruptions. The compact report
  records 2,289 successful and 591 unsuccessful corrupted replays.
- M3A records current 38-D verifier state, candidate action/mask, terminal
  success, before/after scalar progress, failure events, and strong exact-replay
  evidence. It does not record step-aligned future RGB or future proprioception.
- M4A has 1,080 current-state visual packets (three views each) and 3,240
  candidate bindings. All candidates at an anchor share the same pre-action
  packet. These are not candidate-specific future observations.
- State-indexed PickCube archives contain exact complete states at every source
  boundary and the successful source action sequence. They can supervise the
  source continuation only; they cannot stand in for futures of corrupted or
  policy-generated alternatives.
- Raw runtime datasets, RGB, checkpoints, and caches remain outside Git by
  design. Compact committed reports do not contain enough bytes to reconstruct
  future training examples.

## Replay and collection capability

The accepted PickCube integration already supports the mechanics required by a
real collector:

- arbitrary indexed complete-state restoration in a fresh session;
- exact archive digest/inventory checks and the compatibility-bound
  `tolerance_verified_full_state_v1` runtime comparison at `1e-6`;
- independently reset candidate execution from the same anchor;
- per-action public task snapshots;
- state-preserving ordered three-view RGB rendering with complete post-render
  state, task-projection, and verifier-vector verification; and
- content-bound state, action, visual, and evidence identities.

The generic replay evidence currently stores terminal task evidence and action
counts, not a step-aligned observation trajectory. WM-v0 therefore adds an
observation hook without weakening any existing restore or render gate.

## Label availability

Current public PickCube snapshots provide success, placed/static state, grasp
state, cube height, cube-to-goal distance, and TCP-to-cube distance. These can
support task-specific continuous progress plus observed `grasped`, `dropped`,
object-motion, timeout, and success channels. Wrong-object contact, collision,
and workspace violation are not all observable under the accepted public
contract; they must remain explicitly masked until a reviewed sensor contract
supplies them. An unavailable event must never be silently labelled false.

## Candidate authenticity

Accepted PickCube alternatives are predominantly deterministic synthetic
corruptions of the official solver continuation. The source continuation is a
real solver action, not a learned policy proposal. M6A.1 defines an initial-state
LangMani bridge, but its real probe was blocked by environment compatibility,
checkpoint binding, intermediate restoration, and candidate-identity gaps.
Consequently WM-v0 must label present alternatives `synthetic_corruption` and
must not claim policy-generated generalization. Any later real proposal must be
labelled `policy_generated` and evaluated separately.

## Minimal viable scope

The viable first scope is fixed ManiSkill PickCube, three ordered current and
future RGB views, 38-D non-privileged deployable verifier-state projection,
masked 16-by-8 candidate chunks, configurable short prediction horizon/stride,
frozen ResNet-18 features, and a compact temporal model. The direct M3B verifier,
an outcome-only matched model, and WM-v0 must receive the same source groups and
candidate set. The existing sealed portfolio remains unchanged.

## Gate result at audit time

`formal_training_authorized = false`. The required real-future sample count is
zero in committed evidence. The new collector and strict manifest must be run
on an exact pushed revision. A small real collection can validate mechanics and
training smoke, but only a manifest satisfying every declared gate may support
a formal comparative result.
