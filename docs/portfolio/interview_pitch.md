# LatentGuard-VLA Interview Pitch

## 30-second version

I investigated whether VLA actions could be checked before execution. I
integrated VLA-JEPA with LIBERO, collected 320 multi-task episodes, and tested
progress models, zero-shot video reward, numeric action conditioning, and exact
simulator replay. The honest answer was no: held-out progress correlation fell
from 0.977 to 0.378, the best standalone zero-shot reward reached 0.164 failure
AUPRC, action conditioning improved MAE only 0.98%, and 20 of 30 repeated
RoboLab replays violated the official tolerance. I stopped before claiming a
candidate selector, and turned the negative result into a reproducible
evaluation and failure-analysis portfolio.

## 90-second version

The project started from a practical question: a VLA gives an action chunk, but
can we tell whether an alternative would be better before executing either one?
I decomposed that into gated prerequisites.

First, I pinned LeRobot VLA-JEPA on LIBERO. The policy ran well in a 40-episode
task-0 development smoke, but its predictor did not directly accept external
numeric action candidates or emit native risk scores. Next, I built a
task-specific progress baseline. It was strong in-domain—0.977 Spearman—but
collapsed to 0.378 on held-out tasks after I expanded the dataset to 320
episodes with 27 natural failures. Frozen task-agnostic video reward helped
somewhat, but the best standalone zero-shot AUPRC was only 0.164 and still
required video after execution.

I then tested whether numeric actions added information beyond state. They
improved short-horizon MAE by only 0.98%, and the MAE gain reversed in a
leave-task-out check, which exposed the limits of single-policy on-policy data.
Finally, I tested exact replay. LIBERO restoration failed, so I evaluated one
alternative, RoboLab. I fixed a real configuration overlay bug, but repeated
replay still failed the official tolerance 20 out of 30 times.

The innovation is not a successful shield. It is the disciplined experiment
chain: grouped evaluation, negative controls, cross-task tests, replay-fidelity
gates, frozen artifacts, and a defensible stop before an unsupported online
claim.

## 5-minute technical version

### 1. Research dependency chain

The desired claim required five things in order: a candidate-compatible policy
interface, a transferable progress or failure signal, identifiable incremental
action value, faithful same-state branching, and only then ranking or online
intervention. I preregistered phase gates so that later work could not rescue an
earlier failure by changing the claim.

### 2. Policy and data

I pinned the LeRobot/VLA-JEPA stack and ran a 40-episode LIBERO task-0
development smoke, with 39 successes. That established execution, not
verification: the temporal predictor consumed internal action-token features,
not arbitrary external numeric candidates, and exposed no native risk or
success score.

I then built two evidence scales. LG-R1 had 160 episodes but only one natural
failure. LG-R1b expanded to 320 episodes, 293 successes, 27 natural failures,
8 tasks, 2 suites, and 79,326 frames. Splits and derived examples were grouped
by episode to avoid leakage.

### 3. Progress and reward

The SARM-style task-specific progress model achieved 0.0205 MAE, 0.9774
Spearman, and 0.8654 pairwise accuracy in-domain. On held-out tasks, MAE rose
to 0.2112, Spearman fell to 0.3775, pairwise accuracy to 0.5245, and stage
regression recall was zero. That showed the ontology was learning substantial
task structure.

I compared frozen SARM, ROBOMETER, TOPReward, calibration, and a simple
ensemble. ROBOMETER was the strongest standalone zero-shot task-agnostic model,
but its failure AUPRC was 0.1638, with 0.160 precision at 0.630 recall and
0.304 FPR. More importantly, video reward is post-execution evidence, not a
score for an unexecuted numeric action.

### 4. Numeric action conditioning

To test whether actions added signal beyond state, I used episode-grouped folds
and state-only, action-only, state-plus-action, permutation, action-swap, and
leave-task-out controls. State-plus-action changed short-horizon MAE from
0.040997 to 0.040597, a 0.98% reduction; Spearman improved from 0.430 to
0.470 and terminal-failure AUPRC from 0.149 to 0.203. But in leave-task-6-out,
MAE slightly worsened. Because one behavior policy generated the data, state
and action were confounded. I treated this as local predictive signal, not a
causal action-value result.

### 5. Counterfactual replay

VLA-JEPA could generate four unique action chunks at the same observation, but
LIBERO restoration produced 36 stepped-trace failures, three terminal
mismatches, and maximum error 0.467783. I then evaluated one alternative
platform. In RoboLab I diagnosed a callable-bearing list being replaced by a
JSON string and prepared a schema-aware patch without `eval` or `exec`. After
the fix, all 30 replays completed. Repeat 0 passed 10/10; repeats 1 and 2
failed 20/20 at the official 0.01 tolerance, with 800 strict step failures and
maximum error 0.149008. Terminal outcomes still matched, illustrating why
terminal agreement is not a replay-fidelity guarantee.

### 6. Decision and value

I closed the exact same-state route rather than move to a third simulator or
add models. The deliverable is a source-bound research closeout: exact JSON
pointers and hashes, reproducible figures, result matrices, upstream patch
materials, and clear separation between completed engineering, empirical
findings, and unvalidated goals.

## Direct answers to likely questions

### Did the project ultimately succeed?

It succeeded as a rigorous empirical study and research-engineering project,
but it did not validate the original counterfactual candidate verifier. There
was no candidate ranking, online intervention gain, or real-robot claim.

### Why are there so many Result B/C outcomes?

Because each phase had a fixed promotion gate. Result B retained useful partial
evidence without authorizing the next claim; Result C stopped a route when its
validity prerequisite failed. Preserving those outcomes prevented metric
shopping and retrospective scope changes.

### What did you actually innovate?

The work's defensible novelty is methodological and engineering: a staged
evaluation that connects interface audits, held-out-task progress, frozen
task-agnostic reward, numeric-action negative controls, and stepwise replay
fidelity. I also produced an upstream-quality configuration patch. I do not
claim a new successful verifier algorithm.

### Why did you stop?

Two different replay stacks failed the strict same-state requirement, while
reward and action-conditioning results were too weak or unstable to support a
candidate outcome claim. A third simulator or another reward model would have
moved the goalposts.

### What would you do with more resources?

I would not silently continue LatentGuard. I would define a separate project
whose data-generation process deliberately varies actions at matched states and
validate its branching contract before training. It would need a new
preregistered protocol, repository, and success criteria.

### How does this project demonstrate your ability?

It shows I can integrate complex robot-learning stacks, design experiments that
separate correlation from actionable evidence, diagnose simulator validity,
maintain reproducibility across remote runs, preserve negative results, and
make an evidence-based stop decision.
