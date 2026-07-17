# Blind verifier-guided candidate selection

## Question and scope

M3C asks whether the frozen M3B Direct Action Verifiers improve real PickCube
outcomes when choosing one of eight unseen 16-step action chunks from a verified
38-component state. It is a controlled, one-shot candidate-selection study.
It is not receding-horizon control: the policy is not called again after the
candidate prefix, and every replay uses the same byte-identical archived source
continuation.

The original source prefix is never a candidate or fallback. It remains only
part of the mandatory source-remainder baseline and the fixed continuation.
Candidate admission rejects a prefix that is byte-identical to the source,
changes any continuation byte, violates the verified action contract, contains
a non-finite value, or would require clipping or repair. Every accepted group
contains exactly eight deterministic candidates.

## Three-stage blindness contract

Stage A receives only a content-bound source/state reference, the 38-component
verifier state, the eight candidate chunks and masks, frozen selector
configuration, and frozen verifier bundle identities. The selection command has
no evidence, outcome, or replay-output argument. Its input root has a strict
allowlist and its output directory must be new and empty. The finalized
selection manifest records complete rankings, selected proposal IDs,
probabilities, abstentions, and
`outcomes_available_during_selection=false`. It stores no states, actions,
outcomes, or evidence.

This proves capability and input isolation for the selection process; it does
not claim to prove that no unrelated outcome file existed anywhere on the
machine. The manifest's semantic identity excludes the audit timestamp, while
the immutable report envelope still protects the timestamp from tampering.
This capability-level declassification is the Stage A security claim: candidate
content is available, but no interface can name or load evidence, replay, or
outcome artifacts. Complete-pool replay evidence and outcome data become
available only to Stage B/C after the Stage A process has closed.

Stage B begins only after the Stage A manifest is finalized. It replays the
union of all non-abstained selected proposals. If multiple selectors choose the
same proposal, the simulator executes it once. Stage C replays the exact
complement. Both phases reuse the existing M2A ledger and M2B/M2C exact-state
paired replay. Their executable inventories are disjoint and exhaustive, so
each full-pool proposal receives one physical replay in total. A second resume
must perform zero new work.

Only after Stage C may the analysis-only oracle and complete-pool ranking
metrics be constructed. Result loading rejects a changed selected ID, ranking,
bundle, pool, source set, evidence inventory, or full-pool outcome binding.
It also checks the real timestamps persisted by the two evaluation run
manifests. The required order is
`selection < selected start <= selected finish < remainder start <= remainder
finish`, and the final bound outcome timestamp is the recorded remainder finish.
No timestamp is inferred or advanced synthetically.

## Frozen selectors

All selectors see the same candidate IDs and action tensors. Candidate family,
distribution, severity, source/corrupted status, trajectory identity, anchor
reason, evidence, and outcomes are not model inputs.

- Deterministic random hashes the selector configuration and group identity,
  then indexes the candidate IDs in sorted order.
- Action magnitude uses the exact M3B train-fitted heuristic identity.
- Action-only, state+action MLP, and temporal selectors average five
  validation-calibrated failure probabilities arithmetically and rank by
  `(mean failure probability, candidate ID)`.
- Oracle selects a successful candidate with stable ID tie-breaking, and is
  unavailable during Stage A.

The exact Stage A inventory contains eleven selector IDs: the five non-oracle
selectors above plus six temporal variants using maximum validation balanced
accuracy, target validation failure recall, and approximately 90, 80, 70, and
50 percent coverage. The three learned selectors load exactly three five-seed
bundles (action-only, state+action MLP, and temporal), for 15 checkpoints.
Selector IDs, random seed, bundle digests, action-magnitude digest, all six
policy digests, and inference measurement semantics are stored in a strict
selector-configuration report whose outer digest is bound by the manifest.

The temporal ensemble remains the primary selector because M3B selected it on
validation failure AUPRC. The joint MLP remains the efficiency challenger.
M3C outcomes cannot change those roles.

## Frozen-artifact trust and profiling

The committed M3B compact benchmark summary is the authority for runtime
calibration and threshold reports. Preparation requires every runtime report's
outer envelope digest to match its architecture/seed entry in that summary;
inner self-consistency is not enough. Before corrupted validation rows can fit
ensemble abstention, the complete ordered raw logits and labels for each seed
must reproduce the stored validation-prediction digest, and every seed must
cover the same ordered sample/group inventory. The preparation summary binds
the three bundle reports, baseline report, policy report, preprocessing, and
source M3B summaries.

Before any new M3C source trajectory is collected, checkpoint preparation runs
a fixed synthetic, outcome-free eight-candidate forward pass through all three
five-seed ensembles on the requested CPU or GPU device. The compact
`m3c_precollection_inference_smoke_v1` report binds input and output digests but
stores no raw logits, probabilities, states, actions, or M3C source identity.
This makes the CPU/GPU forward gates possible before collection rather than
treating checkpoint loading alone as inference.

CPU and GPU selection runs produce separate compact inference-latency reports.
They include per-candidate and per-group p50/p95/p99, throughput, peak allocated
device memory, bundle and per-model loading time, and an explicit joint-MLP
versus temporal difference. They bind the blinded input, full-pool digest,
protocol, selector configuration, and three bundle digests. These diagnostics
cannot affect ranking and store neither raw predictions nor raw duration
samples. Final evaluation requires distinct `--cpu-latency-report` and
`--gpu-latency-report` inputs.

Stage A is published as one atomic directory containing the manifest, selector
configuration, and an internal latency report. A partial or interrupted staging
directory is never the authoritative Stage A root. An optional external latency
copy is published only after that directory commits and can be recovered from
the internal report by a safe resume.

The final replay evidence digest is semantic: it binds the pool, semantic
selection manifest, exact phase inventories, and complete task/strong-replay
values, while excluding run timestamps, host/platform text, launch paths,
ledger mechanics, and the timestamped manifest envelope. A separate archive
audit digest binds both exact evaluation-dataset digests and the envelope, so
operational tampering is still detected without polluting replay identity.
Completed selected and remainder resumes emit compact zero-work proofs outside
the strict replay bundles. Final evaluation reloads those proofs and writes a
combined resume report plus a byte-bound human-readable review.

## Candidate distributions

The checked-in `candidate-pool-v1.json` fixes four ID-like slots using reviewed
M3A definitions: arm Gaussian noise at 0.005, arm bias at 0.01, arm Gaussian
noise at 0.05, and an all-component hold beginning at step 1. Four shifted
slots use arm Gaussian noise at 0.02, arm bias at 0.05, an all-component hold
over steps 4 through 11, and an arm permutation over steps 2 through 9. Slot and
distribution metadata is reporting-only.

Smoke uses six trajectories solely to test mechanics. Full evaluation uses 60
untouched trajectories from a disjoint seed range. Seed, trajectory, split
group, and every complete state-tree digest are checked against M3A and against
the smoke set. Any post-smoke configuration change requires a new local commit,
new pushed revision, and a new untouched full set.
Full pool construction explicitly reloads the smoke pool and verifies all four
disjointness axes before accepting the full source set. Smoke outcomes cannot
be promoted into the full benchmark or used to tune a full configuration.

## Abstention, metrics, and statistics

Temporal abstention thresholds are derived only from the accepted M3B
validation predictions after averaging the five calibrated seed probabilities.
Per-seed thresholds are not averaged. Maximum-balanced-accuracy and
target-failure-recall thresholds are fitted on the complete candidate-level
corrupted validation inventory, which retains both outcome classes. The
approximately 90/80/70/50 percent coverage thresholds use the per-group minimum
ensemble scores because deployment first selects each group's minimum-score
candidate. All six policies and this fitting semantic are content-bound before
full execution.

An abstention executes no proposal, lowers coverage, and is neither a success
nor a task failure. Outcome rates therefore carry explicit numerators,
denominators, executed counts, and abstained counts. Reports retain all groups,
solvable groups, mixed groups, all-success groups, and all-failure groups, plus
ID-like and shifted slices where meaningful.

Complete-pool analysis includes top-1/top-2 success, pairwise
success-over-failure concordance, reciprocal rank of the first success, binary
NDCG, oracle regret, and eligible counts. Paired uncertainty intervals resample
complete original source trajectories at least 2,000 times; candidate groups
and individual candidates are never the bootstrap unit.

The result report evaluates the predeclared targets without tuning: at least
0.08 temporal success improvement over random on solvable groups, at least 30%
relative task-failure reduction, pairwise concordance at least 0.75, oracle
success regret at most 0.10, zero execution errors, complete strong replay, and
the approximately 70% policy retaining at most half the random failure rate.
It also records whether temporal-versus-random and temporal-versus-joint paired
intervals exclude zero. A failed or undefined target produces an explicit
negative-result disposition; it never triggers full-outcome tuning or a success
claim.

## Limitations

M3C uses privileged structured simulator state for one robot, one task, one
controller, and one fixed continuation policy. It does not establish visual or
language understanding, cross-task performance, general safety, or real-robot
validity. VLM and LangMani work remains deferred until this smaller experiment
demonstrates intervention value without outcome leakage.

The fixed continuation is an experimental control, not a deployable closed-loop
policy. The verifier chooses one 16-step prefix, receives no new observation,
does not call the source policy or selector again, and then executes the
byte-identical archived continuation. Consequently later success or failure can
depend strongly on that continuation. M3C therefore makes no receding-horizon,
online intervention, or policy-improvement claim.

The ordered one-RTX-5090 gates completed at
`a632a702c709edb1fc21e702c83e30964652ff79`. The untouched full set contains 60
source trajectories, 360 groups, and 2,880 complete strong simulator-verified
outcomes. Temporal selected success was 0.98333 versus 0.88889 for deterministic
random, with a trajectory-bootstrap difference of 0.09444 and 95% interval
[0.06389, 0.12500]. Joint MLP selected success was 0.99444; the temporal-minus-
joint interval [-0.02500, 0.00000] does not support temporal superiority. All
predeclared quality targets passed without full-outcome tuning. The compact
review and content-bound reports are stored under
`reports/m3c/20260716T152012Z_m3c-full_a632a70_seed271828/`.
