# WM-v0 D1 real-policy counterfactual dataset scale-up plan

## Objective

Expand the six accepted real PickCube counterfactual futures into a group-isolated,
class-valid dataset dominated by action chunks that were actually produced by a
policy at the current anchor. This milestone ends either with the formal data gate
and required offline comparisons, or with a compact Result B diagnosis that names
every failed gate and starts no formal training.

Development starts from clean commit
`c786e5d03b598ccd4e9f22de25486f87c6c162a4` on independent branch
`codex/wm-v0-d1-policy-dataset`. Historical release evidence, `portfolio-v1`, and
the accepted six-sample smoke remain immutable.

## Assumptions and current blocker

- The accepted PickCube exact-state restoration, state-preserving visual capture,
  action execution, and terminal task evaluation contracts remain the physical
  data source.
- A candidate is `policy_generated` only when a real policy inference consumes the
  current observation, proprioception, and task condition. Its checkpoint digest,
  inference seed/configuration, horizon, and latency are mandatory.
- The 2026-07-21 server audit found no accepted LangMani registry,
  runtime-selection record, ACT checkpoint, or processor artifacts. The earlier
  M6A.1 probe therefore performed zero policy queries. Historical action snippets
  and M3C perturbations remain `replay_reference` or `synthetic_corruption`.
- Until a content-verified provider becomes available, a bounded synthetic-only
  simulator pilot may validate mechanics and cost, but it cannot pass the D1 pilot
  gate or authorize formal collection/training.

## Implementation sequence

1. Preserve a concrete audit of all six accepted samples and current throughput.
2. Add typed `CandidateProvider`, `CandidateContext`, and `PolicyCandidate`
   contracts with strict action/provenance validation and content deduplication.
3. Add phase-aware anchor metadata. Unreliable phase or state fields are recorded
   as `unknown`/`null`, never inferred from outcomes.
4. Commit each anchor atomically with samples, candidate metadata, sample index,
   manifest, and completion marker. Resume quarantines incomplete staging and
   never re-executes complete anchors.
5. Run 25--50 anchors and at least 100 futures as a single-GPU pilot from an exact
   pushed revision. Stop if any pilot gate fails.
6. Only after every pilot gate passes, run the minimum 1,000-sample formal
   expansion and then the three required offline model comparisons.
7. Retrieve only compact sanitized manifests/reports. Raw RGB, states, candidates,
   caches, checkpoints, and predictions remain outside Git.

## Validation and stop conditions

The pilot must prove zero restoration mismatch, split leakage, and duplicate
identity; at least 99% complete futures; 100% complete candidate metadata; at
least 70% real policy candidates; both terminal classes; and acceptable explained
rejections/cost. A failed check stops expansion and training.

Formal training additionally requires at least 1,000 samples, 250 anchors, 100
source episodes, 100 successes, 100 failures, all three splits, nonterminal
progress/event supervision, and all data-quality checks. Synthetic actions never
contribute to the real-policy quota.

## Exclusions

- no closed-loop action execution selected by WM-v0;
- no planner, shield, online intervention, or controller mutation;
- no success-rate or formal model-performance claim from a failed gate;
- no remote tracked-source edits, no multi-GPU/distributed run, and no silent
  candidate fallback; and
- no server shutdown unless the user explicitly requests it in this task.
