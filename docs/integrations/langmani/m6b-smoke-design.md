# M6B bounded smoke design (not executed)

## Status

**Blocked.** This is a design artifact, not authorization to run M6B. Entry is
prohibited until every required gate in the M6A report is passed by a new clean,
pushed LatentGuard revision and the exact LangMani revision remains available.

## Smallest credible scope after remediation

- one canonical LangMani task and its one accepted PerTask ACT checkpoint;
- one fixed canonical instruction;
- one GPU and one LangMani process, with LatentGuard communicating only through
  the versioned JSON boundary;
- a predeclared small set of source seeds that excludes accepted development and
  final seeds;
- an initial-reset-only paired boundary unless complete intermediate simulator
  and policy-state replay is separately validated;
- a small, content-bound candidate count from exactly one source chosen and
  frozen before any outcomes exist;
- both raw postprocessed and projected executable action evidence, with only the
  projected sequence eligible for execution;
- no training, threshold fit, checkpoint selection, generalization claim, or
  performance-improvement claim.

## Required entry gate

1. Freeze one task ID and exact controller-registry entry.
2. Freeze the executable proposal shape, including whether the scored horizon is
   the first 10 actions of the 50-action policy chunk.
3. Freeze one candidate source, count, deterministic seed semantic, and blind
   manifest before loading or generating simulator outcomes.
4. Freeze a compact verifier input allowlist with no target index, task outcome,
   candidate provenance, or reporting-only field.
5. Choose one continuation semantic. For the current capability, the only honest
   starting option is a bounded initial-reset episode contract; intermediate
   same-policy continuation remains blocked.
6. Validate the out-of-process Python/NumPy bridge, exact checkpoint artifacts,
   active action bounds, and zero import-time LangMani dependency in LatentGuard.
7. Reconfirm server online, single GPU available, accepted artifacts preserved,
   clean exact SHAs, and no other active compute.

## Provisional execution design

The candidate source is intentionally not selected in M6A. After a separate
review chooses it, a minimal smoke should use 2--3 untouched source seeds and no
more than 2--4 candidates per seed. Candidate ranking input must be finalized and
content-bound before outcome execution. Each candidate begins from an independently
seeded, state-verified initial reset under the same task and checkpoint binding.

The baseline is the accepted nominal projected policy execution. A mechanically
useful secondary baseline may be random choice over the identical frozen pool.
Expert or wrong-task controllers are not deployable selector inputs. Raw versus
projected action evidence is reported separately and never counted as two
independent candidates without an explicit reviewed semantic.

## Success criterion

The smoke succeeds only as an integration gate when all planned episodes finish
with complete, digest-valid evidence; identical task/checkpoint/action contracts;
no outcome leakage; no missing or duplicate work after resume; and correct
separation of success, failure, timeout, invalid action, projection failure, and
execution errors. Policy performance is descriptive only and cannot promote a
claim in this bounded smoke.

## Stop conditions

Stop before or during execution on SHA drift, dirty worktree, checkpoint or
processor mismatch, task mismatch, action-shape/bounds mismatch, non-finite value,
unfrozen candidate identity, failed state comparison, policy-history ambiguity,
outcome leakage, duplicate resume work, GPU contention, or infrastructure error.

## Estimated work and required artifacts

After the blockers are resolved, expected execution is one short single-GPU job
plus CPU validation. Required compact artifacts are repository identities,
resolved dependency manifests, task/checkpoint binding, blind candidate manifest,
raw/projected action records, replay and continuation evidence, outcome taxonomy,
resume ledger, strict validation summary, and a sanitized retrieval manifest.
Datasets, simulator states, images, checkpoints, and caches stay outside Git.
