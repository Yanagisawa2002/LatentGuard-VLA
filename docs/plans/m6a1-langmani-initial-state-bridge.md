# M6A.1 initial-state LangMani bridge plan

## Goal

Resolve the six M6A entry blockers only for a bounded M6B integration smoke. The two accepted
Python environments communicate through canonical JSON; neither repository imports the other.

## Frozen scope

- LatentGuard branch `codex/m6a1-langmani-initial-state-bridge` from
  `2afdd438e1baf303c112e652877e58f96f8b6cb`.
- LangMani branch `codex/m5b-latentguard-proposal-bridge` from
  `b0fd9115496d3cc3d700c4eac7fb9f3f3ad401fd`.
- Lexically first eligible accepted PerTask controller and one canonical task.
- One reset-only proposal probe, one policy query, zero `env.step`, zero outcomes.
- Identity plus the first three lexically eligible existing corruption IDs, for four candidates.
- Accepted M3B action-only five-seed ensemble and `LangManiActionOnlyVerifierInputV1`.
- Ten projected actions, six canonical zero-padding slots, and mask `true[10] + false[6]`.
- Initial seeded reset plus fixed recorded nominal continuation; no policy requery after divergence.

## Execution order

1. Implement and fully validate both repositories locally.
2. Commit and push each implementation revision.
3. Create separate clean remote worktrees at those exact revisions.
4. Run one relevant integration subset and one reset-only proposal probe.
5. Exchange policy binding, proposal, raw candidates, and projected candidates as canonical JSON.
6. Run the accepted action-only ensemble, mandatory mask-invariance gate, blind ranking, strict
   reload, and readiness audit.
7. Retrieve only compact sanitized evidence, validate locally, update additive registries, commit
   and push result revisions, then confirm parity.

## Stop conditions

Stop on SHA or digest drift, dirty remote tracked source, task/checkpoint ambiguity, fewer than
three eligible existing corruptions, non-finite actions, projection mismatch, failed mask
invariance, outcome availability, any action execution, GPU contention, or cross-process schema
drift. A failed real gate remains retained evidence and readiness stays blocked.

## Exclusions

No training, fine-tuning, added seed, candidate execution, rollout, task outcome, threshold fit,
calibration, model selection, VLM/LLM, multi-GPU work, M6B execution, intermediate policy snapshot,
or closed-loop claim is authorized. The server remains online after completion.

## Final preflight disposition

Both implementation revisions were committed, pushed, and validated in separate clean Linux
worktrees. The current execution server did not contain the accepted LangMani controller registry,
runtime-selection record, ACT checkpoint, or processor artifacts required to rebuild and verify the
selected binding. The deterministic registry declaration identifies the blue-cube/left-bin task and
its exact digests, but that declaration is not a substitute for current-server artifact validation.

The real probe therefore stopped before environment construction. It performed zero resets, zero
policy queries, zero `env.step` calls, zero candidate projections, zero scorer calls, and produced no
candidate pool, ranking, or outcome. The cross-process and mask-invariance implementations passed
CPU fixture tests, but their accepted-artifact gates remain unexecuted. Bounded M6B readiness is
`blocked`; M6B was not started. The server remains online and GPU-idle.
