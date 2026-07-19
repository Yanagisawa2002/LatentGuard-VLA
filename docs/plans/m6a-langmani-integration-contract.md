# M6A LangMani integration contract and readiness audit

## Objective

Define a safe, versioned boundary through which LatentGuard-VLA may eventually
consume LangMani task context, observations, action proposals, replay snapshots,
and outcome evidence. M6A is an infrastructure-only readiness audit. It does not
execute M6B, train a model, run a simulator benchmark, generate outcomes, or make
a LangMani performance claim.

## Frozen inputs

- LatentGuard starting release: `14c2a1513690791d88cc1b6ad46a09b3e852ee60`.
- M0--M4D PickCube evidence and the M5 release remain frozen.
- Audited LangMani repository: `Yanagisawa2002/LangMani`.
- Audited branch: `codex/m5a-language-routing`.
- Audited clean HEAD/upstream: `b0fd9115496d3cc3d700c4eac7fb9f3f3ad401fd`.
- Local and remote absolute paths are operational metadata and are excluded from
  committed semantic identities.

## Planned scope

1. Add strict JSON-only schemas for task, observation, policy, action, replay,
   outcome, and integration-manifest records.
2. Add an import-safe adapter protocol and a deterministic synthetic adapter.
3. Add a default-static `audit-langmani-contract` command with an explicitly
   opt-in schema import probe.
4. Audit the real checkout read-only and record Git blob identities for source
   evidence plus SHA-256 identities for accepted compact artifacts.
5. Freeze a compatibility matrix, blocker registry, readiness gates, and an
   unexecuted bounded M6B design.
6. Extend release records through post-release additive files without changing
   M5 registry or release-manifest bytes.
7. Run ordinary CPU validation, strict release/audit commands, and a read-only
   server preservation check; commit and push implementation and result commits.

## Assumptions

- The real LangMani checkout remains read-only and clean throughout M6A.
- LangMani's accepted PerTask ACT controller registry remains the checkpoint
  identity source; M6A does not choose a task-specific controller for M6B.
- Cross-project interchange uses canonical JSON records across separate Python
  processes because the current Python/NumPy pins are not in-process compatible.
- Synthetic adapter evidence validates only LatentGuard infrastructure.

## Exclusions

- No LangMani tracked-source changes, checkpoint changes, training, rendering,
  simulator rollout, outcome generation, threshold tuning, candidate pool, VLM,
  LLM, distributed execution, multi-GPU work, or real-robot work.
- No claim that LatentGuard improves LangMani.
- No transfer of the PickCube eight-candidate pool.
- No exact replay label without accepted complete simulator plus policy-state
  restoration evidence.

## Initial audit conclusion

The task, action dimension, policy observation, action normalization, projection,
success, termination, and accepted checkpoint-registry contracts are inspectable.
The real integration is not ready for M6B. Candidate-source freezing, one exact
task/checkpoint binding, compact verifier-input freezing, intermediate paired
replay, ACT queue/history restoration, continuation semantics, and the isolated
runtime bridge remain unresolved. Replay readiness is `initial_state_only`.

## Validation and completion log

The implementation commit is
`c06eed9f83325909ce44073705c7628d69a09c6a` and was pushed before the final
read-only audit. The compact audit ID is
`20260719T150000Z_m6a-contract-c06eed9`; its strict contract audit passed with
zero audit issues while correctly reporting overall M6B readiness as `blocked`.

Observed validation at the implementation revision:

- `python -m pytest`: 1,580 passed and three pre-existing Windows
  symbolic-link privilege skips.
- `ruff check .`, `ruff format --check .`, `mypy src`, and `git diff --check`:
  passed.
- `latentguard portfolio-smoke`: passed with deterministic CPU infrastructure
  identity and zero-work resume.
- `latentguard audit-release --strict`: passed with 12 milestones, 22 claims,
  57 results, and zero issues.
- `latentguard audit-langmani-contract --strict`: passed with zero audit issues.

The final matrix records 11 compatible, seven adaptable, five blocked, and zero
unknown rows. Six of 15 M6B readiness gates remain blocked by six explicit
blockers (two critical and four major). Replay readiness is
`initial_state_only`; M6B was not executed.

The server preservation audit was read-only: SSH remained available, no GPU
compute process was running, accepted run roots/environments/checkpoints/caches
remained present, no remote change was made, and the server remained online.
The final compact review and sanitized retrieval manifest are under the audit ID
above. A separate result commit records these additive reports and registries;
the frozen M5 release-manifest byte digest remains
`4023dad2253156b56fc094ddee2e1e99afce5ee8843a3caf27bbaafc6ce07377`.
