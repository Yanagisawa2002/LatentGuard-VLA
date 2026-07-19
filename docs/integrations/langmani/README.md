# LangMani integration status

M6A documents and audits the optional cross-project contract between
LatentGuard-VLA and LangMani. It does not execute a LangMani policy or simulator.

The exact audited checkout is `Yanagisawa2002/LangMani`, branch
`codex/m5a-language-routing`, clean HEAD and upstream
`b0fd9115496d3cc3d700c4eac7fb9f3f3ad401fd`. The path used to locate that checkout
is deliberately absent from Git.

Current readiness is **blocked**. LangMani has a clear six-task schema, accepted
PerTask ACT checkpoint registry, 8D `pd_joint_pos` action contract, saved
MEAN_STD processors, deterministic environment-bound projection, official task
evaluation, and initial-reset replay evidence. It does not yet expose a complete
ACT policy-history snapshot or accepted paired execution from intermediate
restored states. LatentGuard has not frozen an M6B candidate source, compact
verifier input, one task/checkpoint, or continuation semantic.

Start with:

- [architecture.md](architecture.md) for the four-layer boundary and field
  separation;
- [compatibility-matrix.md](compatibility-matrix.md) for the source-backed
  cross-project audit;
- [blockers.md](blockers.md) for required remediation;
- [m6b-smoke-design.md](m6b-smoke-design.md) for the unexecuted bounded design.

Run the static audit by setting `LATENTGUARD_LANGMANI_ROOT` to the operational
checkout or passing `--langmani-root`, then running:

```bash
latentguard audit-langmani-contract --strict
```

The optional `--allow-import-probe` loads only the approved task schema module.
It starts no simulator, GPU model, rollout, or checkpoint load. Default behavior
does not import LangMani.

The synthetic adapter always reports:

> Synthetic LangMani adapter validation does not establish compatibility with
> the real LangMani simulator or policy.
