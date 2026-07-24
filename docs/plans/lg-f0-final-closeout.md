# LG-F0: LatentGuard Research Closeout and Portfolio Release

## Purpose

LG-F0 closes the post-release LatentGuard research line without starting a new
experiment. It converts the committed LG-R0 through LG-RB0.1 evidence into a
portfolio-first README, a source-bound final report, a result matrix, five
reproducible static figures, and interview-ready materials.

## Baseline and branch

- Baseline branch: `codex/lg-rb01-robolab-replay-compat`
- Baseline commit: `6022c1a6821e42a46d69be24d19467de92de842d`
- Closeout branch: `codex/latentguard-final-closeout`
- Local tracked source is authoritative.

## Evidence policy

- Every final number is read from an existing committed JSON artifact.
- `scripts/lg_f0_build_summary.py` derives the canonical compact summary and
  records the source SHA-256 digests and JSON pointers.
- Figures read only `artifacts/final/final_summary.json` and embed its digest in
  their PNG metadata.
- Historical gates and negative results are copied by meaning, not redefined.
- Facts, interpretations, and future directions remain explicitly separated.

## Outputs

- Rewrite `README.md` for a portfolio reader.
- Add `docs/latentguard_final_report.md`.
- Add `docs/latentguard_result_matrix.md`.
- Add `artifacts/final/final_summary.json`.
- Add no more than five source-bound PNG figures under `docs/figures/`.
- Add the four requested portfolio documents under `docs/portfolio/`.
- Add minimal navigation in `docs/index.md` and `artifacts/README.md`.
- Preserve the existing RoboLab issue draft, PR draft, and patch.

## Exclusions

LG-F0 performs no training, rollout, simulator launch, new data collection,
checkpoint generation, final-seed access, model integration, candidate ranking,
intervention, replay repair, third-simulator migration, or external issue/PR
publication. It does not modify LangMani.

## Validation

The closeout must pass:

- the complete CPU-only test suite;
- Ruff lint and format checks;
- strict MyPy checks for `src`;
- wheel and source-distribution builds;
- deterministic final-summary regeneration;
- JSON parsing, Markdown relative-link checking, figure-source metadata checks,
  public-surface source/secret/path checks, and stop-statement checks;
- Git diff, staged-diff, upstream parity, and clean-tree checks.

## Stop decision

The exact same-state counterfactual route is closed. The project will not move
to a third simulator, add another reward model, or continue replay-infrastructure
repair.

精确同状态反事实路线已经关闭。本项目不会迁移到第三个模拟器，不会继续增加奖励模型，也不会继续修复重放基础设施。
