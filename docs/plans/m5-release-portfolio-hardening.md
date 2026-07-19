# M5 Release, Evidence Consolidation, and Portfolio Hardening

## Initial repository state

- Starting revision: `942c6572914ef9c7fa6e5f8affdd8c9004ed4916`.
- Starting branch: `codex/m4d-conservative-fallback-shield` with a clean tree and matching upstream.
- M5 branch: `codex/m5-release-portfolio-hardening`.
- The PickCube research line is frozen at M4D. M5 adds no model training, seeds, simulator execution, candidate outcomes, rendering, threshold tuning, VLM/LLM use, LangMani integration, multi-GPU work, or real-robot work.

## Accepted milestone commits

| Milestone | Implementation or accepted execution | Accepted result |
| --- | --- | --- |
| M0 | `65dac26c774fa347dd08133eca9449a1afe16424` | same commit |
| M1 | `f7e5eb0fb01a3e2a6546ce163ed762292245b197` | same commit |
| M2A | `b55d5ffbc3f94491b1fc3a61b3549f8a65b45a22` | same commit |
| M2B | `246e086340ee3e054de2001c23c631d8a7b03239` | same commit |
| M2C | `aafe83806515662937f6ec6c5d695aa44d1da537` | `ca8e6a8b9d4eb305b8b1bfbdaa9f4d29e0efb842` |
| M3A | `cc01a01a22bd8e53f4a442d0a6f7fd561d0ab85a` | `e960023a4710b8dbd3693fd763c3126115e35e81` |
| M3B | `46f15d8cf50ac45027b62b0d73a4f2f49ac30224` | `66eaee0f69a8f5dfe4bc7a53a777e08b5ce51b88` |
| M3C | `a632a702c709edb1fc21e702c83e30964652ff79` | `ef78a78cb8fecf44b9705d326949f23208f3de4f` |
| M4A | `7a2a073666e455b36da8e72a2b87350a2baf3582` | `6e2c573d55c721158d174f8d76e7d8478a767835` |
| M4B | `d76013e1434c7a0888ad0e2f2e1ec5077a5cd048` | `325a42a8ab56e0386defb24c5411f45d69ad0497` |
| M4C | `ca348a94106bae3a0d19826f2883659451898ae3` | `6f3d3420180d8cebd4eb7bcba7db4438b1deaf2f` |
| M4D | `0f24befbfc84a502c1af60e06bbeb8cd734a7863` | `942c6572914ef9c7fa6e5f8affdd8c9004ed4916` |

## Source reports reviewed

The release registries use committed compact reports under `reports/m2c` through `reports/m4d`. The primary sources are the accepted M2C replay summary, M3A acceptance summary, M3B benchmark summary, M3C candidate-selection report, M4A development/external dataset summaries, M4B result summary, M4C execution summary, and M4D result summary. File-byte SHA-256 digests and report JSON pointers are recorded in the registries so every published metric can be checked against its original field.

## Deliverables

- Strict milestone, claims, results, and release-manifest registries.
- CPU-only deterministic portfolio smoke and strict release audit commands.
- Portfolio-grade README, technical report, case study, architecture diagrams, canonical results table, demo script, and presentation outline.
- English/Chinese project summaries, resume bullets, and interview brief.
- Tests for schema rejection, evidence/digest checking, result consistency, public-surface hygiene, deterministic smoke, and resume idempotence.
- A compact final review under `reports/m5/<release-id>/` after the implementation commit is pushed.

## Validation plan

Run the complete CPU suite, Ruff lint/format checks, strict mypy, `git diff --check`, `latentguard portfolio-smoke`, and `latentguard audit-release --strict`. Also scan tracked public surfaces for secrets, private absolute paths, raw/large artifacts, stale shutdown language, placeholders, and broken internal links.

## Commit, push, and final review

After implementation validation, commit exactly `feat(m5): add release evidence and portfolio package`, push automatically, and prove local/upstream/GitHub SHA equality. From that clean pushed revision, rerun the strict audit, perform the read-only remote preservation audit, create compact sanitized M5 review reports, rerun all validation, commit exactly `docs(m5): publish portfolio-ready release`, push, and verify final parity. No release tag is authorized.

## Server preservation

The remote server is an availability-audit target only in M5. The audit is read-only, performs no GPU computation, does not alter remote Git or artifacts, and records only sanitized compact availability facts. The server must remain powered on and SSH-ready during and after M5; no shutdown monitor is permitted.

## Final results

The implementation validation passed on the M5 branch: 1,565 tests passed and three Windows directory-symlink tests skipped because the current account lacks symlink privilege; Ruff lint and format checks passed; mypy passed for 174 source files; the deterministic CPU portfolio smoke passed with zero-work resume; and the strict release audit passed for 12 milestones, 22 claims, and 57 results with zero issues. The implementation push, clean-revision review, read-only server preservation audit, and final result commit remain pending at this stage.
