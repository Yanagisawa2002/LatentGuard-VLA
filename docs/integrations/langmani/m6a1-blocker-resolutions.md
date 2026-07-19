# M6A.1 blocker resolutions

This additive registry does not replace or rewrite the original M6A blockers. Before the one real
probe, readiness is `conditionally_ready`: implementation and CPU gates exist, while exact
task/checkpoint binding, cross-process exchange, and accepted-checkpoint mask invariance still
require remote evidence.

| Blocker | Bounded-M6B disposition | Retained limitation |
| --- | --- | --- |
| M6A-B003 | Candidate source frozen as four blind, content-bound candidates | No alternate candidate source is authorized |
| M6A-B004 | `scope_resolved_initial_state_only` | No complete ACT queue/processor snapshot |
| M6A-B005 | Action-only allowlist frozen; real mask gate pending | No structured LangMani verifier state |
| M6A-B006 | `scope_resolved_seeded_initial_reset_and_fixed_continuation` | No intermediate paired replay |
| M6A-B007 | Strict JSON boundary implemented; real 3.12↔3.11 exchange pending | In-process dependency use remains invalid |
| M6A-B009 | Outcome-free lexical selection implemented; exact binding pending | No other task/checkpoint is authorized |

The future continuation executes ten candidate actions and then replays recorded projected nominal
actions beginning at index ten. It never requeries ACT after divergence. This is reproducible but
is not faithful same-policy continuation and does not establish closed-loop shielding.
