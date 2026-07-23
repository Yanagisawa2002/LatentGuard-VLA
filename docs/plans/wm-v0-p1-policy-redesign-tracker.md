# WM-v0 P1 experiment tracker

**Status:** design-only tracker. Every remote row is blocked pending a separate
authorization; no run ID or outcome exists yet.

| Run ID | Stage | Purpose | System/variant | Data/split | Selection source | Priority | Status | Stop/go decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1-L00 | P1-0 | Validate contracts and configs | shared P1 infrastructure | synthetic/local fixtures | none | MUST | TODO_NEXT_AUTH | all CPU checks pass |
| P1-D00 | P1-1 | Inventory retained full-state archives | preferred data path | historical train/validation sources | provenance only | MUST | BLOCKED_NO_REMOTE_AUTH | exact state-bound labels possible or conservative new collection required |
| P1-D01 | P1-1 | Build and reload accepted annotated dataset | preferred Direction 1 | source-group train/validation/test | no outcomes | MUST | BLOCKED_NO_REMOTE_AUTH | all data/integrity/leakage gates pass |
| P1-G00 | P1-2 | GPU forward/backward and tiny overfit | Direction 1 | accepted train/validation | validation only | MUST | BLOCKED_NO_REMOTE_AUTH | finite, bounded, save/load/resume pass |
| P1-S00 | P1-3 | Seed-0 full-data screen | Direction 1 preferred | full accepted train; validation selection | validation only | MUST | BLOCKED_NO_REMOTE_AUTH | exactly one checkpoint frozen |
| P1-A00 | P1-3 | Essential no-history ablation | Direction 1 deletion study | same data | validation only | MUST_IF_S00_OFFLINE_PASSES | BLOCKED_NO_REMOTE_AUTH | isolate history contribution without model expansion |
| P1-B00 | P1-4 | Paired direct baseline | frozen P0.2 step-18k/H=2 | proposed `820000..820029` | none; frozen identity | MUST | BLOCKED_NO_REMOTE_AUTH | integrity pass; preserve outcome |
| P1-E00 | P1-4 | Seed-0 preferred development gate | Direction 1 frozen checkpoint | proposed `820000..820029` | no checkpoint selection | MUST | BLOCKED_NO_REMOTE_AUTH | all 24/21/18/15/23 gates pass |
| P1-S01 | P1-5 | Stability training | Direction 1 seed 1 | full accepted train; validation selection | validation only | MUST_IF_E00_PASSES | BLOCKED_NO_REMOTE_AUTH | frozen config/checkpoint rule |
| P1-S02 | P1-5 | Stability training | Direction 1 seed 2 | full accepted train; validation selection | validation only | MUST_IF_E00_PASSES | BLOCKED_NO_REMOTE_AUTH | frozen config/checkpoint rule |
| P1-E01 | P1-5 | Seed-1 development gate | Direction 1 seed 1 | same frozen Direction-1 manifest | none after freeze | MUST_IF_E00_PASSES | BLOCKED_NO_REMOTE_AUTH | all gates pass |
| P1-E02 | P1-5 | Seed-2 development gate | Direction 1 seed 2 | same frozen Direction-1 manifest | none after freeze | MUST_IF_E00_PASSES | BLOCKED_NO_REMOTE_AUTH | all gates pass |
| P1-R00 | fallback | Seed-0 low-cost backup | recurrent one-step BC | existing 400/50 split | validation only | MUST_IF_DIRECTION_1_STOPS | BLOCKED_NO_REMOTE_AUTH | one checkpoint frozen |
| P1-RB0 | fallback | Backup direct baseline | frozen P0.2 step-18k/H=2 | proposed `821000..821029` | none | MUST_IF_R00_RUNS | BLOCKED_NO_REMOTE_AUTH | integrity pass; preserve outcome |
| P1-RE0 | fallback | Backup development gate | recurrent one-step BC | proposed `821000..821029` | no checkpoint selection | MUST_IF_R00_RUNS | BLOCKED_NO_REMOTE_AUTH | all fixed gates pass or stop |
| P1-M00 | conditional | Conditional-action ambiguity audit | train/validation-only audit | existing train/validation | no simulator outcomes | MUST_BEFORE_DIFFUSION | BLOCKED_NO_REMOTE_AUTH | multimodality proven or Direction 3 cut |
| P1-F00 | conditional | Seed-0 compact diffusion screen | diffusion policy | existing 400/50 split | validation only | ONLY_IF_M00_PASSES | BLOCKED_NO_REMOTE_AUTH | one checkpoint frozen |
| P1-FB0 | conditional | Diffusion direct baseline | frozen P0.2 step-18k/H=2 | proposed `822000..822029` | none | ONLY_IF_F00_RUNS | BLOCKED_NO_REMOTE_AUTH | integrity pass; preserve outcome |
| P1-FE0 | conditional | Diffusion development gate | compact diffusion | proposed `822000..822029` | no checkpoint selection | ONLY_IF_F00_RUNS | BLOCKED_NO_REMOTE_AUTH | all fixed gates pass or stop |
| P1-P00 | P1-6 | Package seed-0 candidate | predeclared promoted policy | no new outcomes | frozen identities | MUST_BEFORE_FINAL | BLOCKED_BY_PROMOTION | package/determinism/resume pass |
| P1-Z00 | P1-6 | Sealed final evaluation | seed-0 package | `900000..900099` | never selection/tuning | SEPARATE_AUTH_ONLY | SEALED | report once; no tuning or extra seeds |

## Tracker rules

- `TODO_NEXT_AUTH` means local implementation may begin only after the user
  authorizes the next milestone.
- `BLOCKED_NO_REMOTE_AUTH` means do not start the server, connect over SSH,
  train, collect, or run ManiSkill in the current task.
- A failed run is retained with its exact identity and evidence; its row is not
  overwritten.
- Any policy/config revision after development outcomes requires a new commit
  and a new untouched development manifest.
- Final seed access remains `SEALED` until every gate in the main plan and a
  separate explicit authorization are both satisfied.
