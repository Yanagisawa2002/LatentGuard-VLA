# LangMani compatibility matrix

The machine-readable source of truth is
[`compatibility-matrix.json`](compatibility-matrix.json). It binds the clean
LangMani revision `b0fd9115496d3cc3d700c4eac7fb9f3f3ad401fd`, exact tracked-file
Git blob identities, and accepted compact-report SHA-256 identities. No local
checkout path participates.

| Status | Count |
| --- | ---: |
| Compatible | 11 |
| Adaptable | 7 |
| Blocked | 5 |
| Unknown | 0 |

| ID | Topic | Status | M6B gate | Required change |
| --- | --- | --- | --- | --- |
| LM-C01 | Repository identity | compatible | passed | Reconfirm exact clean SHA. |
| LM-C02 | Dependency versions | blocked | blocked | Validate a separate-process JSON bridge. |
| LM-C03 | Simulator version | compatible | passed | Run only a bounded version/import preflight. |
| LM-C04 | Control mode | compatible | passed | Bind active action-space digest. |
| LM-C05 | Task identity | compatible | passed | Freeze one canonical task. |
| LM-C06 | Observation schema | adaptable | blocked | Freeze a non-privileged verifier allowlist. |
| LM-C07 | Action schema | adaptable | passed | Export the selected chunk/horizon and mask. |
| LM-C08 | Action bounds | compatible | passed | Revalidate exact active bounds. |
| LM-C09 | Normalization | adaptable | passed | Use saved LangMani processors and fingerprints. |
| LM-C10 | Projection | compatible | passed | Preserve both raw and projected evidence. |
| LM-C11 | Policy binding | adaptable | blocked | Export one exact controller entry. |
| LM-C12 | Checkpoint identity | adaptable | blocked | Freeze one task/checkpoint before outcomes. |
| LM-C13 | State restoration | adaptable | blocked | Bind semantic reset plus state archive. |
| LM-C14 | Policy-state restoration | blocked | blocked | Serialize history or use a fresh initial boundary. |
| LM-C15 | Replay | blocked | blocked | Use initial-state-only or validate intermediate paired execution. |
| LM-C16 | Continuation | blocked | blocked | Select one pre-outcome semantic. |
| LM-C17 | Success | compatible | passed | Preserve official evidence before binary projection. |
| LM-C18 | Termination | compatible | passed | Map every status losslessly. |
| LM-C19 | Candidate generation | blocked | blocked | Freeze one blind, executable pool source. |
| LM-C20 | Visual observation | adaptable | passed | Use policy RGB; make no M4A renderer claim. |
| LM-C21 | Deterministic seed | compatible | passed | Bind proposal seed with the pool. |
| LM-C22 | Training/evaluation separation | compatible | passed | Keep the smoke no-training and outcome-blind. |
| LM-C23 | Remote artifacts | compatible | passed | Reconfirm preservation before M6B. |

The matrix's overall status is **blocked** because required gates remain
unresolved. “Adaptable” means the current behavior can cross the new typed
boundary after a named adapter/export step; it does not mean that step has run.
