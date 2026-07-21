# WM-v0 D2 — Accepted Policy Binding and Terminalized Real-Candidate Pilot

## Baseline

- Base branch: `codex/wm-v0-d1-policy-dataset`
- Base commit: `cfc4d27e19d0205bddc0b94628152f209bd71293`
- D1 remains immutable: 120 synthetic counterfactual futures, 0 policy-generated
  candidates, 11 terminal successes, 0 terminal failures, and 109 nonterminal
  samples.

## Plan

1. Audit all accessible LatentGuard, LangMani, Git-history, release-manifest, and
   remote execution roots for controller checkpoints and their complete runtime
   dependencies.
2. Define a path-portable `PolicyPackage` and an audited registry that rejects
   missing files, byte drift, default processors, default normalization, and any
   observation/action/environment mismatch.
3. Permit policy smoke and candidate collection only if at least one registry
   entry is `ACCEPTED_COMPATIBLE`.
4. If the asset gate passes, run the bounded real-candidate pilot and evaluate
   diversity and terminalization. If it does not pass, stop with Result B and
   persist zero-sample reports rather than substitute synthetic actions.
5. Run local CPU validation, commit and push the milestone, then synchronize the
   remote execution checkout to the exact pushed revision for a compact,
   sanitized, no-simulator audit.

## Assumptions

- A matching action tensor dimension and control-mode label are insufficient to
  establish action ordering or task compatibility.
- Historical rollout evidence is accepted only for the exact environment and
  task contract it records.
- A registry declaration or checkpoint fingerprint is not an executable policy
  package without its checkpoint bytes, configuration, processors,
  normalization, and compatible runtime contract.

## Exclusions

- No formal WM-v0, verifier, outcome-only, or policy training.
- No candidate action perturbation or relabeling of D1 synthetic corruptions.
- No world-model action selection or closed-loop intervention.
- No modification of the PickCube environment to accommodate LangMani assets.
- No remote tracked-source edits and no server shutdown.

## Result

Result B was selected at the accepted-policy asset gate. The only complete
controller package is bound to a different LangMani task and observation
contract. Consequently policy smoke, provider construction, candidate diversity,
rollout collection, and terminalization were not authorized.
