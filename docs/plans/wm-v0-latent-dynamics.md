# WM-v0 — Action-conditioned latent dynamics plan

## Objective

Add the smallest honest action-conditioned world-model path above the frozen
LatentGuard-VLA release. Given the current ordered RGB observations, current
proprioception, and a candidate action chunk, the model predicts short-horizon
future visual features, task progress, selected task events, terminal success,
and an uncertainty proxy. The experiment compares that model against the
accepted direct structured verifier and an outcome-only model with the same
world-model backbone.

## Starting point and preservation boundary

Development starts from `portfolio-v1` / main commit
`6a67c41ede76f4e783c1ef40f440fa2f353cdc33`. The release registries, accepted
M2C--M4D reports, checkpoints, selectors, and public claims are read-only. New
WM-v0 reports live outside the frozen release registry and cannot alter the
meaning of an accepted result.

Tracked source remains local-authoritative. A remote simulator or GPU may run
only an exact pushed revision. Raw RGB, state archives, feature caches,
checkpoints, and predictions remain outside Git; only compact sanitized audit,
manifest, and evaluation summaries may be committed.

## Work plan

1. Audit accepted verifier inputs, replay/runtime capabilities, visual data,
   future supervision, candidate provenance, and LangMani readiness.
2. Add a simulator-independent sample contract, strict serialization, manifest,
   group-isolated split validation, and an explicit formal-training gate.
3. Add a generic exact-restore counterfactual collector and a PickCube adapter
   that reuses the accepted M4C restore, execution, and state-preserving render
   contracts.
4. Reuse the frozen official ResNet-18 feature contract through a unified
   observation-encoder interface. Train only on cached features.
5. Add a compact masked temporal model, outcome-only ablation, losses,
   checkpoint/resume-capable training, and deterministic evaluation.
6. Compare identical candidate groups under in-distribution, held-out-scene,
   and held-out-policy-source settings. Report policy-generated and synthetic
   corruption candidates separately.
7. Permit a minimal closed-loop probe only after the offline gate passes. The
   primary action remains the default; any intervention additionally requires
   a risk threshold, score margin, and low uncertainty.

## Formal data gate

Formal training is authorized only when a strictly reloaded manifest proves all
of the following:

- at least 1,000 valid anchor-candidate samples;
- every sample contains current observations, an action, and at least one real
  post-execution simulator observation;
- source-trajectory groups are disjoint across train, validation, and test;
- terminal success and failure both occur; and
- at least one observed nonterminal progress or event target is available.

Unknown event channels are represented by an explicit target mask, never by a
fabricated negative. If the gate fails, this milestone is limited to audit,
collector validation, a small real collection, dataset construction, and a
training/evaluation smoke run explicitly labelled non-formal.

## Exclusions

- no end-to-end VLA, language-conditioned policy, planner, or large-scale
  closed-loop benchmark;
- no tuning from internal-test, external, or closed-loop outcomes;
- no privileged simulator state as a learned model input;
- no replacement or reinterpretation of accepted release results;
- no synthetic future frames presented as simulator supervision; and
- no server shutdown unless explicitly requested in the current task.
