# M3C Blind Verifier-Guided Candidate Selection Plan

## Objective and fixed authority

M3C tests whether frozen M3B verifier ensembles can select successful unseen
PickCube action chunks before any candidate outcome is known. Development and
CPU validation are local. Simulator collection, optional exact checkpoint
reconstruction, inference profiling, and replay run only on one remote RTX 5090
after the implementation revision is committed and pushed.

The starting and accepted M3B result revision is
`66eaee0f69a8f5dfe4bc7a53a777e08b5ce51b88`. Runtime loading must derive and
validate the model, preprocessing, checkpoint, calibration, threshold,
action-magnitude, dataset, and split identities from the checked-in M3B result
records. The accepted M3A dataset digest is
`sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`
and its split digest is
`sha256:173ca40a072a1977cc65dbcccedb4684ae573e97f3984c176e3f91bba09f940a`.
M3C does not rerun architecture selection or change M3B architecture and
training hyperparameters.

## Assumptions and evidence boundaries

- The committed compact M3B benchmark summary, not a runtime directory, is the
  authority for each calibration and threshold outer strict-report digest.
  Runtime self-consistency cannot replace that recorded identity.
- The accepted M3A dataset, split, preprocessing, action contract, M2C
  compatibility identity, and authorized `1e-6` full-state restoration
  tolerance remain unchanged.
- Stage A capability isolation means the process is not given an evidence,
  replay, or outcome path. It does not claim that every unrelated host
  filesystem location is empty.
- Stage A receives only the strict blinded projection of a candidate pool. The
  full pool, its distribution/corruption/severity/provenance fields, replay
  evidence, and outcomes are unavailable to the selection command and are
  first loaded by Stage B/C.
- Missing checkpoints may be reconstructed only for the already authorized
  three families and five seeds. Reconstruction must emit new explicit
  identities; it is not architecture selection and cannot read M3C outcomes.
- Remote execution is sequential on one RTX 5090. No VLM/LLM, image encoder,
  LangMani, distributed job, multi-GPU process, outcome-driven training, or
  new-architecture training is in scope. Exact fixed-M3B checkpoint
  reconstruction is the only missing-artifact exception.

## Blind execution boundary

Candidate-pool generation and selection are separate immutable phases. Stage A
receives only content-bound state and candidate-action inputs, frozen selector
and verifier-bundle identities, and no evidence, replay, or outcome path. It
finalizes a manifest with `outcomes_available_during_selection=false` before
Stage B may create any counterfactual outcome. Stage B replays the union of
non-abstained blind selections. Only after that completes may Stage C replay the
remaining pool, reveal complete-pool outcomes, construct the analysis-only
oracle, and compute ranking and regret. Stage B/C loading must prove that Stage
A rankings, selected IDs, and abstentions are unchanged.

The Stage A manifest binds exactly eleven selectors: deterministic random,
frozen action magnitude, three learned five-seed ensembles, and six temporal
abstention/coverage variants. Only the action-only, state+action MLP, and
temporal families are verifier bundles. The oracle is outside Stage A and
cannot exist until the complete Stage C outcome inventory is available.

Stage C cannot start a simulator session until the selected-union run has been
strictly reloaded as complete, strong, error-free, and bound to the same
envelope, source, pool, selector configuration, seed, and replay dataset. Final
binding requires real persisted phase time order:
`selection < selected start <= selected finish < remainder start <= remainder
finish`. The outcome time is the persisted remainder finish, not a generated
substitute.

The original source chunk is never selectable. It remains only the fixed
continuation and mandatory remaining-trajectory baseline gate. All selectors
consume the same eight candidates per anchor. Abstention reduces executed
coverage and is neither success nor task failure.

## Implementation scope

1. Add a simulator-independent `latentguard.selection` package for strict
   configuration, frozen verifier bundles, typed inference, candidate pools,
   selectors, blind manifests, result binding, metrics, ranking, inference
   performance, trajectory bootstrap, and safe serialization/reporting.
2. Reuse M3B model construction, preprocessing, trusted checkpoint inspection,
   calibration, thresholds, and frozen action-magnitude semantics. Checkpoint
   paths remain runtime-only and checkpoint tensors remain outside Git.
3. Reuse the M1 corruption registry and action-layout contract to generate four
   fixed in-distribution and four fixed shifted candidates without clipping,
   repair, or source mutation.
4. Reuse M3A source collection, state-indexed archives, six-anchor selection,
   baseline validation, 70-component runtime state comparison, and exact
   38-component verifier-state reproduction.
5. Reuse the existing M2B runner, ledger, resume/retry, evidence, replay trust,
   and M2C PickCube adapter. M3C adds orchestration and bindings, not a second
   replay ledger or simulator-native core model.
6. Add six strict CLIs for artifact preparation, pool construction, blind
   selection, selected-only replay, complete-pool replay, and evaluation.
   `select-action-candidates` exposes no outcome/evidence/replay argument.
7. Add checked-in smoke/full seed-range, candidate-pool, selector, profiling,
   and bootstrap configurations. Any post-smoke change follows local edit,
   validation, commit, push, exact remote sync, and a new untouched full range.
8. Extend the repository rules and user documentation without rewriting
   unrelated policy.
9. Bind runtime calibration and threshold reports to the outer digests recorded
   in the committed M3B compact summary, then recompute every seed's complete
   ordered validation logit/label digest before fitting ensemble policies.
10. Persist a versioned selector-configuration report and separate CPU/GPU
    latency reports with per-candidate/group p50/p95/p99, throughput, peak
    memory, model loading, and joint-versus-temporal cost differences. Raw
    predictions remain outside those reports.
11. Run a fixed outcome-free three-ensemble forward smoke on CPU and GPU before
    new source collection. Publish Stage A atomically with its selector
    configuration and internal latency report, while keeping recoverable
    external latency copies outside the immutable root.
12. Separate the path/host/time-independent semantic replay evidence digest
    from the exact archive-audit digest. Emit independent zero-work replay
    resume proofs, a combined final resume summary, and a byte-bound Markdown
    review without raw predictions, states, actions, or evidence payloads.
13. Fit balanced-accuracy and failure-recall abstention thresholds on the full
    corrupted M3B validation candidate inventory, while fitting coverage
    thresholds on validation per-group minima. This remains validation-only and
    handles the observed one-class group-minimum labels without consulting M3C.
14. Bind candidate generation to the trusted compatibility/action-layout
    action-contract digest. Keep the source solver action dtype and byte contract
    independently bound by the anchor manifest; the environment runtime contract
    and source serialization contract are intentionally distinct identities.
15. Bound ManiSkill native-runtime lifetime during production source collection:
    launch the same collection CLI sequentially for one seed and one attempt per
    short-lived subprocess, strictly reload and cross-check both exact archives,
    and merge accepted episodes in deterministic seed order. Run the requested
    outer fresh-state audit in a separate short-lived process per trajectory.
    This is failure containment only, not parallel or distributed collection;
    native signals, malformed worker output, or partial archives fail closed
    before any final archive or summary is published.

## Candidate and evaluation contract

Exactly eight unlabeled candidates are generated per accepted anchor: four use
reviewed M3A definitions and four use explicitly shifted but contract-valid
parameters absent from M3A. Deterministic IDs bind source, anchor, configuration,
resolved parameters, transformed bytes, and generation seed. Source chunks are
rejected by byte identity as well as by semantic role. Static action-contract,
finite-value, dtype, shape, window, unchanged-continuation, inventory, and
content-digest checks run before a pool is accepted.

Smoke uses six successful source trajectories from a dedicated reset-seed range.
Full evaluation uses 60 successful trajectories from a disjoint untouched range.
Both ranges must be disjoint from all M3A reset seeds, trajectory IDs, complete
state-tree digests, and split-group IDs. Smoke mechanics cannot authorize a
claim. Once any full-range outcome exists, candidate/selector/bundle/threshold
configuration is immutable and the seed range cannot be reused after change.

The fixed comparison includes deterministic random, frozen action magnitude,
five-seed action-only, joint-MLP, and temporal ensembles. Oracle is constructed
only after complete-pool outcomes. The temporal ensemble remains primary by the
predeclared M3B validation result; joint MLP remains the efficiency challenger.

Full pool construction reloads the smoke pool and checks reset seeds,
trajectory IDs, split-group IDs, and every complete state-tree digest. Smoke
and full use separate fixed seed windows and output roots; smoke data cannot be
promoted into the full benchmark.

## Implementation and execution status

The M3C implementation and CLI are present locally, including the six command
entry points, capability-level blinded Stage A input, strict manifest/report
binding, complementary replay gates, selector/metric contracts, predeclared
target interpretation, pre-collection inference forward gate, atomic Stage A
publication, semantic/archive replay digest separation, compact resume proofs,
and CPU-only fake protocol coverage. The current native-containment correction
adds strict single-seed collection-worker and fresh-audit-worker coverage. Final
combined-tree validation on the repository CPython 3.12.10 environment passed
`1215` tests with `3` Windows symlink-privilege-only skips. `ruff check .`,
`ruff format --check .`, `mypy src` over 120 source files, all six M3C CLI help
smokes, and `git diff --check` also passed before the correction commit.

The first remote smoke attempt at implementation revision
`93dd0e2a7e989cecc47bf72de1a09fd68d8b9d8d` passed all M3A/M3B artifact gates,
CPU/GPU pre-collection forward smokes, six-source collection, and 36 independent
anchor baselines. Candidate construction then failed closed before publishing a
pool because the CLI compared the trusted environment action-contract digest
with the distinct source-solver serialization contract. No Stage A manifest,
replay evidence, or candidate outcome was created. The local correction keeps
the trusted `sha256:028d8c2cbb10e867d709f1c5d4c31e07ef1e084f1a8ad8370f96185ba892f0eb`
runtime binding and preserves the source `<f8` contract independently in the
anchor manifest. At that revision no M3C full benchmark or result claim had yet
been made.

The corrected smoke at revision
`70a608ff062210ee4c5541eec4f78939f1283fb8` completed the full blind mechanics
protocol: six accepted sources, 36 anchors, 288 candidates, all eleven Stage A
selectors, 122 selected-union replays, 166 complementary replays, complete
strong simulator evidence, and zero-work resume. These are smoke mechanics only
and do not support a research claim. The first two attempts to collect the fixed
60-source full range then terminated inside native code with `SIGSEGV` before
collection returned. Both failure records and the second faulthandler log are
preserved; the full output directory remained empty and no full candidate pool,
Stage A manifest, replay evidence, or outcome was created. The local correction
therefore changes only process containment around the same source collector and
fresh-restoration audit. It does not change seeds, source policy, simulator
contract, candidate configuration, selectors, labels, or evaluation semantics.
The full benchmark subsequently restarted under exact pushed revision
`a632a702c709edb1fc21e702c83e30964652ff79`.

## Full remote result

Run `20260716T152012Z_m3c-full_a632a70_seed271828` completed on one RTX 5090.
All 15 accepted M3B checkpoints were present and digest-validated; no checkpoint
reconstruction, new training, architecture selection, or threshold fitting was
performed. The run accepted 60/60 source trajectories in 60 attempts, containing
4,479 source actions and 4,539 T+1 states. It built 360 anchors with 360 successful
independent baselines, then 360 eight-candidate groups (2,880 candidates). The
smoke and full pools passed the reset-seed, trajectory-ID, split-group-ID, and
complete-state-digest disjointness gate.

The fixed ID-like candidates were arm Gaussian noise at 0.005, arm bias at 0.01,
arm Gaussian noise at 0.05, and an all-dimension hold beginning at step 1. The
shifted candidates were arm Gaussian noise at 0.02, arm bias at 0.05, an
all-dimension hold over `[4, 12)`, and an arm permutation over `[2, 10)`. The
candidate-pool identity is
`sha256:e2982acd30297fb81f000ea53081afdb4e2ca9ad5dc2f16c6aa71d199c8ef1b7`;
the source-set identity is
`sha256:db1f3ece9576850c09a9f95b8c4aee483d11d71ebbbe0772b695db2d60321123`.

Stage A finalized 3,960 decisions from 11 selectors at
`2026-07-16T17:49:21.544698Z`, before any outcome was available. Its semantic
manifest digest is
`sha256:c5aa3798d0d1797a1714f8de6ad76954f6aa1b31835a3aae8d0abbd7e52dcbcc`;
`outcomes_available_during_selection` is false and `outcome_input_paths` is
empty. Persisted chronology proves selection preceded selected replay, which
preceded remainder replay. Stage B replayed the 1,168-candidate selected union;
Stage C replayed the exact 1,712-candidate complement. Both phases initially
encountered native `SIGSEGV` exits after atomically persisted progress. Their
failure manifests and logs were content-hashed, and strict same-SHA resumes
completed without invalid, indeterminate, retried, or execution-error outcomes.

The complete join contains 2,880 strong simulator-verified outcomes and zero
unexplained execution errors. Its semantic replay digest is
`sha256:c44bb6631243c78b0c8d7db8f573454613ee04f919293c0b9b37d561502cbdd4`;
the separate archive-audit digest is
`sha256:048f0ffc784bcf763aa6c89d61ffb84638dff6543f44e5c84a26fd5624fb40dd`.
Independent selected and remainder resumes each evaluated, recovered, and
retried zero attempts. The final strict evaluation resume bound both proofs and
reported `zero_duplicate_replay_work=true`.

| Selector | Coverage | Selected success | Task failure |
|---|---:|---:|---:|
| Deterministic random | 1.000000 | 0.888889 | 0.111111 |
| Frozen action magnitude | 1.000000 | 0.869444 | 0.130556 |
| Action-only ensemble | 1.000000 | 0.986111 | 0.013889 |
| State+Action MLP ensemble | 1.000000 | 0.994444 | 0.005556 |
| Temporal ensemble | 1.000000 | 0.983333 | 0.016667 |
| Oracle, analysis only | 1.000000 | 1.000000 | 0.000000 |

All 360 groups were solvable: 235 were all-success, 125 mixed, and none
all-failure. Temporal improved selected success over random by 0.094444; the
60-trajectory paired-bootstrap 95% interval was `[0.063889, 0.125000]`. Its
relative task-failure reduction was 85%, pairwise success-over-failure
concordance was 0.926502, top-1 success was 0.983333, and top-2 success was 1.0.
The validation-frozen approximately-70% policy executed 272/360 groups
(coverage 0.755556) with failure rate 0.011029. All predeclared quality targets
passed without full-outcome tuning.

The joint MLP point estimate was higher than temporal and its efficiency was
better. Temporal-minus-joint selected success was -0.011111 with 95% interval
`[-0.025000, 0.000000]`; temporal superiority is therefore not claimed. CPU
eight-candidate group p50 latency was 2.253 ms for joint versus 7.074 ms for
temporal. CUDA group p50 was 2.467 ms versus 4.958 ms, with peak allocated
memory 18,105,856 versus 18,610,688 bytes. Full outcomes did not change the
predeclared temporal-primary or joint-efficiency-challenger roles.

The collection summary contains 65 fresh-state audits, each comparing all 70
runtime-state components; their maximum absolute error was
`1.1920928955078125e-7`, below the fixed `1e-6` contract. The corresponding
38-component restored verifier vectors were exact. These are collection/fresh
audit statistics, not an aggregate maximum over all 2,880 replay attempts. The
strong replay evidence and archive digests separately bind the complete replay
gate. Twenty-six source/restored task snapshots differed only on `is_grasped`;
the restored pre-action verifier vector remained exact, and no value was
silently repaired or substituted.

The 24 reviewed files under
`reports/m3c/20260716T152012Z_m3c-full_a632a70_seed271828/` total 7,541,112
bytes. They include bundle summaries, precollection smokes, pool/build/collection
summaries, the outcome-free blind manifest, selector configuration, CPU/CUDA
latency, two phase-resume proofs, bound result, combined metrics, final resume,
human review, execution summary, and byte-exact retrieval manifest. Checkpoints,
arrays, raw states, full replay datasets/evidence, and raw prediction streams
remain outside Git.

Final local closeout reloaded all 17 strict reports, recomputed the complete
cross-artifact digest graph, checked all 24 retrieved/derived files against the
retrieval inventory, and independently recomputed the predeclared target and
bootstrap interpretations. `python -m pytest` passed 1,215 tests with the three
expected Windows symlink-privilege skips. `ruff check .`,
`ruff format --check .`, `mypy src`, all six M3C CLI help smokes, the
sanitization scan, and
`git diff --check` also passed. The compact-only local check does not reload the
external raw arrays or replay evidence payloads; that stronger gate was run on
the execution server by the final exact-SHA evaluation resume before retrieval.

## Ordered local gates

1. Strict direct-construction and parser tests for every identity/configuration.
2. Eight-candidate deterministic fixture generation with source exclusion and
   unchanged continuation.
3. Five-seed bundle/digest/calibration tests with fake checkpoint files.
4. CPU inference, stable candidate-ID tie breaking, ensemble averaging,
   abstention, and latency-report shape tests.
5. Blind Stage A tests proving outcome unavailability and manifest immutability.
6. Fake selected/full replay, strong-evidence binding, resume-with-zero-duplicate,
   and tampering tests using the existing runner boundary.
7. All-success, all-failure, mixed, solvable, ranking, regret, coverage, and
   trajectory-bootstrap tests.
8. Full fake end-to-end blind-protocol smoke and CLI dry runs.
9. `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy src`, all
   six CLI help/smoke paths, and `git diff --check`.

After review, commit exactly
`feat(m3c): add blind verifier-guided candidate selection`, push, verify the
clean local/upstream/GitHub SHA equality, then permit remote work.

## Ordered remote gates

1. Start one RTX 5090 and synchronize a clean checkout to the exact pushed SHA.
2. Validate the accepted M3A directory and all M3B preprocessing, calibration,
   threshold, and 15 checkpoint identities. If checkpoints are absent, rebuild
   only the three authorized five-seed families from exact accepted M3B
   configurations without M3C outcomes or architecture selection.
   Validate calibration/threshold outer report digests against the committed
   M3B compact summary and reproduce all ordered validation-prediction digests.
3. Pass CPU and GPU inference/profile smokes.
4. Collect six smoke sources; build exactly eight candidates per anchor;
   finalize the blind smoke manifest; replay selected candidates; replay the
   remaining pool; validate metrics, evidence, and zero-duplicate resume.
5. If smoke requires a configuration fix, preserve its evidence, fix locally,
   commit/push, synchronize a new exact SHA, and use a new full range.
6. Freeze the full candidate, selector, bundle, threshold, seed-range, profiling,
   and bootstrap identities before collection.
7. Collect 60 untouched successful sources and validate disjointness, baseline
   success, state restoration, verifier-state reproduction, and six anchors.
   Load the smoke pool and prove the four-axis smoke/full exclusion before pool
   acceptance.
8. Build all pools and finalize Stage A with no outcome dataset/path available.
9. Replay blind selections, then the complete remaining pools; preserve strong
   simulator evidence and resume without duplicate work. Require strict Stage B
   reload before any Stage C session and retain the real phase timestamps.
10. Evaluate all/solvable/mixed groups, ID-like/shifted subsets, ranking,
    abstention/coverage, inference latency/memory, and at least 2,000
    trajectory-level paired bootstrap samples.
11. Strictly reload all final bindings, retrieve only compact sanitized reports,
    and leave checkpoints, arrays, raw states, full predictions, and datasets
    outside Git.

## Result closeout and exclusions

After remote evaluation, add reviewed evidence under `reports/m3c/<run-id>/`,
update this plan with observed results and limitations, rerun every local gate,
commit exactly `docs(m3c): record blind candidate-selection results`, push, and
verify the clean local/upstream/GitHub SHA equality before shutting down the paid
server.

M3C excludes receding-horizon replanning, repeated policy calls, source-action
fallback, VLM/LLM/image/language encoders, LangMani, pretrained models, new
architectures, architecture reselection, outcome-driven tuning, multiple tasks
or robots, distributed/multi-GPU execution, real robots, and general safety
claims. A failed research target is preserved as a valid engineering result and
does not authorize tuning or reuse of full seeds.

The fixed source continuation is intentionally an experimental control: after
one 16-step selection the system receives no new observation, does not invoke
the policy or verifier again, and executes the archived continuation unchanged.
It can therefore influence late outcomes and cannot support a closed-loop or
receding-horizon claim. VLM/LangMani work remains deferred until this restricted
structured-state intervention demonstrates value under the blind evaluation.
