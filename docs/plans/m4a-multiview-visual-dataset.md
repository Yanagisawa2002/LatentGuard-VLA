# M4A Multi-View Visual Action-Verifier Dataset

## Scope

M4A adds content-bound three-view RGB observations to the accepted PickCube
structured-state verifier datasets. It generates and validates data only. It
does not train or select a model, fine-tune an encoder, use a VLM/LLM or
LangMani, alter accepted M3A/M3C outcomes, execute video or depth inputs, or use
more than one GPU.

## Initial repository state

- Starting branch: `codex/m3c-blind-candidate-selection`.
- Starting, upstream, and GitHub commit:
  `ef78a78cb8fecf44b9705d326949f23208f3de4f`.
- Starting worktree: clean.
- M4A branch: `codex/m4a-multiview-visual-dataset`.
- Local repository is authoritative for all tracked code, configuration,
  tests, documentation, and compact reports.

## Accepted source identities

The development dataset must reload the complete accepted M3A artifacts and
match all of the following identities:

- verifier dataset:
  `sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`;
- state-indexed archive:
  `sha256:cc48c42f1c8395d348f968be72102b857eb2994702c6ca85bf8e6239f2fb36d5`;
- anchor manifest:
  `sha256:c15d4fdaf36994f5cd58304484110407e8f281057983056aa5a7025fad363e9e`;
- evidence dataset:
  `sha256:d077dcf31012d32f4b4833a0622b58d55a43d5a96a1f81b4eea045da0f747995`;
- trajectory split:
  `sha256:173ca40a072a1977cc65dbcccedb4684ae573e97f3984c176e3f91bba09f940a`.
- accepted full-target report file:
  `sha256:f3d747daffc8dfe3528350de4e193bc2e6d2a2954fe1d4476c3a2797c317c06f`.

The external dataset must reload the complete M3C pool, blind manifest, bound
result, selected/remainder evaluation datasets, source archive, and anchor
manifest. Compact reports alone are insufficient. Its accepted identities are:

- execution SHA:
  `a632a702c709edb1fc21e702c83e30964652ff79`;
- source set:
  `sha256:db1f3ece9576850c09a9f95b8c4aee483d11d71ebbbe0772b695db2d60321123`;
- candidate pool:
  `sha256:e2982acd30297fb81f000ea53081afdb4e2ca9ad5dc2f16c6aa71d199c8ef1b7`;
- blind manifest semantic and envelope:
  `sha256:c5aa3798d0d1797a1714f8de6ad76954f6aa1b31835a3aae8d0abbd7e52dcbcc`
  and
  `sha256:55e077a9818fc13b4a966081c7cbd57d4fb86bdcea447053e8349c62a9f6a37f`;
- full outcome and replay evidence:
  `sha256:06ac372236586ed7e4dcf3af8fc8d09e05e063f24cdaa687ee6bcdcd8e51d05c`
  and
  `sha256:c44bb6631243c78b0c8d7db8f573454613ee04f919293c0b9b37d561502cbdd4`.
- content-bound selection result:
  `sha256:e3cee839f5df10dd47578f214d02e0aee089a65bed37dad290f6a554b4aceaae`.

The accepted PickCube compatibility identity is
`sha256:05a560fd89e0989b6d94017c6456efc535de4a0bf87ededaf468f5e2d9ef71f7`.

## Source artifact availability

Only compact sanitized M3A/M3C reports are present in the local checkout. The
authoritative state archives, M3A verifier bundle, M3C candidate pool, and
selected/remainder replay datasets are not local and must be located on
persistent remote storage before rendering. If an accepted source artifact is
unavailable, M4A stops and preserves a compact diagnostic while the exact
existing artifact is located or restored from persistent storage. M4A must not
regenerate the accepted M3A or M3C physical outcomes, substitute smoke data, or
present a changed identity under an old name.

## Visual contract assumptions

- Observation boundary is reset, exact restore, full state verification,
  restored-boundary verifier/task capture, render, and full post-render
  verification with no physics step or candidate action.
- The complete state comparison covers 70 components at the fixed `1e-6`
  tolerance. The 38-component verifier vector and restored task projection must
  remain exact.
- `PickCubeMultiViewRigV1` has fixed `front_oblique`, `overhead`, and
  `side_oblique` world cameras, each producing RGB uint8 `[224, 224, 3]`.
- Each declared vertical FOV must reproduce the frozen pinhole intrinsics at
  the fixed `1e-9` absolute projection tolerance. Render seeds use only
  `sha256_anchor_domain_base_seed_v1`, which is bound through configuration,
  jobs, inventories, and CLI planning.
- Visual samples use the fixed task ID `maniskill/PickCube-v1` and exact
  canonical text `Pick up the cube and place it at the goal.`; M4A does not
  infer, translate, or otherwise vary task language.
- NPY bytes are authoritative and lossless. Pixel bytes and complete NPY files
  have separate exact SHA-256 inventories.
- Repeated and fresh-environment rendering must first pass byte equality. Pixel
  drift is a hard discovery blocker pending a narrow reviewed rule.
- The probe report content-binds the exact archive, episode, trajectory, reset
  seed, state index, and state/verifier digests used for discovery. Calibration
  comparison uses only the observed intrinsics/extrinsics dtypes recorded in
  that report; no dynamic dtype or numeric fallback is allowed.
- Every packet content-binds the exact 38-component verifier result, unchanged
  elapsed-step evidence, successful environment close, and complete per-view
  physical-state audit needed to reconstruct compact integrity reports.
- Runtime paths, hosts, PIDs, and timestamps are operational only and do not
  enter semantic identities.

## Candidate and packet count semantics

Images are stored once per anchor, domain, and view. A candidate binding points
to the three allowed domain packets for its anchor. A single-packet
`VisualActionVerifierSampleV1` is the expanded future model example and is
derived without copying image or action arrays.

- M3A: 360 anchors, 1,080 packets, 3,240 images, 3,240 candidate bindings,
  and 9,720 expanded single-packet examples.
- M3C external: 360 anchors, 1,080 packets, 3,240 images, 2,880 candidate
  bindings, and 8,640 expanded single-packet examples.

This explicitly resolves the difference between physical candidate count and
candidate-by-domain association count.

## Local implementation and validation gates

1. Implement simulator-independent immutable visual models, identities,
   configuration, packet/dataset binding, leakage checks, safe serialization,
   reporting, and resumable rendering state.
2. Implement lazy ManiSkill integration and a dependency-injected fake runtime;
   ordinary tests must not import ManiSkill, SAPIEN, CUDA, or Vulkan.
3. Add all five M4A commands, strict camera/domain configuration, and bounded
   fake rendering/reload/resume tests.
4. Atomically bind formal probe and render output roots to a sanitized
   operational manifest covering clean Git identity with tracked and untracked
   drift rejected, the current one-RTX-5090 runtime, fixed seed, the canonical
   complete source identity shared with the job inventory, and ordered selected
   packets. Keep this manifest outside packet/dataset identities.
5. Persist a content-bound resume report only after a real complete zero-work
   resume. Let full-target validation optionally publish a fixed, strictly
   reloaded compact report directory covering frozen camera/domain summaries,
   split-by-domain counts, exact image-inventory digests, determinism, fixed
   physical/verifier state error statistics, leakage, external training
   prohibition, and still-bound resume evidence. Partial validation cannot
   publish acceptance reports.
6. Run `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy src`,
   all five CLI help paths, bounded fake M3A/M3C runs, independent reload,
   resume, and `git diff --check`.
7. Review and push the discovery implementation as
   `feat(m4a): add multi-view visual verifier dataset pipeline`.

## Remote discovery and freeze gates

Remote work begins only after the discovery commit is pushed and an exact
one-RTX-5090 checkout is synchronized. The
authoritative execution schedule is exactly Phase A, Phase B, and Phase C under
the cost-aware amendment below; it replaces the earlier expanded gate list.
Before Phase A, a fail-fast preflight validates the persistent source artifacts
and identities, output-root state, exact Git SHA, camera/domain digests, disk,
renderer initialization, and one complete packet. Preflight does not regenerate
accepted physical outcomes.

Phase A contains the single bounded compatibility probe, including its
three-view render, exact repeated/fresh-environment determinism samples, and all
physical-integrity checks. There is no unconditional post-freeze probe rerun.
If a newly discovered material defect requires a local code or configuration
change, the phase stops; the fix is validated, committed, pushed, synchronized
to a new exact SHA, and recorded as the material reason for any new run.

Formal probe output contains exactly `visual-compatibility-report.json` and
`run-manifest.json`. Formal render roots publish the run manifest together with
their ledger and complete job inventory. Resume validates the current Git,
runtime, source, seed, inventory, and selected-packet binding before it may
recover staging or mutate a ledger; a complete zero-work pass then publishes
its immutable bound resume report. Dry runs produce neither manifest nor output
root.

No later gate may run after an earlier failure. Visual configuration is never
edited remotely.

## Cost-aware execution amendment

This amendment changes execution strategy only. It does not reduce the visual,
physical-integrity, evidence, serialization, reload, leakage, or resume
contracts above. Accepted M3A/M3C states, anchors, candidates, evidence,
outcomes, splits, blind selection, and pool bindings are reused; M4A renders
anchor-level observations once and never regenerates physical outcomes.

Remote execution is limited to three phases from one clean pushed SHA per
accepted implementation revision:

1. **Discovery:** one archived state, all three cameras, three repeated renders
   in one initialized environment, two separately initialized fresh-environment
   renders, and all complete state/verifier/task integrity checks. Pixel drift
   stops the phase without remote shader, driver, tolerance, rig, or lighting
   experimentation.
2. **Smoke:** after the visual configuration is frozen and pushed, exactly six
   M3A trajectories and six disjoint M3C trajectories across all assigned
   domains and three views, followed by publication, reload, leakage,
   integrity, and zero-work resume validation. A rerun requires a material
   locally fixed, tested, committed, and pushed defect plus a new run ID.
3. **Full:** exactly one full M3A development render, one full M3C external
   render, one independent combined validation, and one strict zero-work resume
   validation. Cosmetic preference cannot trigger rerendering.

Each render worker may reuse its initialized environment, but every packet must
independently reset with its bound seed, restore its bound anchor state, verify
the complete state, extract verifier state, configure exactly one domain,
render all three cameras without stepping physics, and repeat the complete
state/verifier/task checks afterward. Cross-packet drift fails closed; smoke
failure falls back to fresh environment creation per packet without weakening
the contract. Images are stored once per anchor/domain packet and referenced by
candidate records; no candidate-level rerendering or extra random variants are
allowed.

Compact operational reports record environment initialization count, packet and
image counts, average and nearest-rank percentile packet render time, total
rendering and validation time, resume-reused packets, peak GPU memory when
available, and reliably measured server-active execution duration. These facts
are reporting-only and never enter semantic identities. Paid-server preflight
must check source digests, output resumability, exact Git SHA, configuration
digests, disk, renderer initialization, and one complete packet. A blocker
flushes transactional state, preserves a compact diagnostic, retrieves it, and
stops only the related process. The server remains online unless the user
separately and explicitly requests shutdown.

M4A performs no model training, backbone or teacher feature extraction,
augmentation job, normalization fitting, M4B smoke, or architecture selection,
and it does not begin M4B automatically.

### Future project-wide model screening

Future model milestones screen each proposed architecture once with seed `0`,
full training data, validation-only selection, ordinary early stopping, and
checkpoint/resume validation. Only the best validation model, strongest direct
baseline, and one technically essential ablation may advance to final seeds
`0, 1, 2`. Add seeds `3` and `4` only when the top validation gap is below
`0.02`, three-seed standard deviation exceeds `0.03`, rankings change across
seeds, a seed collapses, or a publication-level statistical claim requires it.
Poor external-test performance must never trigger extra seeds or test tuning.
This policy reduces repeated training without reducing required architectures,
ablations, external evaluation, or technical scope.

### Future M4B efficiency requirements

These requirements are documentation for a later milestone; M4A neither
implements nor runs them. The first visual baseline should prefer a frozen
visual backbone. Its frozen image embeddings and the privileged teacher logits
are each cached once and reused across permitted seeds and fusion heads, while
one raw-RGB end-to-end inference validation is retained. Initial training uses
one seed, and only the selected models advance to seeds `0, 1, 2` under the
project-wide escalation rule. The frozen M3C external visual evaluation runs
exactly once. M4B must not automatically execute
`architecture_count x 5 seeds`.

Required future comparisons remain visual-only, RGB+action fusion,
teacher-distilled RGB+action, the strongest structured-state teacher,
canonical/shifted-domain evaluation, and blind candidate selection after model
freeze. None of those quality components is removed.

## Remote outputs and closeout

Raw NPY images, state archives, action arrays, datasets, caches, and inspection
images remain outside Git. Only compact sanitized compatibility, identity,
count, determinism, integrity, domain, leakage, training-prohibition, resume,
manifest-summary, and human-review reports are retrieved under `reports/m4a/`.
After full acceptance, update this plan, rerun local checks, commit
`docs(m4a): record multi-view visual dataset acceptance`, push, and verify clean
local/upstream/remote SHA equality. Do not shut down the server automatically;
leave it online unless the user separately and explicitly requests shutdown.

The final completion report records observed facts only and must state:

- which discovery, smoke, and full phases actually ran;
- every rerun and its exact material reason;
- packet and image counts;
- physical-integrity statistics and the pixel-determinism result;
- the M3A development and M3C external visual dataset digests;
- the cross-dataset leakage result and strict zero-work resume result;
- total reliably measured remote execution time;
- confirmation that no model training or feature extraction occurred;
- paid-server running/shutdown status and whether the user requested a change;
- branch and commits;
- local, upstream, and GitHub SHA equality; and
- clean working-tree status.

## Remote execution attempt ledger

- Phase A attempt 1 used run ID
  `20260718T041652Z_m4a-phase-a-discovery_6ce55c4_seed271828` from clean pushed
  SHA `6ce55c48241e8eae70c6691fc677779fd7d36779`. The accepted M3A report digest,
  dependency versions, single-GPU identity, 238 GB available disk, clean Git
  state, and absent output root passed preflight.
- The command failed after 30 seconds during renderer preparation, before an
  output root, packet, or image was created. ManiSkill 3.0.1
  `set_shader_pack` requires the selected `ShaderConfig` object from
  `PREBUILT_SHADER_CONFIGS`; the integration had passed its bound string key.
  No pixel-determinism result was produced, so this was not a completed visual
  compatibility probe.
- The raw log and wrapper status were retrieved into ignored local temporary
  storage and content-bound by the sanitized compact failure summary under
  `reports/m4a/`. No training or feature extraction began. The paid server was
  shut down immediately after evidence retrieval.
- This platform defect is the material reason authorizing one new Phase A run
  ID after a narrow local fix passes the complete suite, is committed and
  pushed, and the server is restarted and synchronized to that exact new SHA.
- Phase A attempt 2 used run ID
  `20260718T072457Z_m4a-phase-a-discovery_be5bb0a_seed271828` from clean pushed
  SHA `be5bb0ac2127085ed64aa77993a52d747c9ac9ad`. It passed the shader-selection
  boundary fixed after attempt 1, then failed after 30 seconds at the first
  `RenderCamera.take_picture()` call, before an output root, packet, or image
  was created.
- The pinned ManiSkill 3.0.1 GPU path requires a render camera group for each
  late-added `RenderCamera`. `ManiSkillScene.add_camera` returns the wrapper
  without that group after environment setup, so the wrapper attempted to call
  `take_picture` on `None`. The raw log and wrapper status were retrieved into
  ignored temporary storage and content-bound by the second compact sanitized
  failure summary. No training or feature extraction began.
- The server was deliberately retained online under the user's updated
  operational instruction. This material platform defect authorizes one new
  Phase A run only after a version-bound, fail-closed camera-group fix passes
  the complete local suite, is committed, pushed, and synchronized exactly.
- Phase A attempt 3 used run ID
  `20260718T080331Z_m4a-phase-a-discovery_d30e1f6_seed271828` from clean pushed
  SHA `d30e1f6a13cbd5a6893d53aaddb6cd9989e1ab49`. It passed all three GPU
  camera-group creations and assignments, then failed after 31 seconds during
  render-only lighting setup, before an output root, packet, or image was
  created.
- SAPIEN 3.0 forbids scene modification after its batched render system is
  created. The candidate applied lighting after the camera-group initializer's
  first `scene.update_render`, so the lighting call was rejected. The raw log
  and status were retrieved and content-bound by the third compact sanitized
  failure summary. No training or feature extraction began, and the GPU
  returned to zero allocated memory.
- The server remains online under the user's instruction. This material API
  ordering defect authorizes one new Phase A run after lighting is moved before
  batched render-system initialization, the required order is regression-tested,
  and the complete local validation, commit, push, and exact synchronization
  gates pass again.

## Current status

The cost-aware implementation, shader-object fix, and version-bound GPU
camera-group initialization were pushed and synchronized exactly. The third
bounded Phase A command proved all three groups initialize, then exposed the
SAPIEN ordering defect recorded above. A narrow local candidate now applies
render-only lighting after all cameras are added but before the first
`scene.update_render` creates the batched render system. The CPU fake enforces
the same mutation prohibition and exact event order. Targeted tests, Ruff, and
mypy pass. Complete local validation also passed with 1,429 tests and the three
expected Windows directory-symlink privilege skips, full-repository Ruff,
format, mypy, five-command help smoke, and diff checks; publication is pending.
The paid server remains online. No trusted visual probe, raw visual dataset,
model training, feature extraction, VLM, LangMani, video, or M4A smoke/full
render has completed. The next gates are commit/push, exact remote
synchronization, and one material-reason Phase A rerun.
