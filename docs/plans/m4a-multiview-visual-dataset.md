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
persistent remote storage before rendering. If unavailable, they must be
regenerated from the exact accepted contracts and seeds; smoke data may not be
substituted and a changed identity may not be presented under an old name.

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

The prior paid server is shut down. Remote work begins only after the discovery
commit is pushed and an exact one-RTX-5090 checkout is synchronized. Gates are:

1. validate persistent source artifacts and exact identities;
2. renderer import smoke;
3. visual compatibility probe on one archived state;
4. one-state canonical three-view render;
5. repeated and fresh-environment byte determinism;
6. full state, verifier-state, and task-projection integrity;
7. six-trajectory M3A and disjoint six-trajectory M3C smokes;
8. independent smoke reload and zero-work resume;
9. local review and freeze commit for the rig/domains;
10. exact-SHA probe rerun with agreement;
11. full M3A rendering and validation;
12. full M3C evaluation-only rendering;
13. cross-dataset leakage, independent reload, and zero-work resume.

Formal probe output contains exactly `visual-compatibility-report.json` and
`run-manifest.json`. Formal render roots publish the run manifest together with
their ledger and complete job inventory. Resume validates the current Git,
runtime, source, seed, inventory, and selected-packet binding before it may
recover staging or mutate a ledger; a complete zero-work pass then publishes
its immutable bound resume report. Dry runs produce neither manifest nor output
root.

No later gate may run after an earlier failure. Visual configuration is never
edited remotely.

## Remote outputs and closeout

Raw NPY images, state archives, action arrays, datasets, caches, and inspection
images remain outside Git. Only compact sanitized compatibility, identity,
count, determinism, integrity, domain, leakage, training-prohibition, resume,
manifest-summary, and human-review reports are retrieved under `reports/m4a/`.
After full acceptance, update this plan, rerun local checks, commit
`docs(m4a): record multi-view visual dataset acceptance`, push, verify clean
local/upstream/remote SHA equality, and shut down the paid server.

## Current status

The local discovery candidate is implemented. Python 3.11 passed all 191 M4A
tests; the repository environment passed 1,409 tests with three Windows
directory-symlink tests skipped for missing host privilege. `ruff check .`,
`ruff format --check .`, `mypy src`, all five installed CLI help paths, and the
CPU-import boundary also pass. No visual compatibility probe, trusted render,
raw visual dataset, model training, VLM, LangMani, video, or remote M4A
execution has yet occurred. The paid server remains off; remote source gates
and the visual compatibility probe await power-on after this exact revision is
pushed and confirmed upstream.
