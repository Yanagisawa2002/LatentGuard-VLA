# M2C — ManiSkill PickCube reference adapter

## Objective

Add the first real simulator integration behind the M2B-Core protocols for the
version-locked ManiSkill `3.0.1` `PickCube-v1` task. The integration must probe
the installed runtime, record official Panda motion-planning trajectories,
independently replay every accepted source, safely archive exact state trees,
import accepted sources into M0, generate M1 proposals, and reuse M2A/M2B for
baseline-gated paired replay.

## Authoritative workflow

Tracked source, configuration, tests, and compact reports are changed only in
the local checkout. Remote execution may begin only from a clean pushed commit
whose full SHA is verified on the server. Runtime state archives, trajectories,
simulator caches, and logs remain outside Git. Any mismatch found remotely is
fixed locally, committed, pushed, and synchronized before a new run.

Remote authentication is fail-closed: automated execution requires dedicated
key-based `BatchMode=yes` authentication, with connection details supplied
outside Git. The remote checkout uses a separate repository-scoped, read-only
deploy key. Automated synchronization and execution do not use, store, or print
passwords or private keys.

## Fixed contract

- ManiSkill distribution: exactly `3.0.1`.
- Environment/task: `PickCube-v1` / `maniskill/PickCube-v1`.
- Robot: Panda (`panda`).
- Execution: one environment, GPU simulation, minimal state-free observation,
  `pd_joint_pos`, no rendering or video.
- Source: packaged official
  `mani_skill.examples.motionplanning.panda.solutions.pick_cube:solve`, invoked
  through a project-owned step recorder without vendoring or altering solver
  behavior; the remote probe must verify its installed source digest.
- State: reset-boundary `get_state_dict()` content, serialized as a strict
  structural JSON manifest plus non-pickle numeric NPY leaves. The archived
  source bytes remain digest-bound; simulator set/get readback must preserve
  the complete structure and every numeric component within the locally
  committed tolerance.
- Replay: complete source and transformed action sequences in independent fresh
  sessions restored from the same content-bound state reference.
- Evidence: exact-simulator/strong is only a permission ceiling; every M2B gate
  must pass for each conclusive record.

## Two-stage configuration

The first implementation revision deliberately left the expected contract
unresolved and pinned only `mani_skill==3.0.1`; it did not guess mplib, SAPIEN,
the action contract, controller semantics, or observation mode. The reviewed
schema-1.1 discovery report now binds those values locally, together with the
source, state, and compatibility identities. The authorized `1e-6` restoration
tolerance remains fixed and must never be loosened remotely. This local binding
cannot authorize trusted replay until it is committed, pushed, and matched by a
second trusted probe.

1. Push the locally tested reference integration.
2. Synchronize that exact SHA to the remote checkout.
3. Run the bounded compatibility probe and retrieve only its sanitized report.
4. Review the report locally and commit the exact dependency set, verified
   action layout, compatibility identity, and observed-safe state tolerance.
5. Push, resynchronize, and require a second passing probe that exactly matches
   the checked-in contract before source collection or trusted replay.

## Implementation slices

1. Lazy dependency availability and descriptive integration errors.
2. Versioned compatibility report and path-independent compatibility identity.
3. Strict action/controller/state/task/solver/task-source probe.
4. Canonical state-tree flattening, digesting, safe archive persistence, and
   full inventory/tamper validation.
5. Official-solver recording wrapper with action-copy/mutation/interception
   checks and deterministic bounded seed ordering.
6. Independent baseline replay and accepted-only M0 import.
7. Real `ReplayEnvironmentSession`, adapter, replay bundle, task-evidence
   semantics, and explicit default-registry entry.
8. Probe, collect, and replay CLIs with static dry runs and M2A resume/retry.
9. CPU-only fake tests and documentation.
10. Remote one-GPU gates, six-source collection, at least 18 proposals, at
    least 12 evaluated proposals, result reload, and zero-duplicate resume.

## Assumptions

- Runtime objects expose the public APIs verified by the compatibility probe;
  package version alone is never treated as sufficient.
- The official solver sends every physical action through the supplied
  environment object's public `step` method. The recorder verifies this rather
  than assuming it.
- Runtime paths are supplied separately from semantic adapter configuration and
  cannot participate in replay/evidence identity.
- The checked-in corruption plan will target only semantics actually verified
  by the remote controller configuration.

## Exclusions

No LangMani, model training, policy checkpoints, VLMs, images, videos, multiple
tasks or robots, vectorized replay, subtrajectory starts, CPU-simulator
acceptance, multiprocessing, distributed simulation, real robot execution,
manual relabeling, or general robot-safety claims are included.

## Validation and milestone state

- Starting revision: `246e086340ee3e054de2001c23c631d8a7b03239`.
- Branch: `codex/m2c-maniskill-pickcube`.
- Pre-M2C baseline on 2026-07-15: `694 passed, 3 skipped`.
- Final first-stage local CPU suite on 2026-07-15:
  `816 passed, 3 skipped`; all skips are the existing Windows
  directory-symlink privilege limitation.
- Local interpreter exception: this host currently exposes Python 3.12, 3.13,
  and 3.14 but not Python 3.11, so the first-stage suite ran on Python 3.13.5.
  The isolated remote environment and final M2C acceptance must use Python 3.11;
  this local exception does not relax the repository runtime contract.
- Required before each source/config push: `python -m pytest`, `ruff check .`,
  `ruff format --check .`, `mypy src`, milestone CLI smoke tests, and
  `git diff --check`.
- First-stage implementation status: local code, CLIs, safe archive, M0 import,
  replay adapter, fake tests, and documentation are present. Post-discovery
  dependency, compatibility, action-layout, and corruption bindings are now
  being applied locally from reviewed remote evidence; trusted acceptance is
  not yet claimed.
- Remote bootstrap on 2026-07-15 verified dedicated key-only `BatchMode=yes`
  authentication, a repository-scoped read-only deploy key, a clean checkout at
  `10658cf0554403aeedc291530f52527c7026636a`, and an isolated Python 3.11.15
  environment. The dependency gate passed with ManiSkill 3.0.1, SAPIEN 3.0.3,
  resolver-selected mplib 0.1.1, PyTorch 2.8.0+cu128, CUDA 12.8, and one visible
  RTX 5090.
- Discovery run `20260715T044250Z_m2c-pickcube-compat_10658cf_seed0` stopped
  before ManiSkill initialization because the later RET1 toolkit commit used a
  direct `mappingproxy` dataclass default that Python 3.11 rejects. It produced
  no compatibility report and made no simulator claim. The local fix replaces
  that default with a factory and adds a regression test; the remote must pull
  the resulting pushed SHA and rerun the discovery probe.
- Python 3.11 compatibility fix `aa29c1ef661fe554a427e7ee8be1ca38b78034a0`
  passed the complete local suite (`832 passed, 3 skipped`), Ruff lint/format,
  mypy, and diff checks, was pushed, and was synchronized exactly to the clean
  remote checkout.
- Discovery run `20260715T045027Z_m2c-pickcube-compat_aa29c1e_seed0` exposed a
  missing PhysX GPU runtime. The official precompiled runtime asset was
  downloaded outside the checkout, checked as a valid x86-64 ELF payload, and
  installed under the user runtime cache; `sapien.physx.enable_gpu()` then
  passed. No tracked source was changed remotely.
- Discovery run `20260715T055628Z_m2c-pickcube-compat_aa29c1e_seed0_physx`
  then stopped during environment creation because the headless server had no
  usable Vulkan rendering device. The documented AutoDL headless Vulkan
  runtime packages and NVIDIA EGL ICD were installed outside the checkout;
  both `vulkaninfo` and SAPIEN subsequently identified the single RTX 5090 as
  supported.
- Discovery run
  `20260715T061841Z_m2c-pickcube-compat_aa29c1e_seed0_vulkan-physx` completed
  environment creation, controller inspection, one bounded action, and clean
  close, but correctly stopped before collection. The complete state structure
  matched and the maximum set/get readback error was
  `1.1920928955078125e-07`, below the provisional `1e-6` tolerance, while the
  raw value digests differed. The implementation had incorrectly required both
  the configured numeric tolerance and byte-identical readback. Its sanitized
  report was retrieved for diagnosis; the fix must be made, validated, pushed,
  and rerun from a new exact SHA before resolving the trusted contract.
- Bounded diagnostic run
  `20260715T063126Z_m2c-state-convergence_aa29c1e_seed0` confirmed that repeated
  set/get does not reach a byte-identical fixed point: all five same-session
  iterations and a fresh-session restore preserved the full 4-leaf/70-component
  structure and stayed within `1e-6`, but each retained a
  `1.1920928955078125e-07` maximum error in the Panda articulation leaf and a
  different raw digest. The correct local fix therefore keeps archive digests
  exact while adding a narrowly authorized, configuration-bound complete-state
  numeric restoration contract for M2C; exact-digest remains the default trust
  mode for other adapters.
- The adapter-specific correction is explicitly authorized as
  `tolerance_verified_full_state_v1` with a fixed maximum absolute tolerance of
  `1e-6`. Archive manifests, inventory, leaf bytes, and digests remain exact;
  the runtime comparison must prove identical structure, paths, dtypes, shapes,
  finite values, and complete component coverage. Its semantic, tolerance,
  compatibility identity, observed maximum error, and compared component count
  are identity/evidence inputs. No other adapter inherits this authorization.
- The local correction upgrades the adapter to `1.1.0` and the compatibility
  report to schema `1.1`. Its complete local validation passed with
  `872 passed, 3 skipped` (the three existing Windows directory-symlink
  privilege skips), Ruff lint/format, mypy across 61 source files, all three M2C
  CLI help smokes, and `git diff --check`. The correction was committed, pushed,
  and synchronized exactly before the next remote probe.
- Schema-1.1 discovery run
  `20260715T073250Z_m2c-pickcube-compat_bb2c35c_seed0_numeric-v1` at exact pushed
  revision `bb2c35c6a96a6deac7baee55b294b1025c156074` passed the complete
  compatibility probe. It resolved ManiSkill `3.0.1`, mplib `0.1.1`, SAPIEN
  `3.0.3`, observation mode `none`, an eight-component float32 action contract,
  `physx_cuda`, the solver/task/controller/state identities, and compatibility
  identity `sha256:05a560fd89e0989b6d94017c6456efc535de4a0bf87ededaf468f5e2d9ef71f7`.
  Its sanitized schema-1.1 report was retrieved and reviewed locally.
- The resolved post-discovery contract, dependency pins, action layout, and
  three-transform M1 plan were validated locally (`874 passed, 3 skipped`),
  committed as `d2d918259046a65447a87c4eb4c4a485b0eccecc`, pushed, and synchronized
  exactly to the clean remote checkout.
- Trusted probe `20260715T074810Z_m2c-pickcube-trusted_d2d9182_seed0` passed
  with `trusted_replay_ready=true`. One-source gate
  `20260715T074954Z_m2c-pickcube-collect-gate_d2d9182_seed0` accepted its first
  attempt and independently replayed the baseline successfully. Formal
  collection `20260715T075040Z_m2c-pickcube-collect6_d2d9182_seed0` accepted
  six of six attempts with six successful independent baselines. M1 run
  `20260715T075148Z_m2c-pickcube-corrupt18_d2d9182_seed314159` generated 18
  immutable unlabeled proposals, six per checked-in transform, with no
  applicability skips.
- Paired-replay run
  `20260715T075300Z_m2c-pickcube-replay12_d2d9182_seed271828` was rejected:
  all 12 attempts preserved complete 70-component restoration with maximum
  error `1.1920928955078125e-07`, then raised Gymnasium `ResetNeeded` at
  baseline action step zero. No task failure was fabricated. The cause was an
  uninitialized public Gym wrapper: the replay session restored through the
  unwrapped state API without first calling the wrapper's public `reset`.
- The local adapter-only correction upgrades PickCube to `1.1.1`, binds the
  actual official source reset seed into archive-backed replay identity, and
  requires each fresh baseline/corrupted wrapper to execute
  `reset(bound_source_seed) -> set_state_dict -> get_state_dict` before task
  evaluation or stepping. Missing, out-of-range, or archive-drifted seeds fail
  closed, and source recording now proves the solver used the requested seed.
  Other adapters are unchanged. The correction passed full local validation
  (`879 passed, 3 skipped`, Ruff lint/format, mypy across 61 source files, all
  three M2C CLI help smokes, and `git diff --check`), was committed as
  `aafe83806515662937f6ec6c5d695aa44d1da537`, pushed, and synchronized exactly.
- Trusted probe `20260715T080753Z_m2c-pickcube-trusted_aafe838_seed0` passed on
  the corrected revision. One-proposal replay gate
  `20260715T080829Z_m2c-pickcube-replay1-gate_aafe838_seed271828` then completed
  one valid baseline and one conclusive strong simulator success with zero
  execution errors.
- Final smoke `20260715T080910Z_m2c-pickcube-replay12_aafe838_seed271828`
  completed 12 valid baselines, eight conclusive successes, four conclusive
  task failures, 12 projected strong simulator outcomes, and zero invalid,
  indeterminate, skipped, or execution-error attempts. All baseline and
  corrupted restorations compared the full 70-component state with maximum
  absolute error `1.1920928955078125e-07`. The class-balance and zero-error
  gates passed.
- An immediate `--resume` reused all 12 completed attempts with zero duplicate
  execution. The evaluation manifest SHA-256 remained
  `7b50220704f43f02c5876a73bb88b9fef50a94d33a79d7ea0890f746db3b8caa`
  before and after resume. M2C is accepted only for the documented single-task,
  single-robot, single-backend reference scope. No training was performed.
