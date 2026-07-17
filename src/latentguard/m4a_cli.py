"""Strict CPU-safe command surface for M4A visual dataset generation.

Parser construction and the validation/inspection paths import no simulator
packages.  The production PickCube runtime is constructed only after the
probe or render command has passed all static artifact bindings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast, runtime_checkable

from latentguard.evaluation.security import sanitize_operational_text
from latentguard.vision_data.source_binding import ACCEPTED_M3C_EXECUTION_GIT_SHA

if TYPE_CHECKING:
    from latentguard.integrations.maniskill_pickcube.compatibility import (
        CompatibilityBinding,
    )
    from latentguard.integrations.maniskill_pickcube.configuration import (
        ManiSkillPickCubeActionLayout,
    )
    from latentguard.integrations.maniskill_pickcube.visual_probe import (
        PickCubeVisualProbeRuntime,
        VisualCompatibilityReport,
    )
    from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
    from latentguard.vision_data.domains import RenderDomainConfigurationV1
    from latentguard.vision_data.models import (
        VisualVerifierDevelopmentDatasetV1,
        VisualVerifierExternalDatasetV1,
    )
    from latentguard.vision_data.packet import VisualObservationPacketV1
    from latentguard.vision_data.run_manifest import M4AOperationalRunManifestV1

    VisualDataset = VisualVerifierDevelopmentDatasetV1 | VisualVerifierExternalDatasetV1

M4A_COMMANDS = frozenset(
    {
        "probe-maniskill-pickcube-visual",
        "render-m3a-visual-dataset",
        "render-m3c-external-visual-dataset",
        "validate-visual-verifier-dataset",
        "inspect-visual-packet",
    }
)

DEFAULT_EXPECTED_CONTRACT = Path(
    "configs/integrations/maniskill_pickcube/expected-contract-v1.json"
)
DEFAULT_ACTION_LAYOUT = Path(
    "configs/integrations/maniskill_pickcube/action-layout-v1.json"
)
DEFAULT_CAMERA_RIG = Path("configs/vision/m4a/camera-rig-v1.json")
DEFAULT_RENDER_DOMAINS = Path("configs/vision/m4a/render-domains-v1.json")
VISUAL_PROBE_REPORT_FILENAME = "visual-compatibility-report.json"

OperationalCommandRunner = Callable[[tuple[str, ...], Path | None], str]


class M4ACommandError(ValueError):
    """Raised when an M4A command violates a frozen workflow contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise M4ACommandError(f"{context}: {reason}")


def _run_operational_command(
    arguments: tuple[str, ...], cwd: Path | None = None
) -> str:
    """Run one bounded metadata command without exposing its output on failure."""

    import subprocess

    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise M4ACommandError(
            "operational manifest: metadata command could not run"
        ) from exc
    if completed.returncode != 0:
        _fail("operational manifest", "metadata command failed")
    return completed.stdout.strip()


def _git_operational_identity(
    runner: OperationalCommandRunner = _run_operational_command,
) -> tuple[str, str]:
    """Return a clean checkout's full Git SHA and attached branch."""

    repository_root = Path(__file__).resolve().parents[2]
    if runner(("git", "status", "--porcelain"), repository_root):
        _fail("operational manifest", "Git checkout is modified")
    git_sha = runner(("git", "rev-parse", "HEAD"), repository_root)
    branch = runner(("git", "symbolic-ref", "--short", "HEAD"), repository_root)
    if not git_sha or not branch:
        _fail("operational manifest", "Git SHA or attached branch is unavailable")
    return git_sha, branch


def _single_gpu_operational_identity(
    runner: OperationalCommandRunner = _run_operational_command,
) -> tuple[str, str]:
    """Return the sole NVIDIA GPU model and driver, rejecting multi-GPU runs."""

    output = runner(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ),
        None,
    )
    lines = tuple(line.strip() for line in output.splitlines() if line.strip())
    if len(lines) != 1:
        _fail("operational manifest", "M4A requires exactly one visible GPU")
    parts = tuple(item.strip() for item in lines[0].rsplit(",", maxsplit=1))
    if len(parts) != 2 or not all(parts):
        _fail("operational manifest", "GPU model or driver is unavailable")
    gpu_model, driver_version = parts
    if "RTX 5090" not in gpu_model.upper():
        _fail("operational manifest", "M4A requires one RTX 5090")
    return gpu_model, driver_version


def _installed_operational_versions() -> dict[str, str]:
    """Read current runtime package/CUDA versions only on real GPU execution."""

    import importlib.metadata

    try:
        import torch

        cuda_version = torch.version.cuda
        if not isinstance(cuda_version, str) or not cuda_version:
            _fail("operational manifest", "PyTorch CUDA runtime is unavailable")
        return {
            "cuda": cuda_version,
            "maniskill": importlib.metadata.version("mani_skill"),
            "pytorch": torch.__version__,
            "sapien": importlib.metadata.version("sapien"),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise M4ACommandError(
            "operational manifest: required runtime package is unavailable"
        ) from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _uint32_int(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed < 2**32:
        raise argparse.ArgumentTypeError("must be a uint32 integer")
    return parsed


def _add_static_visual_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument(
        "--expected-contract", type=Path, default=DEFAULT_EXPECTED_CONTRACT
    )
    parser.add_argument("--action-layout", type=Path, default=DEFAULT_ACTION_LAYOUT)
    parser.add_argument("--camera-rig", type=Path, default=DEFAULT_CAMERA_RIG)
    parser.add_argument("--render-domains", type=Path, default=DEFAULT_RENDER_DOMAINS)


def _add_render_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--visual-probe-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=_uint32_int, required=True)
    parser.add_argument("--trajectory-limit", type=_positive_int)
    parser.add_argument("--anchor-limit", type=_positive_int)
    parser.add_argument("--domain-limit", type=_positive_int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-execution-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def add_m4a_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register exactly the five bounded M4A commands."""

    probe = subparsers.add_parser(
        "probe-maniskill-pickcube-visual",
        help="probe exact-state multi-view rendering and pixel determinism",
    )
    _add_static_visual_arguments(probe)
    probe.add_argument("--runtime-archive-dir", type=Path, required=True)
    probe.add_argument("--episode-id", required=True)
    probe.add_argument("--state-index", type=_nonnegative_int, required=True)
    probe.add_argument("--render-seed", type=_uint32_int, default=0)
    probe.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="publish one immutable probe report and operational manifest directory",
    )

    development = subparsers.add_parser(
        "render-m3a-visual-dataset",
        help="render the accepted M3A development dataset from exact states",
    )
    _add_static_visual_arguments(development)
    _add_render_control_arguments(development)
    development.add_argument("--runtime-archive-dir", type=Path, required=True)
    development.add_argument("--dataset-dir", type=Path, required=True)
    development.add_argument("--anchor-manifest-dir", type=Path, required=True)
    development.add_argument("--acceptance-report", type=Path, required=True)

    external = subparsers.add_parser(
        "render-m3c-external-visual-dataset",
        help="render the accepted M3C candidate pool as evaluation-only data",
    )
    _add_static_visual_arguments(external)
    _add_render_control_arguments(external)
    external.add_argument("--runtime-archive-dir", type=Path, required=True)
    external.add_argument("--source-dir", type=Path, required=True)
    external.add_argument("--anchor-manifest-dir", type=Path, required=True)
    external.add_argument("--candidate-pool-dir", type=Path, required=True)
    external.add_argument("--blind-manifest", type=Path, required=True)
    external.add_argument("--bound-result", type=Path, required=True)
    external.add_argument("--corruption-dir", type=Path, required=True)
    external.add_argument("--selected-evaluation-dir", type=Path, required=True)
    external.add_argument("--remainder-evaluation-dir", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate-visual-verifier-dataset",
        help="strictly reload visual data and enforce integrity/leakage gates",
    )
    _add_static_visual_arguments(validate)
    validate.add_argument("--visual-probe-report", type=Path, required=True)
    validate.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        required=True,
        help="one dataset root; repeat once to validate development/external leakage",
    )
    validate.add_argument(
        "--allow-partial",
        action="store_true",
        help="validate bounded smoke counts instead of the complete target counts",
    )
    validate.add_argument(
        "--report-dir",
        type=Path,
        help="publish a new immutable directory of compact sanitized JSON reports",
    )
    validate.add_argument(
        "--development-render-root",
        type=Path,
        help="development render root containing the bound render-jobs manifest",
    )
    validate.add_argument(
        "--external-render-root",
        type=Path,
        help="external render root containing the bound render-jobs manifest",
    )
    validate.add_argument("--m3a-runtime-archive-dir", type=Path)
    validate.add_argument("--m3a-source-dataset-dir", type=Path)
    validate.add_argument("--m3a-anchor-manifest-dir", type=Path)
    validate.add_argument("--m3a-acceptance-report", type=Path)
    validate.add_argument("--m3c-runtime-archive-dir", type=Path)
    validate.add_argument("--m3c-source-dir", type=Path)
    validate.add_argument("--m3c-anchor-manifest-dir", type=Path)
    validate.add_argument("--m3c-candidate-pool-dir", type=Path)
    validate.add_argument("--m3c-blind-manifest", type=Path)
    validate.add_argument("--m3c-bound-result", type=Path)
    validate.add_argument("--m3c-corruption-dir", type=Path)
    validate.add_argument("--m3c-selected-evaluation-dir", type=Path)
    validate.add_argument("--m3c-remainder-evaluation-dir", type=Path)

    inspect = subparsers.add_parser(
        "inspect-visual-packet",
        help="print a compact metadata-only packet summary",
    )
    source_group = inspect.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--packet-dir", type=Path)
    source_group.add_argument("--dataset-dir", type=Path)
    inspect.add_argument("--packet-id")
    inspect.add_argument("--output", type=Path)


@dataclass(frozen=True, slots=True)
class M4AStaticScope:
    """Strict static runtime and visual-configuration binding."""

    compatibility_binding: CompatibilityBinding
    action_layout: ManiSkillPickCubeActionLayout
    camera_rig: PickCubeMultiViewRigV1
    render_domains: RenderDomainConfigurationV1
    visual_probe_report: VisualCompatibilityReport | None = None


def _build_operational_run_manifest(
    *,
    args: argparse.Namespace,
    scope: M4AStaticScope,
    source_binding: object,
    run_id: str,
    external: bool | None,
    render_seed: int,
    canonical_source_identity_digest: str | None,
    render_job_inventory_digest: str | None,
    selected_packet_inventory_digest: str | None,
    visual_report: VisualCompatibilityReport | None = None,
    prior_manifest: M4AOperationalRunManifestV1 | None = None,
    runner: OperationalCommandRunner = _run_operational_command,
    started_at: str | None = None,
    launch_argv: tuple[str, ...] | None = None,
    hostname: str | None = None,
    python_version: str | None = None,
    installed_versions: Mapping[str, str] | None = None,
) -> M4AOperationalRunManifestV1:
    """Build strict operational evidence from the current clean GPU runtime."""

    import platform
    import socket

    from latentguard.vision_data.run_manifest import (
        M4AOperationalEnvironmentV1,
        M4AOperationalRunManifestV1,
        M4AOperationalSourceIdentityV1,
    )

    visual = scope.visual_probe_report if visual_report is None else visual_report
    if visual is None:
        _fail("operational manifest", "trusted visual compatibility is absent")
    compatibility = scope.compatibility_binding.report
    compatibility_runtime = compatibility.operational
    runtime_pairs = (
        (
            "ManiSkill",
            compatibility.mani_skill_version,
            visual.mani_skill_version,
        ),
        ("SAPIEN", compatibility.sapien_version, visual.sapien_version),
        ("PyTorch", compatibility_runtime.torch_version, visual.torch_version),
        (
            "CUDA",
            compatibility_runtime.cuda_runtime_version,
            visual.cuda_runtime_version,
        ),
        ("GPU", compatibility_runtime.gpu_model, visual.gpu_model),
    )
    drifted = tuple(
        name for name, current, probed in runtime_pairs if current != probed
    )
    if drifted:
        _fail(
            "operational manifest",
            "compatibility and visual probe runtime differ: " + ", ".join(drifted),
        )
    current_python = (
        platform.python_version() if python_version is None else python_version
    )
    if current_python != compatibility_runtime.python_version:
        _fail(
            "operational manifest",
            "current Python differs from the compatibility probe",
        )
    current_versions = (
        _installed_operational_versions()
        if installed_versions is None
        else dict(installed_versions)
    )
    expected_versions = {
        "cuda": visual.cuda_runtime_version,
        "maniskill": visual.mani_skill_version,
        "pytorch": visual.torch_version,
        "sapien": visual.sapien_version,
    }
    if current_versions != expected_versions:
        _fail(
            "operational manifest",
            "installed runtime versions differ from the visual probe",
        )
    live_gpu_model, driver_version = _single_gpu_operational_identity(runner)
    if live_gpu_model != visual.gpu_model:
        _fail("operational manifest", "live GPU differs from the visual probe")
    git_sha, branch = _git_operational_identity(runner)
    environment = M4AOperationalEnvironmentV1.create(
        hostname=socket.gethostname() if hostname is None else hostname,
        python_version=current_python,
        gpu_model=live_gpu_model,
        gpu_driver_version=driver_version,
        cuda_version=current_versions["cuda"],
        pytorch_version=current_versions["pytorch"],
        maniskill_version=current_versions["maniskill"],
        sapien_version=current_versions["sapien"],
        renderer_backend=visual.renderer_api.renderer_backend,
    )
    if external is None:
        if canonical_source_identity_digest is not None:
            _fail(
                "operational manifest",
                "visual probe cannot carry a render source identity digest",
            )
        source_identity = M4AOperationalSourceIdentityV1.for_visual_probe(
            visual.probe_source.as_mapping()
        )
    else:
        if canonical_source_identity_digest is None:
            _fail("operational manifest", "render source identity digest is missing")
        source_identity = (
            M4AOperationalSourceIdentityV1.for_m3c(
                cast(Any, source_binding).source_set_digest,
                cast(Any, source_binding).candidate_pool_digest,
                canonical_source_identity_digest=canonical_source_identity_digest,
            )
            if external
            else M4AOperationalSourceIdentityV1.for_m3a(
                cast(Any, source_binding).dataset_digest,
                canonical_source_identity_digest=canonical_source_identity_digest,
            )
        )
    effective_started_at = (
        prior_manifest.started_at
        if prior_manifest is not None
        else (
            datetime.now(UTC).isoformat().replace("+00:00", "Z")
            if started_at is None
            else started_at
        )
    )
    effective_argv = (
        prior_manifest.launch_argv
        if prior_manifest is not None
        else (tuple(sys.argv) if launch_argv is None else launch_argv)
    )
    return M4AOperationalRunManifestV1.create(
        run_id=run_id,
        command=args.command,
        git_sha=git_sha,
        branch=branch,
        started_at=effective_started_at,
        render_seed=render_seed,
        launch_argv=effective_argv,
        environment=environment,
        visual_compatibility_identity=visual.visual_compatibility_identity,
        camera_rig_digest=scope.camera_rig.rig_digest,
        render_domain_configuration_digest=scope.render_domains.content_digest,
        source_identity=source_identity,
        render_job_inventory_digest=render_job_inventory_digest,
        selected_packet_inventory_digest=selected_packet_inventory_digest,
    )


@dataclass(frozen=True, slots=True)
class M4ARenderCommandResult:
    """Sanitized completion facts returned by an injectable render backend."""

    dataset_digest: str | None
    packet_count: int
    image_count: int
    candidate_count: int
    rendered_packet_count: int
    preserved_packet_count: int = 0
    execution_error_count: int = 0
    zero_work_resume_verified: bool = False

    def __post_init__(self) -> None:
        if self.dataset_digest is not None and (
            len(self.dataset_digest) != 71
            or not self.dataset_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in self.dataset_digest[7:]
            )
        ):
            _fail("render result", "dataset digest is invalid")
        for name in (
            "packet_count",
            "image_count",
            "candidate_count",
            "rendered_packet_count",
            "preserved_packet_count",
            "execution_error_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail("render result", f"{name} must be non-negative")
        if self.image_count != self.packet_count * 3:
            _fail("render result", "each packet must contain exactly three images")
        if type(self.zero_work_resume_verified) is not bool:
            _fail("render result", "zero_work_resume_verified must be boolean")
        if (
            self.rendered_packet_count
            + self.preserved_packet_count
            + self.execution_error_count
            > self.packet_count
        ):
            _fail("render result", "packet accounting exceeds the job inventory")

    @property
    def dataset_published(self) -> bool:
        """Return whether a complete three-domain dataset was published."""

        return self.dataset_digest is not None


@runtime_checkable
class M4ARenderBackend(Protocol):
    """Simulator boundary used by production and CPU-only fake render tests."""

    def render_m3a(
        self,
        *,
        args: argparse.Namespace,
        scope: M4AStaticScope,
        source_binding: object,
    ) -> M4ARenderCommandResult:
        """Render one bounded or complete M3A visual dataset."""
        ...

    def render_m3c(
        self,
        *,
        args: argparse.Namespace,
        scope: M4AStaticScope,
        source_binding: object,
    ) -> M4ARenderCommandResult:
        """Render one bounded or complete M3C external visual dataset."""
        ...


def _load_static_scope(
    args: argparse.Namespace, *, require_visual_probe: bool
) -> M4AStaticScope:
    """Load and bind all static compatibility and visual configuration."""

    from latentguard.integrations.maniskill_pickcube.compatibility import (
        load_compatibility_report,
        validate_compatibility_report,
    )
    from latentguard.integrations.maniskill_pickcube.configuration import (
        load_expected_contract,
        load_maniskill_pickcube_action_layout,
        validate_maniskill_pickcube_action_layout_binding,
    )
    from latentguard.vision_data.configuration import (
        load_camera_rig_configuration,
        load_render_domain_configuration,
    )

    expected = load_expected_contract(args.expected_contract)
    report = load_compatibility_report(args.compatibility_report)
    compatibility = validate_compatibility_report(
        report, expected, require_trusted=True
    )
    layout = load_maniskill_pickcube_action_layout(args.action_layout)
    validate_maniskill_pickcube_action_layout_binding(
        layout, compatibility, layout.m1_action_layout
    )
    rig = cast("PickCubeMultiViewRigV1", load_camera_rig_configuration(args.camera_rig))
    domains = cast(
        "RenderDomainConfigurationV1",
        load_render_domain_configuration(args.render_domains),
    )
    visual_report: VisualCompatibilityReport | None = None
    if require_visual_probe:
        from latentguard.integrations.maniskill_pickcube.visual_probe import (
            load_visual_compatibility_report,
        )

        visual_report = load_visual_compatibility_report(
            args.visual_probe_report, require_trusted_generation=True
        )
        if (
            visual_report.pickcube_compatibility_identity
            != compatibility.report.compatibility_identity
        ):
            _fail(
                "visual compatibility",
                "probe is bound to a different PickCube compatibility identity",
            )
        if visual_report.camera_rig_digest != rig.rig_digest:
            _fail("visual compatibility", "camera rig digest drifted after probe")
        if visual_report.render_domain_configuration_digest != domains.content_digest:
            _fail(
                "visual compatibility",
                "render-domain configuration digest drifted after probe",
            )
    return M4AStaticScope(
        compatibility_binding=compatibility,
        action_layout=layout,
        camera_rig=rig,
        render_domains=domains,
        visual_probe_report=visual_report,
    )


def _default_probe_runtime(scope: M4AStaticScope) -> PickCubeVisualProbeRuntime:
    """Construct the lazy real-session probe boundary after static validation."""

    from latentguard.integrations.maniskill_pickcube.serialization import (
        action_contract_from_compatibility,
        environment_settings_from_compatibility,
    )
    from latentguard.integrations.maniskill_pickcube.task_evidence import (
        PickCubeTaskKeyContract,
    )
    from latentguard.integrations.maniskill_pickcube.visual_probe import (
        SessionPickCubeVisualProbeRuntime,
    )
    from latentguard.integrations.maniskill_pickcube.visual_session import (
        create_default_pickcube_visual_session,
    )

    binding = scope.compatibility_binding
    layout = scope.action_layout
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding, coordinate_frame=layout.coordinate_frame
    )
    task_keys = PickCubeTaskKeyContract.from_compatibility_report(binding.report)
    session = create_default_pickcube_visual_session(
        settings=settings,
        action_contract=action_contract,
        task_key_contract=task_keys,
    )
    return SessionPickCubeVisualProbeRuntime(session)


def _validate_probe_output_root(args: argparse.Namespace) -> Path:
    """Require an absent, non-overlapping probe evidence directory."""

    from latentguard.vision_data.serialization import _is_link_or_junction

    destination = Path(args.output_root).absolute()
    resolved_output = destination.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if (repository_root / ".git").exists() and resolved_output.is_relative_to(
        repository_root
    ):
        _fail("visual probe output", "raw evidence root must remain outside Git")
    if destination.exists() or _is_link_or_junction(destination):
        _fail("visual probe output", "output root must be absent")
    for name, value in vars(args).items():
        if name in {"command", "output_root"} or not isinstance(value, Path):
            continue
        resolved_input = value.absolute().resolve()
        if (
            resolved_output == resolved_input
            or resolved_output.is_relative_to(resolved_input)
            or resolved_input.is_relative_to(resolved_output)
        ):
            _fail("visual probe paths", "output root must not overlap an input")
    return destination


def _run_probe_maniskill_pickcube_visual(
    args: argparse.Namespace,
    *,
    runtime: PickCubeVisualProbeRuntime | None = None,
    operational_runner: OperationalCommandRunner = _run_operational_command,
) -> int:
    """Run the exact-state probe and atomically publish report plus run evidence."""

    destination = _validate_probe_output_root(args)
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    scope = _load_static_scope(args, require_visual_probe=False)
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        find_state_indexed_episode,
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.visual_probe import (
        load_visual_compatibility_report,
        probe_maniskill_pickcube_visual,
        write_visual_compatibility_report,
    )
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        build_pickcube_visual_render_plan,
    )

    archive = load_state_indexed_archive(args.runtime_archive_dir)
    episode = find_state_indexed_episode(archive, args.episode_id)
    if args.state_index >= len(episode.states):
        _fail("visual probe state", "state index is outside the selected episode")
    state = episode.states[args.state_index]
    rig = scope.camera_rig
    domains = scope.render_domains
    canonical = domains.domain("canonical")
    plan = build_pickcube_visual_render_plan(rig, canonical, domains, args.render_seed)
    selected_runtime = runtime if runtime is not None else _default_probe_runtime(scope)
    report = probe_maniskill_pickcube_visual(
        compatibility_binding=scope.compatibility_binding,
        runtime=selected_runtime,
        source_archive=archive,
        source_episode=episode,
        source_state=state,
        camera_rig=rig,
        render_domain_configuration=domains,
        render_plan=plan,
        report_path=None,
        require_trusted_generation=False,
    )
    from latentguard.vision_data.run_manifest import (
        M4A_RUN_MANIFEST_FILENAME,
        load_m4a_run_manifest,
        save_m4a_run_manifest,
    )
    from latentguard.vision_data.serialization import _is_link_or_junction

    run_id = "m4a-probe-" + report.visual_compatibility_identity[7:23]
    manifest = _build_operational_run_manifest(
        args=args,
        scope=scope,
        source_binding=report.probe_source,
        run_id=run_id,
        external=None,
        render_seed=args.render_seed,
        canonical_source_identity_digest=None,
        render_job_inventory_digest=None,
        selected_packet_inventory_digest=None,
        visual_report=report,
        runner=operational_runner,
        started_at=started_at,
    )
    import shutil
    import tempfile

    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_junction(destination.parent):
        _fail("visual probe output", "parent cannot be a link or junction")
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.probe-staging-",
            dir=destination.parent,
        )
    )
    staging = temporary_root / "probe"
    try:
        staging.mkdir()
        report_path = staging / VISUAL_PROBE_REPORT_FILENAME
        manifest_path = staging / M4A_RUN_MANIFEST_FILENAME
        write_visual_compatibility_report(report, report_path)
        save_m4a_run_manifest(manifest, manifest_path)
        reloaded_report = load_visual_compatibility_report(
            report_path,
            require_trusted_generation=False,
        )
        if (
            reloaded_report.visual_compatibility_identity
            != report.visual_compatibility_identity
        ):
            _fail("visual probe output", "report staging reload changed identity")
        if load_m4a_run_manifest(manifest_path) != manifest:
            _fail("visual probe output", "manifest staging reload changed content")
        staging.rename(destination)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    report.require_trusted_visual_generation_ready()
    print(
        "probe-maniskill-pickcube-visual OK: "
        f"visual_compatibility_identity={report.visual_compatibility_identity} "
        "repeated_pixels_exact=true fresh_environment_pixels_exact=true "
        "state_integrity_verified=true output_published=true"
    )
    return 0


def _validate_render_controls(args: argparse.Namespace) -> None:
    if args.retry_execution_errors and not args.resume:
        _fail("render controls", "--retry-execution-errors requires --resume")
    resolved_output = Path(args.output_root).absolute().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if (repository_root / ".git").exists() and resolved_output.is_relative_to(
        repository_root
    ):
        _fail("render output", "raw visual root must remain outside Git")
    input_paths = [
        value
        for name, value in vars(args).items()
        if name not in {"output_root", "command"} and isinstance(value, Path)
    ]
    for input_path in input_paths:
        resolved_input = input_path.absolute().resolve()
        if (
            resolved_output == resolved_input
            or resolved_output.is_relative_to(resolved_input)
            or resolved_input.is_relative_to(resolved_output)
        ):
            _fail("render paths", "output root must not overlap an input artifact")
    if args.resume:
        if not resolved_output.is_dir() or resolved_output.is_symlink():
            _fail("render resume", "output root must be an existing regular directory")
    elif resolved_output.exists() or resolved_output.is_symlink():
        _fail("render output", "must be absent unless --resume is used")


def _render_summary(command: str, result: M4ARenderCommandResult) -> int:
    dataset_digest = result.dataset_digest or "not_published"
    print(
        f"{command} OK: dataset_digest={dataset_digest} "
        f"packets={result.packet_count} images={result.image_count} "
        f"candidates={result.candidate_count} "
        f"rendered={result.rendered_packet_count} "
        f"preserved={result.preserved_packet_count} "
        f"execution_errors={result.execution_error_count}"
        f" dataset_published={str(result.dataset_published).lower()}"
        " zero_work_resume_verified="
        f"{str(result.zero_work_resume_verified).lower()}"
    )
    return 0


def _run_render_m3a_visual_dataset(
    args: argparse.Namespace, *, backend: M4ARenderBackend | None = None
) -> int:
    """Preflight accepted M3A artifacts and run the injectable renderer."""

    _validate_render_controls(args)
    scope = _load_static_scope(args, require_visual_probe=True)
    from latentguard.vision_data.source_binding import (
        load_m3a_development_source_binding,
    )

    source = load_m3a_development_source_binding(
        args.dataset_dir,
        args.runtime_archive_dir,
        args.anchor_manifest_dir,
        acceptance_report=args.acceptance_report,
    )
    if source.compatibility_identity != (
        scope.compatibility_binding.report.compatibility_identity
    ):
        _fail("M3A source", "compatibility identity differs from visual runtime")
    selected_backend = backend if backend is not None else _default_render_backend()
    result = selected_backend.render_m3a(args=args, scope=scope, source_binding=source)
    if args.dry_run:
        print(
            "render-m3a-visual-dataset dry-run OK: "
            f"source_dataset_digest={source.dataset_digest} "
            f"packets={result.packet_count} images={result.image_count} "
            f"candidates={result.candidate_count} dataset_published=false "
            "output_created=false"
        )
        return 0
    return _render_summary(args.command, result)


def _run_render_m3c_external_visual_dataset(
    args: argparse.Namespace, *, backend: M4ARenderBackend | None = None
) -> int:
    """Preflight accepted M3C outcomes and run the evaluation-only renderer."""

    _validate_render_controls(args)
    scope = _load_static_scope(args, require_visual_probe=True)
    from latentguard.vision_data.source_binding import (
        load_m3c_external_source_binding,
        require_m3c_anchor_archive_binding,
    )

    source = load_m3c_external_source_binding(
        args.candidate_pool_dir,
        args.blind_manifest,
        args.bound_result,
        args.corruption_dir,
        args.selected_evaluation_dir,
        args.remainder_evaluation_dir,
        expected_execution_git_sha=ACCEPTED_M3C_EXECUTION_GIT_SHA,
    )
    require_m3c_anchor_archive_binding(
        source,
        args.source_dir,
        args.runtime_archive_dir,
        args.anchor_manifest_dir,
    )
    if source.simulator_compatibility_identity != (
        scope.compatibility_binding.report.compatibility_identity
    ):
        _fail("M3C source", "compatibility identity differs from visual runtime")
    selected_backend = backend if backend is not None else _default_render_backend()
    result = selected_backend.render_m3c(args=args, scope=scope, source_binding=source)
    if args.dry_run:
        print(
            "render-m3c-external-visual-dataset dry-run OK: "
            f"source_set_digest={source.source_set_digest} "
            f"packets={result.packet_count} images={result.image_count} "
            f"candidates={result.candidate_count} training_allowed=false "
            "dataset_published=false output_created=false"
        )
        return 0
    return _render_summary(args.command, result)


@dataclass(frozen=True, slots=True)
class _RenderAnchorContext:
    source_collection: Any
    split: Any
    source_trajectory_id: str
    source_reset_seed: int
    split_group_id: str
    anchor_id: str
    episode: Any
    state: Any
    candidate_group: Any
    candidates: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PreparedRenderPlan:
    inventory: Any
    contexts: tuple[_RenderAnchorContext, ...]
    context_by_packet_id: Mapping[str, _RenderAnchorContext]
    selected_packet_ids: tuple[str, ...]
    source_model: Any
    archive_digest: str
    manifest_digest: str
    all_required_domains: bool


def _semantic_digest(payload: object, *, context: str) -> str:
    import hashlib

    from latentguard.replay.identity import canonical_json_bytes

    encoded = canonical_json_bytes(payload, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _selected_values(
    values: tuple[Any, ...], limit: int | None, *, context: str
) -> tuple[Any, ...]:
    if limit is None:
        return values
    selected = values[:limit]
    if not selected:
        _fail(context, "limit selected no content")
    return selected


def _load_m3a_render_contexts(
    args: argparse.Namespace, source_binding: Any
) -> tuple[tuple[_RenderAnchorContext, ...], Any, str, str]:
    from latentguard.action_verifier import load_action_verifier_dataset
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.vision_data.models import (
        PICKCUBE_CANONICAL_TASK_TEXT,
        PICKCUBE_VISUAL_TASK_ID,
        SourceCollection,
        VisualDatasetSplit,
    )

    dataset = load_action_verifier_dataset(args.dataset_dir)
    archive = load_state_indexed_archive(args.runtime_archive_dir)
    manifest = load_anchor_manifest(args.anchor_manifest_dir)
    if dataset.content_digest != source_binding.dataset_digest:
        _fail("M3A render source", "dataset changed after accepted binding")
    if manifest.content_digest != source_binding.anchor_manifest_digest:
        _fail("M3A render source", "anchor manifest changed after accepted binding")
    if archive.content_digest != source_binding.source_archive_digest:
        _fail("M3A render source", "runtime archive changed after accepted binding")
    if manifest.source_archive_content_digest != archive.content_digest:
        _fail("M3A render source", "runtime archive differs from anchor manifest")
    assignments = tuple(dataset.split_assignments)
    selected_assignments = _selected_values(
        assignments, args.trajectory_limit, context="M3A trajectories"
    )
    trajectory_ids = {
        assignment.source_trajectory_id for assignment in selected_assignments
    }
    groups = tuple(
        group
        for group in dataset.candidate_groups
        if group.source_trajectory_id in trajectory_ids
    )
    groups = _selected_values(groups, args.anchor_limit, context="M3A anchors")
    records = {record.anchor.anchor_id: record for record in manifest.records}
    episodes = {episode.episode_id: episode for episode in archive.episodes}
    samples = {sample.sample_id: sample for sample in dataset.samples}
    contexts: list[_RenderAnchorContext] = []
    for group in groups:
        record = records.get(group.anchor_id)
        if record is None:
            _fail(group.anchor_id, "accepted anchor record is absent")
        episode = episodes.get(record.source_archive_episode_id)
        if episode is None or record.anchor.state_index >= len(episode.states):
            _fail(group.anchor_id, "archived anchor state is absent")
        state = episode.states[record.anchor.state_index]
        if (
            group.source_trajectory_id != record.anchor.source_trajectory_id
            or group.source_seed != record.anchor.source_seed
            or group.split_group_id != record.anchor.split_group_id
            or group.state_content_digest != state.content_digest
            or record.source_state_content_digest != state.content_digest
            or record.source_state_digest != state.state_digest
            or record.verifier_state_content_digest
            != state.verifier_state.content_digest
        ):
            _fail(group.anchor_id, "dataset, manifest, and archived state differ")
        candidate_ids = (group.source_sample_id, *group.corrupted_sample_ids)
        group_samples: list[Any] = []
        for sample_id in candidate_ids:
            sample = samples.get(sample_id)
            if sample is None:
                _fail(group.anchor_id, "candidate sample is absent")
            if (
                sample.anchor_id != group.anchor_id
                or sample.source_trajectory_id != group.source_trajectory_id
                or sample.split_group_id != group.split_group_id
                or sample.task_id != PICKCUBE_VISUAL_TASK_ID
                or sample.instruction != PICKCUBE_CANONICAL_TASK_TEXT
            ):
                _fail(sample_id, "candidate differs from fixed visual source contract")
            group_samples.append(sample)
        contexts.append(
            _RenderAnchorContext(
                source_collection=SourceCollection.M3A_DEVELOPMENT,
                split=VisualDatasetSplit(group.dataset_split.value),
                source_trajectory_id=group.source_trajectory_id,
                source_reset_seed=group.source_seed,
                split_group_id=group.split_group_id,
                anchor_id=group.anchor_id,
                episode=episode,
                state=state,
                candidate_group=group,
                candidates=tuple(group_samples),
            )
        )
    if not contexts:
        _fail("M3A render source", "no anchor remained after limits")
    return tuple(contexts), dataset, archive.content_digest, manifest.content_digest


def _load_m3c_render_contexts(
    args: argparse.Namespace, source_binding: Any
) -> tuple[tuple[_RenderAnchorContext, ...], Any, str, str]:
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.selection.serialization import load_candidate_pool
    from latentguard.selection.source import load_candidate_pool_sources
    from latentguard.vision_data.models import (
        PICKCUBE_VISUAL_TASK_ID,
        SourceCollection,
        VisualDatasetSplit,
    )

    pool = load_candidate_pool(args.candidate_pool_dir)
    archive = load_state_indexed_archive(args.runtime_archive_dir)
    manifest = load_anchor_manifest(args.anchor_manifest_dir)
    sources = load_candidate_pool_sources(
        args.source_dir,
        args.runtime_archive_dir,
        args.anchor_manifest_dir,
    )
    if pool.content_digest != source_binding.candidate_pool_digest:
        _fail("M3C render source", "candidate pool changed after accepted binding")
    if manifest.source_archive_content_digest != archive.content_digest:
        _fail("M3C render source", "runtime archive differs from anchor manifest")
    ordered_trajectories = tuple(
        dict.fromkeys(group.source_trajectory_id for group in pool.groups)
    )
    selected_trajectories = set(
        _selected_values(
            ordered_trajectories,
            args.trajectory_limit,
            context="M3C trajectories",
        )
    )
    groups = tuple(
        group
        for group in pool.groups
        if group.source_trajectory_id in selected_trajectories
    )
    groups = _selected_values(groups, args.anchor_limit, context="M3C anchors")
    records = {record.anchor.anchor_id: record for record in manifest.records}
    episodes = {episode.episode_id: episode for episode in archive.episodes}
    projected_sources = {source.anchor_id: source for source in sources}
    contexts: list[_RenderAnchorContext] = []
    for group in groups:
        record = records.get(group.anchor_id)
        projected = projected_sources.get(group.anchor_id)
        if record is None or projected is None:
            _fail(group.anchor_id, "external anchor source is absent")
        episode = episodes.get(record.source_archive_episode_id)
        if episode is None or record.anchor.state_index >= len(episode.states):
            _fail(group.anchor_id, "external archived state is absent")
        state = episode.states[record.anchor.state_index]
        if (
            projected.source_task_id != PICKCUBE_VISUAL_TASK_ID
            or group.source_trajectory_id != record.anchor.source_trajectory_id
            or group.trajectory.source_seed != record.anchor.source_seed
            or group.trajectory.split_group_id != record.anchor.split_group_id
            or group.state_content_digest != state.content_digest
            or group.verifier_state_content_digest
            != state.verifier_state.content_digest
            or record.source_state_content_digest != state.content_digest
        ):
            _fail(group.anchor_id, "external pool, source, and archive differ")
        contexts.append(
            _RenderAnchorContext(
                source_collection=SourceCollection.M3C_EXTERNAL,
                split=VisualDatasetSplit.EXTERNAL,
                source_trajectory_id=group.source_trajectory_id,
                source_reset_seed=group.trajectory.source_seed,
                split_group_id=group.trajectory.split_group_id,
                anchor_id=group.anchor_id,
                episode=episode,
                state=state,
                candidate_group=group,
                candidates=tuple(group.candidates),
            )
        )
    if not contexts:
        _fail("M3C render source", "no anchor remained after limits")
    return tuple(contexts), pool, archive.content_digest, manifest.content_digest


def _build_render_plan(
    args: argparse.Namespace,
    scope: M4AStaticScope,
    source_binding: Any,
    contexts: tuple[_RenderAnchorContext, ...],
    source_model: Any,
    archive_digest: str,
    manifest_digest: str,
) -> _PreparedRenderPlan:
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        VISUAL_RENDERER_SEMANTIC_VERSION,
        build_pickcube_visual_render_plan,
    )
    from latentguard.vision_data.domains import assigned_render_domain_ids
    from latentguard.vision_data.pipeline import (
        VisualPacketJobV1,
        VisualRenderJobInventoryV1,
        derive_render_seed,
    )

    if args.domain_limit is not None and args.domain_limit > 3:
        _fail("render domains", "--domain-limit cannot exceed three")
    rig = scope.camera_rig
    domains = scope.render_domains
    visual = scope.visual_probe_report
    if visual is None:
        _fail("render planning", "trusted visual compatibility is absent")
    jobs: list[Any] = []
    selected_packet_ids: list[str] = []
    context_by_packet: dict[str, _RenderAnchorContext] = {}
    for context in contexts:
        authorized = assigned_render_domain_ids(
            context.source_collection, context.split
        )
        selected_domains = set(
            authorized if args.domain_limit is None else authorized[: args.domain_limit]
        )
        for domain_id in authorized:
            domain = domains.domain(domain_id)
            render_seed = derive_render_seed(
                context.anchor_id,
                domain_id,
                args.seed,
                semantic=domains.seed_derivation,
            )
            resolved_plan = build_pickcube_visual_render_plan(
                rig,
                domain,
                domains,
                render_seed,
            )
            camera_digests = cast(
                tuple[str, str, str],
                tuple(
                    camera.camera_configuration_digest
                    for camera in resolved_plan.cameras
                ),
            )
            job = VisualPacketJobV1.create(
                job_ordinal=len(jobs),
                source_collection=context.source_collection,
                source_trajectory_id=context.source_trajectory_id,
                source_reset_seed=context.source_reset_seed,
                anchor_id=context.anchor_id,
                split=context.split,
                split_group_id=context.split_group_id,
                state_reference_id=context.state.content_digest,
                expected_state_digest=context.state.state_digest,
                verifier_state_semantic=context.state.verifier_state.semantic,
                verifier_state_digest=context.state.verifier_state.content_digest,
                camera_rig_id=rig.rig_id,
                camera_rig_digest=rig.rig_digest,
                render_domain_id=domain_id,
                render_domain_digest=domain.domain_digest,
                base_render_seed=args.seed,
                seed_derivation=domains.seed_derivation,
                camera_configuration_digests=camera_digests,
                visual_compatibility_identity=(visual.visual_compatibility_identity),
                pickcube_compatibility_identity=(
                    scope.compatibility_binding.report.compatibility_identity
                ),
                renderer_semantic_version=VISUAL_RENDERER_SEMANTIC_VERSION,
            )
            jobs.append(job)
            context_by_packet[job.packet_id] = context
            if domain_id in selected_domains:
                selected_packet_ids.append(job.packet_id)
    source_identity = _semantic_digest(
        {
            "archive_digest": archive_digest,
            "manifest_digest": manifest_digest,
            "source_binding": {
                name: getattr(source_binding, name)
                for name in (
                    (
                        "dataset_digest",
                        "source_archive_digest",
                        "anchor_manifest_digest",
                        "split_digest",
                        "evidence_digest",
                        "acceptance_report_digest",
                        "compatibility_identity",
                    )
                    if contexts[0].source_collection.value == "m3a_development"
                    else (
                        "source_set_digest",
                        "candidate_pool_digest",
                        "blind_manifest_digest",
                        "bound_result_digest",
                        "replay_evidence_digest",
                        "evidence_projection_digest",
                        "full_outcome_digest",
                    )
                )
            },
            "source_model_digest": source_model.content_digest,
        },
        context="M4ARenderSourceIdentityV1",
    )
    inventory = VisualRenderJobInventoryV1.create(
        source_identity_digest=source_identity,
        camera_rig_digest=rig.rig_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=visual.visual_compatibility_identity,
        base_render_seed=args.seed,
        seed_derivation=domains.seed_derivation,
        jobs=jobs,
    )
    return _PreparedRenderPlan(
        inventory=inventory,
        contexts=contexts,
        context_by_packet_id=context_by_packet,
        selected_packet_ids=tuple(selected_packet_ids),
        source_model=source_model,
        archive_digest=archive_digest,
        manifest_digest=manifest_digest,
        all_required_domains=args.domain_limit in (None, 3),
    )


def _create_visual_session(scope: M4AStaticScope) -> Any:
    from latentguard.integrations.maniskill_pickcube.serialization import (
        action_contract_from_compatibility,
        environment_settings_from_compatibility,
    )
    from latentguard.integrations.maniskill_pickcube.task_evidence import (
        PickCubeTaskKeyContract,
    )
    from latentguard.integrations.maniskill_pickcube.visual_session import (
        create_default_pickcube_visual_session,
    )

    binding = scope.compatibility_binding
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding, coordinate_frame=scope.action_layout.coordinate_frame
    )
    return create_default_pickcube_visual_session(
        settings=settings,
        action_contract=action_contract,
        task_key_contract=PickCubeTaskKeyContract.from_compatibility_report(
            binding.report
        ),
    )


def _pipeline_callback(
    plan: _PreparedRenderPlan, scope: M4AStaticScope
) -> Callable[[Any], Any]:
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        ManiSkillVisualContractError,
        build_pickcube_visual_render_plan,
    )
    from latentguard.integrations.maniskill_pickcube.visual_session import (
        ManiSkillVisualInvalidContextError,
        build_visual_observation_packet,
    )
    from latentguard.vision_data.pipeline import InvalidVisualRenderPacketError

    session = _create_visual_session(scope)
    visual = scope.visual_probe_report
    if visual is None:
        _fail("render callback", "trusted visual compatibility is absent")

    def render(job: Any) -> Any:
        context = plan.context_by_packet_id[job.packet_id]
        try:
            render_plan = build_pickcube_visual_render_plan(
                scope.camera_rig,
                scope.render_domains.domain(job.render_domain_id),
                scope.render_domains,
                job.render_seed,
            )
            result = session.render_state(
                source_episode=context.episode,
                source_state=context.state,
                render_plan=render_plan,
                repeat_count=1,
            )
            return build_visual_observation_packet(
                result,
                source_collection=context.source_collection,
                anchor_id=context.anchor_id,
                split=context.split,
                split_group_id=context.split_group_id,
                state_reference_id=context.state.content_digest,
                camera_rig=scope.camera_rig,
                render_domain_configuration=scope.render_domains,
                visual_compatibility=visual,
            )
        except (
            ManiSkillVisualContractError,
            ManiSkillVisualInvalidContextError,
        ) as exc:
            message = sanitize_operational_text(exc)
            raise InvalidVisualRenderPacketError(f"{job.packet_id}: {message}") from exc

    return render


def _packet_images(pipeline_result: Any) -> dict[str, Any]:
    from latentguard.vision_data.serialization import load_visual_image

    images: dict[str, Any] = {}
    for root, packet in zip(
        pipeline_result.packet_bundle_roots,
        pipeline_result.packets,
        strict=True,
    ):
        for view in packet.views:
            if view.image_reference in images:
                _fail("visual dataset", "duplicate image reference across packets")
            images[view.image_reference] = load_visual_image(root, view.image_reference)
    return images


def _packet_ids_by_anchor(packets: tuple[Any, ...]) -> dict[str, tuple[str, ...]]:
    from latentguard.vision_data.domains import assigned_render_domain_ids

    grouped: dict[str, list[Any]] = {}
    for packet in packets:
        grouped.setdefault(packet.anchor_id, []).append(packet)
    result: dict[str, tuple[str, ...]] = {}
    for anchor_id, values in grouped.items():
        first = values[0]
        by_domain = {packet.render_domain_id: packet.packet_id for packet in values}
        domains = assigned_render_domain_ids(first.source_collection, first.split)
        if set(by_domain) != set(domains):
            _fail(anchor_id, "cannot finalize without all three render domains")
        result[anchor_id] = tuple(by_domain[domain] for domain in domains)
    return result


def _build_m3a_visual_dataset(
    plan: _PreparedRenderPlan,
    source_binding: Any,
    scope: M4AStaticScope,
    packets: tuple[Any, ...],
    *,
    full_target: bool,
) -> object:
    from latentguard.vision_data.dataset import build_visual_development_dataset
    from latentguard.vision_data.models import (
        VisualActionVerifierSampleV1,
        VisualCandidateBindingV1,
    )
    from latentguard.vision_data.source_binding import (
        visual_candidate_array_reference,
    )

    packet_ids = _packet_ids_by_anchor(packets)
    bindings: list[Any] = []
    samples: list[Any] = []
    pending_digest = _semantic_digest(
        {"inventory": plan.inventory.content_digest, "semantic": "pending_v1"},
        context="M4APendingDatasetV1",
    )
    visual = scope.visual_probe_report
    if visual is None:
        _fail("M3A dataset", "trusted visual compatibility is absent")
    for context in plan.contexts:
        group = context.candidate_group
        ids = cast(tuple[str, str, str], packet_ids[context.anchor_id])
        for candidate in context.candidates:
            bindings.append(
                VisualCandidateBindingV1(
                    candidate_sample_id=candidate.sample_id,
                    candidate_group_id=group.group_id,
                    anchor_id=context.anchor_id,
                    source_collection=context.source_collection,
                    packet_ids=ids,
                )
            )
            for packet_id in ids:
                samples.append(
                    VisualActionVerifierSampleV1(
                        packet_id=packet_id,
                        candidate_sample_id=candidate.sample_id,
                        task_id=candidate.task_id,
                        canonical_task_text=candidate.instruction,
                        candidate_action_chunk_reference=visual_candidate_array_reference(
                            candidate.sample_id, "action-chunk"
                        ),
                        action_mask_reference=visual_candidate_array_reference(
                            candidate.sample_id, "action-mask"
                        ),
                        final_success=candidate.final_task_success,
                        final_unsafe=candidate.final_unsafe,
                        failure_events=candidate.failure_events,
                        evidence_id=candidate.strong_simulator_evidence_id,
                        source_dataset_digest=source_binding.dataset_digest,
                        candidate_dataset_digest=source_binding.dataset_digest,
                        visual_dataset_digest=pending_digest,
                        source_collection=context.source_collection,
                    )
                )
    return build_visual_development_dataset(
        source_dataset_digest=source_binding.dataset_digest,
        split_digest=source_binding.split_digest,
        evidence_digest=source_binding.evidence_digest,
        source_compatibility_identity=source_binding.compatibility_identity,
        packets=packets,
        candidate_bindings=bindings,
        samples=samples,
        camera_rig_digest=scope.camera_rig.rig_digest,
        render_domain_configuration_digest=scope.render_domains.content_digest,
        visual_compatibility_identity=visual.visual_compatibility_identity,
        full_target=full_target,
    )


def _build_m3c_visual_dataset(
    plan: _PreparedRenderPlan,
    source_binding: Any,
    scope: M4AStaticScope,
    packets: tuple[Any, ...],
    *,
    full_target: bool,
) -> object:
    from latentguard.vision_data.dataset import build_visual_external_dataset
    from latentguard.vision_data.models import (
        PICKCUBE_CANONICAL_TASK_TEXT,
        PICKCUBE_VISUAL_TASK_ID,
        VisualActionVerifierSampleV1,
        VisualCandidateBindingV1,
    )
    from latentguard.vision_data.source_binding import (
        visual_candidate_array_reference,
    )

    packet_ids = _packet_ids_by_anchor(packets)
    bindings: list[Any] = []
    samples: list[Any] = []
    pending_digest = _semantic_digest(
        {"inventory": plan.inventory.content_digest, "semantic": "pending_v1"},
        context="M4APendingExternalDatasetV1",
    )
    visual = scope.visual_probe_report
    if visual is None:
        _fail("M3C dataset", "trusted visual compatibility is absent")
    for context in plan.contexts:
        group = context.candidate_group
        ids = cast(tuple[str, str, str], packet_ids[context.anchor_id])
        for candidate in context.candidates:
            candidate_id = candidate.proposal_id
            evidence = source_binding.evidence_by_candidate[candidate_id]
            bindings.append(
                VisualCandidateBindingV1(
                    candidate_sample_id=candidate_id,
                    candidate_group_id=group.group_id,
                    anchor_id=context.anchor_id,
                    source_collection=context.source_collection,
                    packet_ids=ids,
                )
            )
            for packet_id in ids:
                samples.append(
                    VisualActionVerifierSampleV1(
                        packet_id=packet_id,
                        candidate_sample_id=candidate_id,
                        task_id=PICKCUBE_VISUAL_TASK_ID,
                        canonical_task_text=PICKCUBE_CANONICAL_TASK_TEXT,
                        candidate_action_chunk_reference=visual_candidate_array_reference(
                            candidate_id, "action-chunk"
                        ),
                        action_mask_reference=visual_candidate_array_reference(
                            candidate_id, "action-mask"
                        ),
                        final_success=cast(bool, evidence.success),
                        final_unsafe=cast(bool, evidence.unsafe),
                        failure_events=evidence.failure_events,
                        evidence_id=evidence.evidence_id,
                        source_dataset_digest=source_binding.source_set_digest,
                        candidate_dataset_digest=(source_binding.candidate_pool_digest),
                        visual_dataset_digest=pending_digest,
                        source_collection=context.source_collection,
                    )
                )
    return build_visual_external_dataset(
        source_set_identity=source_binding.source_set_digest,
        candidate_pool_identity=source_binding.candidate_pool_digest,
        blind_manifest_digest=source_binding.blind_manifest_digest,
        full_outcome_digest=source_binding.full_outcome_digest,
        source_compatibility_identity=(source_binding.simulator_compatibility_identity),
        packets=packets,
        candidate_bindings=bindings,
        samples=samples,
        camera_rig_digest=scope.camera_rig.rig_digest,
        render_domain_configuration_digest=scope.render_domains.content_digest,
        visual_compatibility_identity=visual.visual_compatibility_identity,
        full_target=full_target,
    )


class _PickCubeVisualDatasetRenderer:
    """Production facade joining accepted sources to the generic packet pipeline."""

    def render_m3a(
        self,
        *,
        args: argparse.Namespace,
        scope: M4AStaticScope,
        source_binding: Any,
    ) -> M4ARenderCommandResult:
        contexts, dataset, archive_digest, manifest_digest = _load_m3a_render_contexts(
            args, source_binding
        )
        plan = _build_render_plan(
            args,
            scope,
            source_binding,
            contexts,
            dataset,
            archive_digest,
            manifest_digest,
        )
        return self._execute(
            args,
            scope,
            source_binding,
            plan,
            external=False,
        )

    def render_m3c(
        self,
        *,
        args: argparse.Namespace,
        scope: M4AStaticScope,
        source_binding: Any,
    ) -> M4ARenderCommandResult:
        contexts, pool, archive_digest, manifest_digest = _load_m3c_render_contexts(
            args, source_binding
        )
        plan = _build_render_plan(
            args,
            scope,
            source_binding,
            contexts,
            pool,
            archive_digest,
            manifest_digest,
        )
        return self._execute(
            args,
            scope,
            source_binding,
            plan,
            external=True,
        )

    @staticmethod
    def _execute(
        args: argparse.Namespace,
        scope: M4AStaticScope,
        source_binding: Any,
        plan: _PreparedRenderPlan,
        *,
        external: bool,
    ) -> M4ARenderCommandResult:
        candidate_count = sum(len(context.candidates) for context in plan.contexts)
        selected_packet_count = len(plan.selected_packet_ids)
        if args.dry_run:
            return M4ARenderCommandResult(
                dataset_digest=None,
                packet_count=selected_packet_count,
                image_count=selected_packet_count * 3,
                candidate_count=candidate_count,
                rendered_packet_count=0,
            )
        from latentguard.vision_data.pipeline import run_visual_render_pipeline
        from latentguard.vision_data.run_manifest import (
            M4A_RUN_MANIFEST_FILENAME,
            compute_selected_packet_inventory_digest,
            load_m4a_run_manifest,
        )
        from latentguard.vision_data.serialization import save_visual_dataset

        run_id = (
            "m4a-m3c-" if external else "m4a-m3a-"
        ) + plan.inventory.content_digest[7:23]
        prior_manifest = (
            load_m4a_run_manifest(Path(args.output_root) / M4A_RUN_MANIFEST_FILENAME)
            if args.resume
            else None
        )
        operational_manifest = _build_operational_run_manifest(
            args=args,
            scope=scope,
            source_binding=source_binding,
            run_id=run_id,
            external=external,
            render_seed=args.seed,
            canonical_source_identity_digest=plan.inventory.source_identity_digest,
            render_job_inventory_digest=plan.inventory.content_digest,
            selected_packet_inventory_digest=(
                compute_selected_packet_inventory_digest(plan.selected_packet_ids)
            ),
            prior_manifest=prior_manifest,
        )
        pipeline_result = run_visual_render_pipeline(
            plan.inventory,
            args.output_root,
            run_id=run_id,
            render_callback=_pipeline_callback(plan, scope),
            selected_packet_ids=plan.selected_packet_ids,
            resume=args.resume,
            retry_execution_errors=args.retry_execution_errors,
            operational_manifest=operational_manifest,
        )
        dataset_digest: str | None = None
        if plan.all_required_domains:
            full_target = (
                args.trajectory_limit is None
                and args.anchor_limit is None
                and args.domain_limit is None
            )
            packets = tuple(pipeline_result.packets)
            dataset: Any = (
                _build_m3c_visual_dataset(
                    plan,
                    source_binding,
                    scope,
                    packets,
                    full_target=full_target,
                )
                if external
                else _build_m3a_visual_dataset(
                    plan,
                    source_binding,
                    scope,
                    packets,
                    full_target=full_target,
                )
            )
            save_visual_dataset(
                dataset,
                _packet_images(pipeline_result),
                Path(args.output_root) / "dataset",
            )
            dataset_digest = dataset.content_digest
        zero_work_resume_verified = False
        if pipeline_result.zero_work_proof is not None:
            from latentguard.vision_data.reporting import (
                persist_render_resume_report,
            )

            persist_render_resume_report(
                pipeline_result.zero_work_proof,
                Path(args.output_root),
                source_identity_digest=plan.inventory.source_identity_digest,
                camera_rig_digest=plan.inventory.camera_rig_digest,
                render_domain_configuration_digest=(
                    plan.inventory.render_domain_configuration_digest
                ),
            )
            zero_work_resume_verified = True
        rendered = len(pipeline_result.rendered_packet_ids)
        return M4ARenderCommandResult(
            dataset_digest=dataset_digest,
            packet_count=pipeline_result.packet_count,
            image_count=pipeline_result.image_count,
            candidate_count=candidate_count,
            rendered_packet_count=rendered,
            preserved_packet_count=pipeline_result.packet_count - rendered,
            zero_work_resume_verified=zero_work_resume_verified,
        )


def _default_render_backend() -> M4ARenderBackend:
    """Return the lazy production PickCube-to-core pipeline facade."""

    return _PickCubeVisualDatasetRenderer()


def _load_validation_roots(paths: list[Path]) -> tuple[VisualDataset, ...]:
    if not 1 <= len(paths) <= 2:
        _fail("visual validation", "provide one or two --dataset-dir values")
    absolute = tuple(path.absolute() for path in paths)
    resolved = tuple(path.resolve() for path in absolute)
    if len(set(resolved)) != len(resolved):
        _fail("visual validation", "dataset roots must be distinct")
    from latentguard.vision_data.serialization import load_visual_dataset

    datasets = cast(
        "tuple[VisualDataset, ...]",
        tuple(load_visual_dataset(path) for path in absolute),
    )
    if len(datasets) == 2 and type(datasets[0]) is type(datasets[1]):
        _fail(
            "visual validation",
            "two roots must be one development and one external dataset",
        )
    return datasets


def _namespace_input_paths(
    args: argparse.Namespace, *, excluded_names: frozenset[str]
) -> tuple[Path, ...]:
    """Return every scalar or list-valued path input on one parsed command."""

    paths: list[Path] = []
    for name, value in vars(args).items():
        if name in excluded_names or value is None:
            continue
        if isinstance(value, Path):
            paths.append(value)
            continue
        if isinstance(value, list):
            paths.extend(item for item in value if isinstance(item, Path))
    return tuple(paths)


def _resolved_paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two lexical paths resolve equal or contain one another."""

    left = Path(first).absolute().resolve()
    right = Path(second).absolute().resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _require_new_disjoint_output(
    output: object,
    inputs: tuple[Path, ...],
    *,
    context: str,
) -> Path:
    """Require an absent plain output path disjoint from every input artifact."""

    if not isinstance(output, Path):
        _fail(context, "must be a path")
    destination = output.absolute()
    from latentguard.vision_data.serialization import _is_link_or_junction

    if destination.exists() or _is_link_or_junction(destination):
        _fail(context, "must be absent")
    for input_path in inputs:
        if _resolved_paths_overlap(destination, input_path):
            _fail(context, "must not overlap any input artifact")
    current = destination.parent
    while True:
        if _is_link_or_junction(current):
            _fail(context, "parent cannot traverse a link or junction")
        if current.exists() and not current.is_dir():
            _fail(context, "parent must be a directory")
        if current == current.parent:
            break
        current = current.parent
    return destination


def _run_validate_visual_verifier_dataset(args: argparse.Namespace) -> int:
    """Strictly reload datasets, images, packet bindings, and leakage evidence."""

    report_dir = getattr(args, "report_dir", None)
    if report_dir is not None:
        _require_new_disjoint_output(
            report_dir,
            _namespace_input_paths(
                args,
                excluded_names=frozenset({"report_dir"}),
            ),
            context="visual validation report directory",
        )
        if bool(args.allow_partial):
            _fail(
                "visual validation reports",
                "--report-dir requires full-target validation; --allow-partial "
                "cannot publish acceptance reports",
            )

    from latentguard.vision_data.dataset import validate_visual_verifier_dataset
    from latentguard.vision_data.models import (
        VisualVerifierDevelopmentDatasetV1,
        VisualVerifierExternalDatasetV1,
    )

    scope = _load_static_scope(args, require_visual_probe=True)
    datasets = _load_validation_roots(args.dataset_dir)
    for dataset in datasets:
        validate_visual_verifier_dataset(dataset, full_target=not args.allow_partial)
    development_seeds: tuple[int, ...] | None = None
    external_seeds: tuple[int, ...] | None = None
    for dataset in datasets:
        if isinstance(dataset, VisualVerifierDevelopmentDatasetV1):
            development_seeds = _validate_m3a_visual_source(dataset, args, scope)
        elif isinstance(dataset, VisualVerifierExternalDatasetV1):
            external_seeds = _validate_m3c_visual_source(dataset, args, scope)
        else:  # pragma: no cover - strict loader already rejects unsupported models
            _fail("visual validation", "unsupported visual dataset model")
    leakage = "not_applicable"
    leakage_report: Any | None = None
    if len(datasets) == 2:
        from latentguard.vision_data.leakage import validate_cross_dataset_leakage

        if development_seeds is None or external_seeds is None:
            _fail(
                "visual validation",
                "cross-dataset validation requires both source-bound inventories",
            )

        development = next(
            value
            for value in datasets
            if isinstance(value, VisualVerifierDevelopmentDatasetV1)
        )
        external = next(
            value
            for value in datasets
            if isinstance(value, VisualVerifierExternalDatasetV1)
        )
        leakage_report = validate_cross_dataset_leakage(
            development,
            external,
            development_reset_seeds=development_seeds,
            external_reset_seeds=external_seeds,
        )
        leakage = "passed"
    report_count: int | None = None
    if report_dir is not None:
        from latentguard.training.reporting import StrictReportV1
        from latentguard.vision_data.reporting import (
            M4A_RENDER_RESUME_REPORT_FILENAME,
            build_camera_rig_summary_report,
            build_domain_assignment_report,
            build_external_training_prohibition_report,
            build_image_digest_summary_report,
            build_leakage_report,
            build_render_determinism_report,
            build_render_domain_summary_report,
            build_state_integrity_report_from_dataset,
            build_visual_dataset_summary_report,
            load_bound_render_resume_report,
            publish_visual_validation_reports,
        )

        visual = scope.visual_probe_report
        if visual is None:
            _fail("compact visual reports", "trusted visual probe report is absent")
        repeated = visual.repeated_render
        fresh = visual.fresh_environment_render
        if repeated.spatially_stable != fresh.spatially_stable:
            _fail("compact visual reports", "pixel spatial-stability evidence differs")
        compact_reports: dict[str, StrictReportV1] = {
            "camera-rig-summary.json": build_camera_rig_summary_report(
                scope.camera_rig
            ),
            "render-determinism.json": build_render_determinism_report(
                visual_compatibility_identity=visual.visual_compatibility_identity,
                repeated_changed_pixel_counts=tuple(
                    item.changed_pixel_count for item in repeated.views
                ),
                repeated_maximum_channel_differences=tuple(
                    item.maximum_per_channel_absolute_difference
                    for item in repeated.views
                ),
                repeated_mean_absolute_differences=tuple(
                    item.mean_absolute_pixel_difference for item in repeated.views
                ),
                fresh_environment_changed_pixel_counts=tuple(
                    item.changed_pixel_count for item in fresh.views
                ),
                fresh_environment_maximum_channel_differences=tuple(
                    item.maximum_per_channel_absolute_difference for item in fresh.views
                ),
                fresh_environment_mean_absolute_differences=tuple(
                    item.mean_absolute_pixel_difference for item in fresh.views
                ),
                spatially_stable=repeated.spatially_stable,
            ),
            "render-domain-summary.json": build_render_domain_summary_report(
                scope.render_domains
            ),
        }
        for dataset in datasets:
            if isinstance(dataset, VisualVerifierDevelopmentDatasetV1):
                prefix = "development"
                render_root = getattr(args, "development_render_root", None)
            else:
                prefix = "external"
                render_root = getattr(args, "external_render_root", None)
            compact_reports[f"{prefix}-dataset-summary.json"] = (
                build_visual_dataset_summary_report(dataset)
            )
            compact_reports[f"{prefix}-domain-assignment.json"] = (
                build_domain_assignment_report(dataset)
            )
            compact_reports[f"{prefix}-image-digest-summary.json"] = (
                build_image_digest_summary_report(dataset)
            )
            integrity = build_state_integrity_report_from_dataset(dataset)
            if integrity.payload.get("passed") is not True:
                _fail(
                    f"{prefix} state integrity report",
                    "validated dataset did not project a passing integrity result",
                )
            compact_reports[f"{prefix}-state-integrity.json"] = integrity
            if isinstance(dataset, VisualVerifierExternalDatasetV1):
                compact_reports["external-training-prohibition.json"] = (
                    build_external_training_prohibition_report(dataset)
                )
            if render_root is not None:
                if not isinstance(render_root, Path):
                    _fail(f"{prefix} resume report", "render root must be a path")
                resume_path = render_root / M4A_RENDER_RESUME_REPORT_FILENAME
                if resume_path.exists() or not args.allow_partial:
                    from latentguard.vision_data.pipeline import (
                        load_render_job_inventory,
                    )
                    from latentguard.vision_data.rendering import load_render_ledger

                    inventory = load_render_job_inventory(render_root)
                    ledger = load_render_ledger(render_root)
                    compact_reports[f"{prefix}-resume.json"] = (
                        load_bound_render_resume_report(
                            render_root,
                            run_id=ledger.run_id,
                            source_identity_digest=inventory.source_identity_digest,
                            camera_rig_digest=inventory.camera_rig_digest,
                            render_domain_configuration_digest=(
                                inventory.render_domain_configuration_digest
                            ),
                            ledger_content_digest=ledger.content_digest,
                            packet_count=len(inventory.packet_ids),
                            ledger_entry_count=len(ledger.entries),
                        )
                    )
        if leakage_report is not None:
            compact_reports["cross-dataset-leakage.json"] = build_leakage_report(
                leakage_report
            )
        published = publish_visual_validation_reports(
            compact_reports,
            report_dir,
        )
        report_count = len(published)
    packet_count = sum(len(item.packets) for item in datasets)
    image_count = packet_count * 3
    sample_count = sum(len(item.samples) for item in datasets)
    compact_report_status = (
        str(report_count) if report_count is not None else "not_requested"
    )
    print(
        "validate-visual-verifier-dataset OK: "
        f"datasets={len(datasets)} packets={packet_count} images={image_count} "
        f"samples={sample_count} cross_dataset_leakage={leakage} "
        f"compact_reports={compact_report_status}"
    )
    return 0


def _require_validation_paths(
    args: argparse.Namespace, names: tuple[str, ...], *, context: str
) -> tuple[Path, ...]:
    values: list[Path] = []
    missing: list[str] = []
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            missing.append("--" + name.replace("_", "-"))
        elif not isinstance(value, Path):
            _fail(context, f"{name} must be a path")
        else:
            values.append(value)
    if missing:
        _fail(context, "missing required source inputs: " + ", ".join(missing))
    return tuple(values)


def _matrix_matches_runtime_cast(
    observed: object, planned: object, *, runtime_dtype: str
) -> bool:
    """Apply ``exact_after_runtime_dtype_cast_v1`` to serialized calibration."""

    import numpy as np

    if runtime_dtype not in {"float16", "float32", "float64"}:
        return False
    observed_array = np.asarray(observed, dtype=np.float64)
    planned_array = np.asarray(planned, dtype=np.float64)
    if observed_array.shape != planned_array.shape:
        return False
    projected = np.asarray(planned_array, dtype=np.dtype(runtime_dtype)).astype(
        np.float64
    )
    return bool(np.array_equal(observed_array, projected))


def _validate_render_inventory(
    dataset: VisualDataset,
    render_root: Path,
    scope: M4AStaticScope,
    *,
    expected_source_identity_digest: str,
    expected_reset_seeds_by_anchor: Mapping[str, int],
) -> tuple[int, ...]:
    """Bind packet metadata and reset seeds to the immutable render jobs."""

    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        CALIBRATION_COMPARISON_SEMANTIC,
        build_pickcube_visual_render_plan,
    )
    from latentguard.vision_data.pipeline import load_render_job_inventory

    inventory = load_render_job_inventory(render_root)
    packets = tuple(dataset.packets)
    if inventory.packet_ids != tuple(packet.packet_id for packet in packets):
        _fail("visual render inventory", "packet inventory or ordering differs")
    visual = scope.visual_probe_report
    if visual is None:
        _fail("visual trust roots", "trusted visual probe report is absent")
    if (
        visual.renderer_api.calibration_comparison_semantic
        != CALIBRATION_COMPARISON_SEMANTIC
    ):
        _fail("visual trust roots", "calibration comparison semantic drifted")
    runtime_intrinsics_dtype = visual.renderer_api.runtime_intrinsics_dtype
    runtime_extrinsics_dtype = visual.renderer_api.runtime_extrinsics_dtype
    if runtime_intrinsics_dtype not in {"float16", "float32", "float64"}:
        _fail("visual trust roots", "runtime intrinsics dtype is unsupported")
    if runtime_extrinsics_dtype not in {"float16", "float32", "float64"}:
        _fail("visual trust roots", "runtime extrinsics dtype is unsupported")
    compatibility_identity = scope.compatibility_binding.report.compatibility_identity
    dataset_expected = {
        "camera_rig_digest": scope.camera_rig.rig_digest,
        "render_domain_configuration_digest": scope.render_domains.content_digest,
        "visual_compatibility_identity": visual.visual_compatibility_identity,
        "source_compatibility_identity": compatibility_identity,
    }
    changed_dataset = [
        name
        for name, expected in dataset_expected.items()
        if getattr(dataset, name) != expected
    ]
    if changed_dataset:
        _fail(
            "visual trust roots",
            "dataset differs in " + ", ".join(changed_dataset),
        )
    inventory_expected = {
        "source_collection": dataset.source_collection,
        "source_identity_digest": expected_source_identity_digest,
        "camera_rig_digest": scope.camera_rig.rig_digest,
        "render_domain_configuration_digest": scope.render_domains.content_digest,
        "visual_compatibility_identity": visual.visual_compatibility_identity,
    }
    changed_inventory = [
        name
        for name, expected in inventory_expected.items()
        if getattr(inventory, name) != expected
    ]
    if changed_inventory:
        _fail(
            "visual render inventory",
            "dataset contract differs in " + ", ".join(changed_inventory),
        )
    for job, packet in zip(inventory.jobs, packets, strict=True):
        expected_reset_seed = expected_reset_seeds_by_anchor.get(job.anchor_id)
        if expected_reset_seed is None or job.source_reset_seed != expected_reset_seed:
            _fail(job.packet_id, "source reset seed differs from accepted archive")
        domain = scope.render_domains.domain(job.render_domain_id)
        resolved_plan = build_pickcube_visual_render_plan(
            scope.camera_rig,
            domain,
            scope.render_domains,
            job.render_seed,
        )
        planned_camera_digests = tuple(
            camera.camera_configuration_digest for camera in resolved_plan.cameras
        )
        trusted_job_expected = {
            "camera_rig_id": scope.camera_rig.rig_id,
            "camera_rig_digest": scope.camera_rig.rig_digest,
            "render_domain_digest": domain.domain_digest,
            "camera_configuration_digests": planned_camera_digests,
            "visual_compatibility_identity": visual.visual_compatibility_identity,
            "pickcube_compatibility_identity": compatibility_identity,
            "renderer_semantic_version": visual.renderer_semantic_version,
        }
        changed_job = [
            name
            for name, expected_value in trusted_job_expected.items()
            if getattr(job, name) != expected_value
        ]
        if changed_job:
            _fail(
                job.packet_id,
                "render job differs from trust roots in " + ", ".join(changed_job),
            )
        expected = {
            "packet_id": job.packet_id,
            "source_collection": job.source_collection,
            "source_trajectory_id": job.source_trajectory_id,
            "anchor_id": job.anchor_id,
            "split": job.split,
            "split_group_id": job.split_group_id,
            "state_reference_id": job.state_reference_id,
            "expected_state_digest": job.expected_state_digest,
            "verifier_state_semantic": job.verifier_state_semantic,
            "verifier_state_digest": job.verifier_state_digest,
            "camera_rig_id": job.camera_rig_id,
            "camera_rig_digest": job.camera_rig_digest,
            "render_domain_id": job.render_domain_id,
            "render_domain_digest": job.render_domain_digest,
            "render_seed": job.render_seed,
            "visual_compatibility_identity": job.visual_compatibility_identity,
            "pickcube_compatibility_identity": (job.pickcube_compatibility_identity),
            "renderer_semantic_version": job.renderer_semantic_version,
            "schema_version": job.packet_schema_version,
        }
        changed = [
            name
            for name, expected_value in expected.items()
            if getattr(packet, name) != expected_value
        ]
        if (
            tuple(view.camera_configuration_digest for view in packet.views)
            != job.camera_configuration_digests
        ):
            changed.append("camera_configuration_digests")
        if tuple(view.camera_id for view in packet.views) != tuple(
            camera.camera_id for camera in resolved_plan.cameras
        ):
            changed.append("camera_ids")
        for view, camera in zip(packet.views, resolved_plan.cameras, strict=True):
            if view.intrinsics_dtype != runtime_intrinsics_dtype:
                changed.append(f"{view.camera_id}.intrinsics_dtype")
            if view.extrinsics_dtype != runtime_extrinsics_dtype:
                changed.append(f"{view.camera_id}.extrinsics_dtype")
            if not _matrix_matches_runtime_cast(
                view.intrinsics,
                camera.intrinsics,
                runtime_dtype=runtime_intrinsics_dtype,
            ):
                changed.append(f"{view.camera_id}.intrinsics")
            if not _matrix_matches_runtime_cast(
                view.extrinsics,
                camera.extrinsics,
                runtime_dtype=runtime_extrinsics_dtype,
            ):
                changed.append(f"{view.camera_id}.extrinsics")
        if changed:
            _fail(
                job.packet_id,
                "render job differs from packet in " + ", ".join(changed),
            )
    return tuple(dict.fromkeys(job.source_reset_seed for job in inventory.jobs))


def _validate_m3a_visual_source(
    dataset: VisualDataset, args: argparse.Namespace, scope: M4AStaticScope
) -> tuple[int, ...]:
    """Reload and reconcile every accepted M3A source required by a dataset."""

    paths = _require_validation_paths(
        args,
        (
            "development_render_root",
            "m3a_runtime_archive_dir",
            "m3a_source_dataset_dir",
            "m3a_anchor_manifest_dir",
            "m3a_acceptance_report",
        ),
        context="M3A visual source validation",
    )
    (
        render_root,
        archive_dir,
        source_dataset_dir,
        anchor_manifest_dir,
        acceptance_report,
    ) = paths
    from latentguard.action_verifier import load_action_verifier_dataset
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.vision_data.source_binding import (
        load_m3a_development_source_binding,
        validate_visual_development_source_binding,
    )

    binding = load_m3a_development_source_binding(
        source_dataset_dir,
        archive_dir,
        anchor_manifest_dir,
        acceptance_report=acceptance_report,
    )
    source_dataset = load_action_verifier_dataset(source_dataset_dir)
    manifest = load_anchor_manifest(anchor_manifest_dir)
    archive = load_state_indexed_archive(archive_dir)
    if manifest.source_archive_content_digest != archive.content_digest:
        _fail("M3A visual source validation", "runtime archive digest differs")
    if archive.content_digest != binding.source_archive_digest:
        _fail(
            "M3A visual source validation",
            "runtime archive differs from accepted binding",
        )
    validate_visual_development_source_binding(
        dataset,
        source_dataset,
        manifest,
        binding,
        allow_partial=bool(args.allow_partial),
    )
    source_identity = _semantic_digest(
        {
            "archive_digest": archive.content_digest,
            "manifest_digest": manifest.content_digest,
            "source_binding": {
                "dataset_digest": binding.dataset_digest,
                "source_archive_digest": binding.source_archive_digest,
                "anchor_manifest_digest": binding.anchor_manifest_digest,
                "split_digest": binding.split_digest,
                "evidence_digest": binding.evidence_digest,
                "acceptance_report_digest": binding.acceptance_report_digest,
                "compatibility_identity": binding.compatibility_identity,
            },
            "source_model_digest": source_dataset.content_digest,
        },
        context="M4ARenderSourceIdentityV1",
    )
    return _validate_render_inventory(
        dataset,
        render_root,
        scope,
        expected_source_identity_digest=source_identity,
        expected_reset_seeds_by_anchor={
            record.anchor.anchor_id: record.anchor.source_seed
            for record in manifest.records
        },
    )


def _validate_m3c_visual_source(
    dataset: VisualDataset, args: argparse.Namespace, scope: M4AStaticScope
) -> tuple[int, ...]:
    """Reload and reconcile the complete accepted M3C external source chain."""

    paths = _require_validation_paths(
        args,
        (
            "external_render_root",
            "m3c_runtime_archive_dir",
            "m3c_source_dir",
            "m3c_anchor_manifest_dir",
            "m3c_candidate_pool_dir",
            "m3c_blind_manifest",
            "m3c_bound_result",
            "m3c_corruption_dir",
            "m3c_selected_evaluation_dir",
            "m3c_remainder_evaluation_dir",
        ),
        context="M3C visual source validation",
    )
    (
        render_root,
        archive_dir,
        source_dir,
        anchor_manifest_dir,
        candidate_pool_dir,
        blind_manifest,
        bound_result,
        corruption_dir,
        selected_evaluation_dir,
        remainder_evaluation_dir,
    ) = paths
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.selection.serialization import load_candidate_pool
    from latentguard.vision_data.source_binding import (
        load_m3c_external_source_binding,
        require_m3c_anchor_archive_binding,
        validate_visual_external_source_binding,
    )

    binding = load_m3c_external_source_binding(
        candidate_pool_dir,
        blind_manifest,
        bound_result,
        corruption_dir,
        selected_evaluation_dir,
        remainder_evaluation_dir,
        expected_execution_git_sha=ACCEPTED_M3C_EXECUTION_GIT_SHA,
    )
    require_m3c_anchor_archive_binding(
        binding,
        source_dir,
        archive_dir,
        anchor_manifest_dir,
    )
    pool = load_candidate_pool(candidate_pool_dir)
    archive = load_state_indexed_archive(archive_dir)
    manifest = load_anchor_manifest(anchor_manifest_dir)
    validate_visual_external_source_binding(
        dataset,
        pool,
        manifest,
        archive,
        binding,
        allow_partial=bool(args.allow_partial),
    )
    source_identity = _semantic_digest(
        {
            "archive_digest": archive.content_digest,
            "manifest_digest": manifest.content_digest,
            "source_binding": {
                "source_set_digest": binding.source_set_digest,
                "candidate_pool_digest": binding.candidate_pool_digest,
                "blind_manifest_digest": binding.blind_manifest_digest,
                "bound_result_digest": binding.bound_result_digest,
                "replay_evidence_digest": binding.replay_evidence_digest,
                "evidence_projection_digest": binding.evidence_projection_digest,
                "full_outcome_digest": binding.full_outcome_digest,
            },
            "source_model_digest": pool.content_digest,
        },
        context="M4ARenderSourceIdentityV1",
    )
    return _validate_render_inventory(
        dataset,
        render_root,
        scope,
        expected_source_identity_digest=source_identity,
        expected_reset_seeds_by_anchor={
            record.anchor.anchor_id: record.anchor.source_seed
            for record in manifest.records
        },
    )


def _packet_summary(packet: VisualObservationPacketV1) -> dict[str, object]:
    views = tuple(packet.views)
    return {
        "anchor_id": packet.anchor_id,
        "camera_ids": [view.camera_id for view in views],
        "compared_state_component_count": min(
            view.compared_state_component_count for view in views
        ),
        "expected_state_digest": packet.expected_state_digest,
        "image_count": len(views),
        "packet_content_digest": packet.content_digest,
        "packet_id": packet.packet_id,
        "render_domain_id": packet.render_domain_id,
        "source_collection": packet.source_collection.value,
        "source_trajectory_id": packet.source_trajectory_id,
        "split": packet.split.value,
        "state_integrity_verified": all(
            view.state_before_render_digest == view.state_after_render_digest
            for view in views
        ),
        "training_allowed": packet.source_collection.value == "m3a_development",
        "verifier_state_digest": packet.verifier_state_digest,
    }


def _write_new_text(path: Path, value: str) -> None:
    destination = Path(path)
    from latentguard.vision_data.serialization import _is_link_or_junction

    if destination.exists() or _is_link_or_junction(destination):
        _fail("inspection output", "refuses to overwrite an existing path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_junction(destination.parent) or not destination.parent.is_dir():
        _fail("inspection output", "parent must be a regular directory")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def _run_inspect_visual_packet(args: argparse.Namespace) -> int:
    """Print or save one compact metadata-only packet summary."""

    if args.output is not None:
        _require_new_disjoint_output(
            args.output,
            tuple(
                path
                for path in (args.packet_dir, args.dataset_dir)
                if isinstance(path, Path)
            ),
            context="inspection output",
        )
    if args.packet_dir is not None:
        if args.packet_id is not None:
            _fail("packet inspection", "--packet-id is invalid with --packet-dir")
        from latentguard.vision_data.serialization import load_visual_packet

        packet = cast(VisualObservationPacketV1, load_visual_packet(args.packet_dir))
    else:
        if args.packet_id is None:
            _fail("packet inspection", "--packet-id is required with --dataset-dir")
        from latentguard.vision_data.serialization import load_visual_dataset

        dataset = cast("VisualDataset", load_visual_dataset(args.dataset_dir))
        matches = tuple(
            item for item in dataset.packets if item.packet_id == args.packet_id
        )
        if len(matches) != 1:
            _fail("packet inspection", "packet ID was not found exactly once")
        packet = matches[0]
    payload = (
        json.dumps(_packet_summary(packet), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    if args.output is None:
        print(payload, end="")
    else:
        _write_new_text(args.output, payload)
        print("inspect-visual-packet OK: summary_written=true")
    return 0


def run_m4a_command(
    args: argparse.Namespace,
    *,
    probe_runtime: PickCubeVisualProbeRuntime | None = None,
    render_backend: M4ARenderBackend | None = None,
    operational_runner: OperationalCommandRunner = _run_operational_command,
) -> int | None:
    """Dispatch an M4A command, returning ``None`` for unrelated commands."""

    if args.command not in M4A_COMMANDS:
        return None
    handlers: Mapping[str, Callable[[], int]] = {
        "probe-maniskill-pickcube-visual": lambda: _run_probe_maniskill_pickcube_visual(
            args,
            runtime=probe_runtime,
            operational_runner=operational_runner,
        ),
        "render-m3a-visual-dataset": lambda: _run_render_m3a_visual_dataset(
            args, backend=render_backend
        ),
        "render-m3c-external-visual-dataset": lambda: (
            _run_render_m3c_external_visual_dataset(args, backend=render_backend)
        ),
        "validate-visual-verifier-dataset": lambda: (
            _run_validate_visual_verifier_dataset(args)
        ),
        "inspect-visual-packet": lambda: _run_inspect_visual_packet(args),
    }
    try:
        return handlers[args.command]()
    except KeyboardInterrupt:
        print(
            f"{args.command} interrupted: persisted state may be resumed",
            file=sys.stderr,
        )
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"{args.command} failed: {sanitize_operational_text(error)}",
            file=sys.stderr,
        )
        return 1


__all__ = [
    "M4A_COMMANDS",
    "M4ACommandError",
    "M4ARenderBackend",
    "M4ARenderCommandResult",
    "M4AStaticScope",
    "add_m4a_subparsers",
    "run_m4a_command",
]
