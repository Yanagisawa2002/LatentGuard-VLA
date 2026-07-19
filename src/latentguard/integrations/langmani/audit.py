"""Read-only static audit of a LangMani checkout for the M6A contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

from latentguard.integrations.langmani.synthetic import validate_synthetic_adapter

LANGMANI_REPOSITORY = "Yanagisawa2002/LangMani"
EXPECTED_ACTION_DIMENSION = 8
EXPECTED_CONTROL_MODE = "pd_joint_pos"
EXPECTED_CHUNK_SIZE = 50
EXPECTED_EXECUTION_HORIZON = 10
EXPECTED_STATE_TOLERANCE = 1e-6

_REQUIRED_SOURCE_CHECKS: Mapping[str, tuple[str, ...]] = {
    "src/langmani/environments/specs.py": (
        "class TaskSpec",
        "canonical_v0",
        "langmani-pick-place-task-v0",
    ),
    "src/langmani/environments/task_logic.py": (
        "def evaluate_task_state",
        '"success"',
        '"target_off_table"',
    ),
    "src/langmani/datasets/policy_state.py": (
        "PANDA_POLICY_STATE_COMPONENTS",
        "_POLICY_STATE_SIZE",
    ),
    "src/langmani/datasets/observation_reconstruction.py": (
        "base_camera",
        "pd_joint_pos",
        "set_state_dict",
    ),
    "src/langmani/policies/act_types.py": (
        "ACT_ACTION_COMPONENTS = 8",
        "chunk_size: int = 50",
        "n_action_steps: int = 10",
    ),
    "src/langmani/policies/act_action_bounds.py": (
        "class BoundedActionEnvPostprocessorV0",
        "retain_raw_and_executed_actions",
        "torch.minimum(torch.maximum(action, low), high)",
    ),
    "src/langmani/policies/act_rollout.py": (
        "def reset_policy_state",
        "policy.select_action",
        "action_bound_processor.process",
    ),
    "src/langmani/policies/act_checkpoint.py": (
        "CHECKPOINT_SCHEMA_VERSION",
        "RNG_STATE_FILE",
    ),
    "src/langmani/collection/replay.py": (
        "STATE_ROUND_TRIP_ABS_TOLERANCE = 1e-6",
        "set_state_dict",
        "get_state_dict",
    ),
    "src/langmani/language/dispatcher.py": (
        "class ControllerDispatcher",
        "def dispatch",
    ),
}

_EVIDENCE_FILES = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "environment/environment.yml",
    "docs/ARCHITECTURE.md",
    "docs/ACT_BASELINE_SPEC.md",
    "docs/M42_ORACLE_CONTROL_SPEC.md",
    "docs/M5A_LANGUAGE_ROUTING_SPEC.md",
    *_REQUIRED_SOURCE_CHECKS,
    "src/langmani/environments/pick_place_by_instruction.py",
    "src/langmani/datasets/archive.py",
    "src/langmani/language/router_types.py",
    "tests/unit/test_m3a_replay.py",
    "tests/integration/test_act_rollout.py",
    "tests/unit/test_act_action_bounds.py",
)


class LangManiAuditError(RuntimeError):
    """Raised when the audit cannot establish a trustworthy checkout."""


def _run_git(root: Path, *arguments: str, allow_failure: bool = False) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        if allow_failure:
            return None
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git failure"
        raise LangManiAuditError(f"Git inspection failed: {detail}")
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def discover_langmani_checkout(candidates: Iterable[Path]) -> Path:
    """Select exactly one candidate containing Git metadata and LangMani sources."""

    valid: list[Path] = []
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (
            root.is_dir()
            and (root / ".git").exists()
            and (root / "src/langmani/environments/specs.py").is_file()
            and (root / "pyproject.toml").is_file()
        ):
            valid.append(root)
    unique = tuple(dict.fromkeys(valid))
    if not unique:
        raise LangManiAuditError("no valid LangMani Git checkout was found")
    if len(unique) != 1:
        raise LangManiAuditError("LangMani checkout discovery is ambiguous")
    return unique[0]


def _git_identity(root: Path) -> dict[str, object]:
    head = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "branch", "--show-current")
    status = _run_git(root, "status", "--short")
    upstream = _run_git(root, "rev-parse", "@{upstream}", allow_failure=True)
    remote = _run_git(root, "remote", "get-url", "origin", allow_failure=True)
    assert head is not None and branch is not None and status is not None
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise LangManiAuditError("LangMani HEAD is not a full Git SHA")
    return {
        "repository": LANGMANI_REPOSITORY,
        "branch": branch,
        "head_sha": head,
        "upstream_sha": upstream,
        "head_matches_upstream": upstream == head if upstream is not None else None,
        "working_tree_clean": status == "",
        "origin_matches_expected_repository": bool(
            remote
            and re.search(r"(?:^|[/:])Yanagisawa2002/LangMani(?:\.git)?$", remote)
        ),
        "checkout_locator": "redacted_local_operational_path",
    }


def _source_evidence(root: Path, head: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in _EVIDENCE_FILES:
        path = root / relative
        if not path.is_file():
            continue
        blob = _run_git(root, "rev-parse", f"{head}:{relative}", allow_failure=True)
        records.append(
            {
                "path": relative,
                "git_blob": blob,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _dependencies(root: Path, latentguard_root: Path) -> dict[str, object]:
    langmani = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    latentguard = tomllib.loads(
        (latentguard_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = langmani.get("project", {})
    lg_project = latentguard.get("project", {})
    dependencies = project.get("dependencies", [])
    if not isinstance(project, Mapping) or not isinstance(dependencies, list):
        raise LangManiAuditError("LangMani project dependencies are malformed")
    versions: dict[str, str] = {}
    for raw in dependencies:
        if not isinstance(raw, str):
            continue
        name = re.split(r"[<>=!~\[]", raw, maxsplit=1)[0].lower()
        versions[name] = raw
    python_contract = project.get("requires-python")
    latentguard_python = (
        lg_project.get("requires-python") if isinstance(lg_project, Mapping) else None
    )
    return {
        "langmani_python": python_contract,
        "latentguard_python": latentguard_python,
        "langmani_dependencies": versions,
        "in_process_compatible": False,
        "compatibility_reason": (
            "LangMani pins Python 3.12 and NumPy 2.2.6 while LatentGuard supports "
            "Python 3.11+ "
            "with NumPy <2.0; use a serialized process boundary."
        ),
    }


def _static_contract_checks(root: Path) -> tuple[list[dict[str, object]], list[str]]:
    checks: list[dict[str, object]] = []
    failures: list[str] = []
    for relative, tokens in _REQUIRED_SOURCE_CHECKS.items():
        path = root / relative
        missing = list(tokens)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            missing = [token for token in tokens if token not in content]
        passed = not missing
        checks.append(
            {
                "check_id": relative.replace("/", ":"),
                "passed": passed,
                "missing_tokens": missing,
            }
        )
        if not passed:
            failures.append(relative)
    return checks, failures


def _accepted_artifacts(root: Path) -> dict[str, object]:
    relative_paths = {
        "m4_projection_verification": (
            "outputs/diagnostics/m4/target-m41-f144/verification.json"
        ),
        "m42_verification": "outputs/diagnostics/m42/verification.json",
        "m5a_summary": "outputs/diagnostics/m5a/sealed-final-synced-0c5bb7d/summary.md",
        "m5a_controller_registry": (
            "outputs/diagnostics/m5a/sealed-final-synced-0c5bb7d/controller_registry.json"
        ),
        "m5a_final_result": (
            "outputs/diagnostics/m5a/sealed-final-synced-0c5bb7d/final_result.json"
        ),
    }
    records: dict[str, object] = {}
    for key, relative in relative_paths.items():
        path = root / relative
        records[key] = {
            "present": path.is_file(),
            "relative_path": relative,
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
    registry_path = root / relative_paths["m5a_controller_registry"]
    registry_summary: dict[str, object] = {"controller_count": None, "task_ids": []}
    if registry_path.is_file():
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            entries = raw.get("entries")
            tasks = raw.get("canonical_task_ids")
            registry_summary = {
                "controller_count": len(entries) if isinstance(entries, list) else None,
                "task_ids": tasks if isinstance(tasks, list) else [],
                "registry_fingerprint": raw.get("registry_fingerprint"),
            }
    return {"artifacts": records, "controller_registry": registry_summary}


def _blockers(
    *, dirty: bool, source_failures: Sequence[str], artifact_info: Mapping[str, object]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def add(
        identifier: str,
        severity: str,
        description: str,
        owner: str,
        decision: str,
        risk: str,
    ) -> None:
        records.append(
            {
                "blocker_id": identifier,
                "severity": severity,
                "description": description,
                "evidence": "M6A static source and accepted-artifact audit",
                "owning_repository": owner,
                "required_decision_or_change": decision,
                "m6b_can_proceed": False,
                "workaround_status": "not_authorized",
                "workaround_risk": risk,
            }
        )

    if dirty:
        add(
            "M6A-B001",
            "critical",
            "The authoritative LangMani checkout is dirty.",
            "LangMani",
            "Return the checkout to an explicitly reviewed clean revision "
            "without losing work.",
            "A dirty checkout cannot bind source identity.",
        )
    if source_failures:
        add(
            "M6A-B002",
            "critical",
            "Required LangMani contract source was missing or mismatched.",
            "LangMani",
            "Review the missing source contracts and update the audited adapter "
            "contract.",
            "Guessing a schema can execute the wrong task or action.",
        )
    add(
        "M6A-B003",
        "critical",
        "No M6B candidate source and blind pool identity is frozen.",
        "LatentGuard-VLA",
        "Choose one available source and freeze the outcome-blind pool before "
        "execution.",
        "Ad-hoc candidates would invalidate paired selection evidence.",
    )
    add(
        "M6A-B004",
        "critical",
        "ACT action-queue, processor history, and episode-prefix state have no "
        "serialized policy snapshot.",
        "LangMani",
        "Add or avoid mid-episode policy-state restoration; freeze an "
        "initial-state-only smoke.",
        "Resetting the queue at an intermediate state changes the deployed policy "
        "distribution.",
    )
    add(
        "M6A-B005",
        "major",
        "No compact deployable LangMani verifier vector is frozen.",
        "LatentGuard-VLA",
        "Freeze a minimal allowlist without privileged task or outcome metadata.",
        "Concatenating available fields would leak privileged information.",
    )
    add(
        "M6A-B006",
        "major",
        "Intermediate set_state_dict round trips are audited but paired action "
        "execution from restored states is not accepted evidence.",
        "LangMani",
        "Either validate complete intermediate paired replay or constrain M6B "
        "to seeded initial resets.",
        "Treating restore audit as exact replay would overstate physical evidence.",
    )
    add(
        "M6A-B007",
        "major",
        "LatentGuard and LangMani dependency pins are incompatible in one Python "
        "environment.",
        "LatentGuard-VLA",
        "Use the versioned JSON/process boundary for M6B and verify both environments.",
        "Forcing NumPy or Python pins can invalidate accepted environments.",
    )
    registry = artifact_info.get("controller_registry")
    controller_count = (
        registry.get("controller_count") if isinstance(registry, Mapping) else None
    )
    if controller_count != 6:
        add(
            "M6A-B008",
            "critical",
            "The accepted six-controller registry was not available.",
            "LangMani",
            "Restore or identify the accepted compact M5A controller registry.",
            "A checkpoint chosen without accepted identity is not reproducible.",
        )
    else:
        add(
            "M6A-B009",
            "major",
            "M6B has not frozen one task and its accepted checkpoint entry.",
            "LatentGuard-VLA",
            "Select one canonical task and bind its registry entry before M6B.",
            "Choosing after outcomes would create selection leakage.",
        )
    return records


def _readiness_gates(
    blockers: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blocked = {
        9: ("M6A-B003", "Freeze a blind candidate source."),
        10: ("M6A-B003", "Bind the candidate manifest before outcomes."),
        11: (
            "M6A-B006",
            "Constrain the smoke to initial reset or validate restored execution.",
        ),
        12: (
            "M6A-B004",
            "Freeze initial-state-only handling or serialize policy state.",
        ),
        13: ("M6A-B007", "Use and validate an out-of-process JSON boundary."),
    }
    names = (
        "Exact LangMani Git revision identified",
        "Working tree and accepted result status understood",
        "Task ID and instruction contract frozen",
        "Executable proposal tensor contract frozen",
        "Raw/projected action boundary frozen",
        "Action normalization and bounds frozen",
        "Policy checkpoint identity frozen",
        "Official success and termination frozen",
        "Candidate source for smoke frozen",
        "Outcome blindness enforceable",
        "Simulator state restoration sufficient for planned smoke",
        "Policy state/history sufficient for planned continuation",
        "Cross-project dependency versions compatible",
        "No training required for first smoke",
        "Bounded smoke can run on one GPU",
    )
    blocker_ids = {str(item["blocker_id"]) for item in blockers}
    gates: list[dict[str, object]] = []
    for index, name in enumerate(names, start=1):
        status = "passed"
        remediation = "none"
        evidence = "M6A static code, accepted-report, and synthetic-adapter audit"
        if index in blocked and blocked[index][0] in blocker_ids:
            status = "blocked"
            remediation = blocked[index][1]
        if index == 7 and "M6A-B009" in blocker_ids:
            status = "blocked"
            remediation = "Freeze one task-specific controller registry entry."
        gates.append(
            {
                "gate_id": f"M6B-G{index:02d}",
                "name": name,
                "status": status,
                "evidence": evidence,
                "remediation": remediation,
            }
        )
    return gates


def _optional_import_probe(root: Path) -> dict[str, object]:
    path = root / "src/langmani/environments/specs.py"
    spec = importlib.util.spec_from_file_location(
        "_latentguard_langmani_specs_probe", path
    )
    if spec is None or spec.loader is None:
        raise LangManiAuditError(
            "could not construct approved LangMani schema import probe"
        )
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        task_spec = getattr(module, "TaskSpec", None)
        return {"executed": True, "task_spec_available": isinstance(task_spec, type)}
    finally:
        sys.modules.pop(spec.name, None)


def audit_langmani_contract(
    langmani_root: Path,
    *,
    latentguard_root: Path,
    strict: bool = False,
    allow_import_probe: bool = False,
) -> dict[str, object]:
    """Audit one exact checkout without importing LangMani by default."""

    root = discover_langmani_checkout((langmani_root,))
    identity = _git_identity(root)
    checks, source_failures = _static_contract_checks(root)
    dependencies = _dependencies(root, latentguard_root.resolve())
    artifacts = _accepted_artifacts(root)
    blockers = _blockers(
        dirty=not bool(identity["working_tree_clean"]),
        source_failures=source_failures,
        artifact_info=artifacts,
    )
    gates = _readiness_gates(blockers)
    audit_issues = list(source_failures)
    if not identity["working_tree_clean"]:
        audit_issues.append("dirty_checkout")
    if identity["origin_matches_expected_repository"] is False:
        audit_issues.append("repository_origin_mismatch")
    if identity["head_matches_upstream"] is False:
        audit_issues.append("upstream_sha_mismatch")
    import_probe = (
        _optional_import_probe(root)
        if allow_import_probe
        else {"executed": False, "reason": "disabled_by_default"}
    )
    report: dict[str, object] = {
        "schema_version": "latentguard-m6a-langmani-audit-v1",
        "audit_mode": "static_read_only",
        "strict": strict,
        "passed": not audit_issues,
        "audit_issues": audit_issues,
        "repository_identity": identity,
        "dependency_contract": dependencies,
        "source_checks": checks,
        "source_evidence": _source_evidence(root, str(identity["head_sha"])),
        "accepted_artifact_status": artifacts,
        "observed_contract": {
            "task_count": 6,
            "task_template": "canonical_v0",
            "environment_id": "LangMani-PickPlaceByInstruction-v0",
            "control_mode": EXPECTED_CONTROL_MODE,
            "control_frequency_hz": 20,
            "policy_observation": (
                "base_camera uint8[256,256,3] plus PandaPolicyStateV0 float32[9]"
            ),
            "policy_chunk_shape": [EXPECTED_CHUNK_SIZE, EXPECTED_ACTION_DIMENSION],
            "execution_horizon": EXPECTED_EXECUTION_HORIZON,
            "normalization": "saved LeRobot MEAN_STD processors",
            "bounds_source": "active environment single_action_space",
            "projection": "deterministic per-component min(max(raw, low), high)",
            "proposal_stages": [
                "raw_policy_proposal",
                "projected_executable_proposal",
                "actually_executed_action_sequence",
            ],
            "state_round_trip_tolerance": EXPECTED_STATE_TOLERANCE,
            "replay_readiness": "initial_state_only",
            "policy_state_restoration": "not_supported",
            "official_success": (
                "target in correct bin, no wrong object in target bin, released, "
                "static, on table"
            ),
            "official_failure": (
                "target_off_table only; other unsuccessful endings remain distinct"
            ),
        },
        "import_probe": import_probe,
        "synthetic_adapter": validate_synthetic_adapter(),
        "readiness_gates": gates,
        "blockers": blockers,
        "blocker_counts": {
            severity: sum(item["severity"] == severity for item in blockers)
            for severity in ("critical", "major", "minor")
        },
        "overall_readiness": "blocked",
        "m6b_executed": False,
        "training_executed": False,
        "simulator_executed": False,
    }
    return report


__all__ = [
    "LangManiAuditError",
    "audit_langmani_contract",
    "discover_langmani_checkout",
]
