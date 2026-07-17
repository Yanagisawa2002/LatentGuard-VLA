from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from latentguard.vision_data.models import SourceCollection
from latentguard.vision_data.run_manifest import (
    M4A_RUN_MANIFEST_ARTIFACT_TYPE,
    M4A_RUN_MANIFEST_SCHEMA_VERSION,
    M4AOperationalEnvironmentV1,
    M4AOperationalRunManifestV1,
    M4AOperationalSourceIdentityV1,
    M4ARunManifestError,
    M4AVisualProbeSourceIdentityV1,
    compute_selected_packet_inventory_digest,
    load_m4a_run_manifest,
    persist_m4a_run_manifest,
    require_m4a_run_manifest_resume_match,
    save_m4a_run_manifest,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


_M4A_PATH_OPTIONS = (
    "--acceptance-report",
    "--action-layout",
    "--anchor-manifest-dir",
    "--blind-manifest",
    "--bound-result",
    "--camera-rig",
    "--candidate-pool-dir",
    "--compatibility-report",
    "--corruption-dir",
    "--dataset-dir",
    "--development-render-root",
    "--expected-contract",
    "--external-render-root",
    "--m3a-acceptance-report",
    "--m3a-anchor-manifest-dir",
    "--m3a-runtime-archive-dir",
    "--m3a-source-dataset-dir",
    "--m3c-anchor-manifest-dir",
    "--m3c-blind-manifest",
    "--m3c-bound-result",
    "--m3c-candidate-pool-dir",
    "--m3c-corruption-dir",
    "--m3c-remainder-evaluation-dir",
    "--m3c-runtime-archive-dir",
    "--m3c-selected-evaluation-dir",
    "--m3c-source-dir",
    "--output",
    "--output-root",
    "--packet-dir",
    "--remainder-evaluation-dir",
    "--render-domains",
    "--report-dir",
    "--runtime-archive-dir",
    "--selected-evaluation-dir",
    "--source-dir",
    "--visual-probe-report",
)


def _environment(hostname: str = "m4a-worker-01") -> M4AOperationalEnvironmentV1:
    return M4AOperationalEnvironmentV1.create(
        hostname=hostname,
        python_version="3.11.13",
        gpu_model="NVIDIA GeForce RTX 5090",
        gpu_driver_version="575.64.03",
        cuda_version="12.8",
        pytorch_version="2.7.1+cu128",
        maniskill_version="3.0.1",
        sapien_version="3.0.0.b1",
        renderer_backend="sapien-vulkan",
    )


def _source_m3a(label: str = "m3a") -> M4AOperationalSourceIdentityV1:
    return M4AOperationalSourceIdentityV1.for_m3a(
        _digest(label),
        canonical_source_identity_digest=_digest(f"{label}-canonical-source"),
    )


def _probe_evidence() -> M4AVisualProbeSourceIdentityV1:
    return M4AVisualProbeSourceIdentityV1(
        source_archive_digest=_digest("probe-archive"),
        source_episode_id="probe-episode-0",
        source_episode_content_digest=_digest("probe-episode"),
        source_trajectory_id="probe-trajectory-0",
        source_reset_seed=314159,
        state_index=7,
        state_content_digest=_digest("probe-state-content"),
        expected_state_digest=_digest("probe-state"),
        verifier_state_digest=_digest("probe-verifier-state"),
    )


_UNSET = object()


def _manifest(
    *,
    environment: M4AOperationalEnvironmentV1 | None = None,
    source: M4AOperationalSourceIdentityV1 | None = None,
    run_id: object = "20260718T120000Z_m4a-render_deadbee_seed271828",
    command: str | None = None,
    branch: object = "codex/m4a-multiview-visual-dataset",
    render_seed: int = 271828,
    render_job_inventory_digest: str | None | object = _UNSET,
    selected_packet_inventory_digest: str | None | object = _UNSET,
    launch_argv: tuple[str, ...] = (
        "latentguard",
        "render-m3a-visual-dataset",
        "--seed",
        "271828",
    ),
) -> M4AOperationalRunManifestV1:
    effective_source = _source_m3a() if source is None else source
    probe = effective_source.is_visual_probe
    effective_inventory = (
        None
        if probe and render_job_inventory_digest is _UNSET
        else (
            _digest("render-job-inventory")
            if render_job_inventory_digest is _UNSET
            else render_job_inventory_digest
        )
    )
    effective_selected = (
        None
        if probe and selected_packet_inventory_digest is _UNSET
        else (
            compute_selected_packet_inventory_digest(("packet-0", "packet-1"))
            if selected_packet_inventory_digest is _UNSET
            else selected_packet_inventory_digest
        )
    )
    return M4AOperationalRunManifestV1.create(
        run_id=run_id,
        command=effective_source.required_command if command is None else command,
        git_sha="a" * 40,
        branch=branch,
        started_at="2026-07-18T12:00:00Z",
        render_seed=render_seed,
        launch_argv=launch_argv,
        environment=_environment() if environment is None else environment,
        visual_compatibility_identity=_digest("visual-compatibility"),
        camera_rig_digest=_digest("camera-rig"),
        render_domain_configuration_digest=_digest("render-domains"),
        source_identity=effective_source,
        render_job_inventory_digest=effective_inventory,  # type: ignore[arg-type]
        selected_packet_inventory_digest=effective_selected,  # type: ignore[arg-type]
    )


def test_m3a_manifest_round_trip_sanitizes_launch_secrets_and_paths(
    tmp_path: Path,
) -> None:
    private_output = tmp_path / "private" / "visual-dataset"
    manifest = _manifest(
        launch_argv=(
            "latentguard",
            "render-m3a-visual-dataset",
            "--output-root",
            str(private_output),
            "--password",
            "super-secret-value",
        )
    )
    path = save_m4a_run_manifest(manifest, tmp_path / "run-manifest.json")

    assert load_m4a_run_manifest(path) == manifest
    assert manifest.artifact_type == M4A_RUN_MANIFEST_ARTIFACT_TYPE
    assert manifest.schema_version == M4A_RUN_MANIFEST_SCHEMA_VERSION
    assert manifest.content_digest == manifest.expected_content_digest
    serialized = path.read_text(encoding="utf-8")
    assert str(private_output) not in serialized
    assert "super-secret-value" not in serialized
    assert "--password" not in serialized
    assert "redacted" in serialized.lower()


def test_persisted_launch_argv_redacts_every_m4a_path_option_and_secret(
    tmp_path: Path,
) -> None:
    path_styles = (
        "relative/private/m4a-artifact",
        r"C:\private\m4a-artifact",
        "/srv/private/m4a-artifact",
    )
    raw_paths: list[str] = []
    launch_argv = ["latentguard", "validate-visual-verifier-dataset"]
    for index, option in enumerate(_M4A_PATH_OPTIONS):
        raw_path = f"{path_styles[index % len(path_styles)]}-{index}"
        raw_paths.append(raw_path)
        launch_argv.extend((option, raw_path))
    secret = "m4a-super-secret-value"
    launch_argv.extend(("--password", secret))

    manifest = _manifest(launch_argv=tuple(launch_argv))
    path = save_m4a_run_manifest(manifest, tmp_path / "run-manifest.json")
    loaded = load_m4a_run_manifest(path)
    serialized = path.read_text(encoding="utf-8")

    assert loaded.launch_argv.count("<path>") == len(_M4A_PATH_OPTIONS)
    assert all(option in loaded.launch_argv for option in _M4A_PATH_OPTIONS)
    assert all(raw_path not in serialized for raw_path in raw_paths)
    assert secret not in serialized
    assert "--password" not in serialized
    assert "[REDACTED]" in loaded.launch_argv


def test_m3c_source_variant_records_only_source_set_and_candidate_pool() -> None:
    source = M4AOperationalSourceIdentityV1.for_m3c(
        _digest("source-set"),
        _digest("candidate-pool"),
        canonical_source_identity_digest=_digest("canonical-source-identity"),
    )
    manifest = _manifest(source=source)

    assert source.as_mapping() == {
        "canonical_source_identity_digest": _digest("canonical-source-identity"),
        "m3c_candidate_pool_digest": _digest("candidate-pool"),
        "m3c_source_set_digest": _digest("source-set"),
        "source_collection": "m3c_external",
    }
    assert manifest.source_identity.source_collection is SourceCollection.M3C_EXTERNAL
    assert M4AOperationalRunManifestV1.from_mapping(manifest.as_mapping()) == manifest


def test_visual_probe_source_binds_complete_evidence_without_dataset_identity() -> None:
    evidence = _probe_evidence()
    source = M4AOperationalSourceIdentityV1.for_visual_probe(evidence)
    manifest = _manifest(source=source)

    assert source.as_mapping() == {
        "source_collection": "visual_probe",
        "visual_probe_source": dict(evidence.as_mapping()),
    }
    assert source.source_collection is None
    assert source.is_visual_probe
    assert manifest.command == "probe-maniskill-pickcube-visual"
    assert manifest.render_job_inventory_digest is None
    assert manifest.selected_packet_inventory_digest is None
    assert M4AOperationalRunManifestV1.from_mapping(manifest.as_mapping()) == manifest


def test_core_run_manifest_import_does_not_load_simulator_integration() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    source_root = str(repository_root / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import latentguard.vision_data.run_manifest; "
                "print(int('latentguard.integrations.maniskill_pickcube.visual_probe' "
                "in sys.modules))"
            ),
        ],
        check=False,
        capture_output=True,
        cwd=repository_root,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


def test_visual_probe_source_rejects_missing_mixed_or_private_evidence() -> None:
    source = M4AOperationalSourceIdentityV1.for_visual_probe(_probe_evidence())
    raw = source.as_mapping()
    del raw["visual_probe_source"]
    with pytest.raises(M4ARunManifestError, match="unexpected or missing fields"):
        M4AOperationalSourceIdentityV1.from_mapping(raw)

    mixed = source.as_mapping()
    mixed["m3a_dataset_digest"] = _digest("masquerading-m3a")
    with pytest.raises(M4ARunManifestError, match="unexpected or missing fields"):
        M4AOperationalSourceIdentityV1.from_mapping(mixed)

    with pytest.raises(M4ARunManifestError, match="cannot masquerade"):
        M4AOperationalSourceIdentityV1(
            source_collection=SourceCollection.M3A_DEVELOPMENT,
            visual_probe_source=_probe_evidence(),
            m3a_dataset_digest=_digest("m3a"),
        )
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        M4AOperationalSourceIdentityV1.for_visual_probe(
            replace(_probe_evidence(), source_episode_id="/private/probe/episode")
        )


@pytest.mark.parametrize(
    ("source", "command"),
    (
        (_source_m3a(), "render-m3c-external-visual-dataset"),
        (
            M4AOperationalSourceIdentityV1.for_m3c(
                _digest("source-set"),
                _digest("candidate-pool"),
                canonical_source_identity_digest=_digest("canonical-source-identity"),
            ),
            "render-m3a-visual-dataset",
        ),
        (
            M4AOperationalSourceIdentityV1.for_visual_probe(_probe_evidence()),
            "render-m3a-visual-dataset",
        ),
    ),
)
def test_command_cannot_masquerade_as_another_source_variant(
    source: M4AOperationalSourceIdentityV1, command: str
) -> None:
    with pytest.raises(M4ARunManifestError, match="exact source variant"):
        _manifest(source=source, command=command)


def test_render_inventory_and_probe_absence_are_strictly_variant_bound() -> None:
    with pytest.raises(M4ARunManifestError, match="render_job_inventory_digest"):
        _manifest(render_job_inventory_digest=None)
    with pytest.raises(M4ARunManifestError, match="selected_packet_inventory_digest"):
        _manifest(selected_packet_inventory_digest=None)

    probe = M4AOperationalSourceIdentityV1.for_visual_probe(_probe_evidence())
    with pytest.raises(M4ARunManifestError, match="visual probe cannot carry"):
        _manifest(source=probe, render_job_inventory_digest=_digest("inventory"))
    with pytest.raises(M4ARunManifestError, match="visual probe cannot carry"):
        _manifest(
            source=probe,
            selected_packet_inventory_digest=_digest("selected-packets"),
        )


@pytest.mark.parametrize("render_seed", (-1, 2**32, True))
def test_render_seed_is_strict_uint32(render_seed: object) -> None:
    with pytest.raises(M4ARunManifestError, match="uint32"):
        _manifest(render_seed=render_seed)  # type: ignore[arg-type]


def test_selected_packet_inventory_digest_binds_order_and_rejects_bad_input() -> None:
    first = compute_selected_packet_inventory_digest(("packet-0", "packet-1"))
    assert first == compute_selected_packet_inventory_digest(("packet-0", "packet-1"))
    assert first != compute_selected_packet_inventory_digest(("packet-1", "packet-0"))
    with pytest.raises(M4ARunManifestError, match="cannot be empty"):
        compute_selected_packet_inventory_digest(())
    with pytest.raises(M4ARunManifestError, match="duplicate"):
        compute_selected_packet_inventory_digest(("packet-0", "packet-0"))
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        compute_selected_packet_inventory_digest(("/private/packet",))


@pytest.mark.parametrize(
    "source",
    (
        M4AOperationalSourceIdentityV1,
        None,
    ),
)
def test_source_model_symbol_is_not_accidentally_serializable(source: object) -> None:
    with pytest.raises(M4ARunManifestError, match="source_identity"):
        replace(_manifest(), source_identity=source)  # type: ignore[arg-type]


def test_source_variant_rejects_mixed_or_missing_identities() -> None:
    with pytest.raises(M4ARunManifestError, match="M3A cannot carry M3C"):
        M4AOperationalSourceIdentityV1(
            source_collection=SourceCollection.M3A_DEVELOPMENT,
            canonical_source_identity_digest=_digest("canonical-source-identity"),
            m3a_dataset_digest=_digest("m3a"),
            m3c_source_set_digest=_digest("source-set"),
        )
    with pytest.raises(M4ARunManifestError, match="candidate pool"):
        M4AOperationalSourceIdentityV1(
            source_collection=SourceCollection.M3C_EXTERNAL,
            canonical_source_identity_digest=_digest("canonical-source-identity"),
            m3c_source_set_digest=_digest("source-set"),
        )


def test_direct_models_reject_private_paths_credentials_and_naive_time() -> None:
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        replace(_environment(), hostname=r"C:\Users\private\machine")
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        replace(_manifest(), launch_argv=("--api-key", "cleartext"))
    with pytest.raises(M4ARunManifestError, match="timezone-aware"):
        M4AOperationalRunManifestV1.create(
            run_id="m4a-run",
            command="render-m3a-visual-dataset",
            git_sha="a" * 40,
            branch="codex/m4a",
            started_at="2026-07-18T12:00:00",
            render_seed=271828,
            launch_argv=("latentguard", "render-m3a-visual-dataset"),
            environment=_environment(),
            visual_compatibility_identity=_digest("visual"),
            camera_rig_digest=_digest("rig"),
            render_domain_configuration_digest=_digest("domains"),
            source_identity=_source_m3a(),
            render_job_inventory_digest=_digest("render-job-inventory"),
            selected_packet_inventory_digest=(
                compute_selected_packet_inventory_digest(("packet-0",))
            ),
        )


def test_required_operational_fields_refuse_missing_or_whole_field_redaction() -> None:
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        _environment(r"C:\private\worker")
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        _manifest(run_id=None)
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        _manifest(run_id="/private/run-id")
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        _manifest(branch="password=cleartext")
    with pytest.raises(M4ARunManifestError, match="sanitized"):
        _manifest(command="token=cleartext")


def test_manifest_exact_fields_and_content_digest_detect_tampering(
    tmp_path: Path,
) -> None:
    path = save_m4a_run_manifest(_manifest(), tmp_path / "run-manifest.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(M4ARunManifestError, match="unexpected or missing fields"):
        load_m4a_run_manifest(path)

    raw.pop("unexpected")
    raw["branch"] = "codex/drifted-branch"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(M4ARunManifestError, match="complete content changed"):
        load_m4a_run_manifest(path)


@pytest.mark.parametrize(
    "field",
    (
        "render_seed",
        "render_job_inventory_digest",
        "selected_packet_inventory_digest",
    ),
)
def test_manifest_rejects_missing_render_identity_fields(field: str) -> None:
    raw = _manifest().as_mapping()
    del raw[field]
    with pytest.raises(M4ARunManifestError, match="unexpected or missing fields"):
        M4AOperationalRunManifestV1.from_mapping(raw)


def test_manifest_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    path.write_text('{"run_id":"first","run_id":"second"}', encoding="utf-8")
    with pytest.raises(M4ARunManifestError, match="duplicate fields"):
        load_m4a_run_manifest(path)


def test_save_is_transactional_new_only_and_rejects_hardlinks(tmp_path: Path) -> None:
    manifest = _manifest()
    path = save_m4a_run_manifest(manifest, tmp_path / "run-manifest.json")
    before = path.read_bytes()

    with pytest.raises(M4ARunManifestError, match="refusing to overwrite"):
        save_m4a_run_manifest(manifest, path)
    assert path.read_bytes() == before

    linked = tmp_path / "hard-linked-manifest.json"
    os.link(path, linked)
    with pytest.raises(M4ARunManifestError, match="hard-linked"):
        load_m4a_run_manifest(linked)


def test_atomic_save_failure_leaves_no_manifest_or_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.vision_data.run_manifest as run_manifest_module

    destination = tmp_path / "run-manifest.json"

    def fail_link(_source: object, _destination: object) -> None:
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(run_manifest_module.os, "link", fail_link)
    with pytest.raises(M4ARunManifestError, match="atomic create failed"):
        save_m4a_run_manifest(_manifest(), destination)
    assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_manifest_load_rejects_reparse_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.vision_data.serialization as visual_serialization

    path = save_m4a_run_manifest(_manifest(), tmp_path / "run-manifest.json")
    real_guard = visual_serialization._is_link_or_junction
    monkeypatch.setattr(
        visual_serialization,
        "_is_link_or_junction",
        lambda candidate: Path(candidate) == path or real_guard(Path(candidate)),
    )
    with pytest.raises(M4ARunManifestError, match="non-linked"):
        load_m4a_run_manifest(path)


def test_resume_is_zero_write_and_rejects_environment_or_source_drift(
    tmp_path: Path,
) -> None:
    path = save_m4a_run_manifest(_manifest(), tmp_path / "run-manifest.json")
    before = path.read_bytes()
    prior_mtime = path.stat().st_mtime_ns

    assert persist_m4a_run_manifest(_manifest(), path, resume=True) == path
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == prior_mtime

    environment_drift = _manifest(environment=_environment("m4a-worker-02"))
    with pytest.raises(M4ARunManifestError, match="operational identity drift"):
        require_m4a_run_manifest_resume_match(path, environment_drift)
    source_drift = _manifest(source=_source_m3a("different-m3a"))
    with pytest.raises(M4ARunManifestError, match="operational identity drift"):
        require_m4a_run_manifest_resume_match(path, source_drift)
    seed_drift = _manifest(render_seed=271829)
    with pytest.raises(M4ARunManifestError, match="operational identity drift"):
        require_m4a_run_manifest_resume_match(path, seed_drift)
    inventory_drift = _manifest(
        render_job_inventory_digest=_digest("different-render-inventory")
    )
    with pytest.raises(M4ARunManifestError, match="operational identity drift"):
        require_m4a_run_manifest_resume_match(path, inventory_drift)
    selected_drift = _manifest(
        selected_packet_inventory_digest=compute_selected_packet_inventory_digest(
            ("packet-0",)
        )
    )
    with pytest.raises(M4ARunManifestError, match="operational identity drift"):
        require_m4a_run_manifest_resume_match(path, selected_drift)


def test_operational_fields_change_only_operational_manifest_digest() -> None:
    first = _manifest(environment=_environment("m4a-worker-01"))
    second = _manifest(environment=_environment("m4a-worker-02"))

    assert first.source_identity == second.source_identity
    assert first.camera_rig_digest == second.camera_rig_digest
    assert first.render_domain_configuration_digest == (
        second.render_domain_configuration_digest
    )
    assert first.content_digest != second.content_digest
