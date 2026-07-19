"""M6A.1 initial-state LangMani bridge and bounded M6B freeze contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.registry import (
    create_corruption,
    default_corruption_registry,
)
from latentguard.models import ActionChunk
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.checkpoint_bundle import (
    LoadedVerifierBundleV1,
    load_verifier_bundle,
)
from latentguard.selection.inference import (
    NormalizedCandidateBatchV1,
    infer_prepared_seed_logits_once,
)
from latentguard.selection.preparation import build_verifier_bundle_identities
from latentguard.training.config import ModelType
from latentguard.training.dataset import (
    ACCEPTED_ACTION_DIMENSION,
    ACCEPTED_ACTION_HORIZON,
    ACCEPTED_STATE_DIMENSION,
)
from latentguard.training.reporting import load_strict_report

LANGMANI_POLICY_BINDING_SCHEMA = "LangManiLatentGuardPolicyBindingV1"
LANGMANI_INITIAL_PROPOSAL_SCHEMA = "LangManiLatentGuardInitialProposalV1"
RAW_CANDIDATE_REQUEST_SCHEMA = "LatentGuardRawCandidateRequestV1"
LANGMANI_PROJECTED_CANDIDATES_SCHEMA = "LangManiProjectedCandidatesV1"
SMOKE_CONTRACT_SCHEMA = "LatentGuardLangManiBoundedSmokeContractV1"
SCORE_RESULT_SCHEMA = "LatentGuardLangManiActionOnlyScoreV1"
READINESS_SCHEMA = "LatentGuardLangManiM6BReadinessV1"
CANDIDATE_SOURCE_SEMANTIC = "ProjectedPrefixPerturbationPoolV1"
VERIFIER_INPUT_SEMANTIC = "LangManiActionOnlyVerifierInputV1"
MASK_SEMANTIC = "ten_true_then_six_false_zero_padding_v1"
MODEL_EQUIVALENCE_SEMANTIC = "bitwise_equal_float64_logits_and_probabilities_v1"
FIXED_CONTINUATION_SEMANTIC = (
    "candidate_prefix_10_then_recorded_projected_nominal_from_index_10_v1"
)
INITIAL_RESET_SEMANTIC = "seeded_initial_reset_only_no_policy_state_restore_v1"
PROBE_SEED_SEMANTIC = "sha256_prefix_u31_m6a1_bridge_probe_v1"
M6B_SOURCE_SEED_SEMANTIC = "sha256_prefix_u31_m6b_bounded_source_seed_index_v1"
M5_RELEASE_MANIFEST_SHA256 = (
    "sha256:4023dad2253156b56fc094ddee2e1e99afce5ee8843a3caf27bbaafc6ce07377"
)
ACCEPTED_ACTION_ONLY_BUNDLE_DIGEST = (
    "sha256:f4c28ec414e7b68abb4f49cd395572a2f52f1ee6e6fc110d41c13ffbcc30affb"
)
EXPECTED_PREFIX_HORIZON = 10
EXPECTED_CANDIDATE_COUNT = 4
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_FROZEN_CONFIGURATIONS: Mapping[str, Mapping[str, object]] = {
    "additive_gaussian_noise": {
        "mean": 0.0,
        "standard_deviation": 0.05,
        "target_indices": list(range(8)),
    },
    "constant_bias": {"bias": 0.05, "target_indices": list(range(8))},
    "local_temporal_permutation": {
        "end_step": 10,
        "start_step": 0,
        "target_indices": list(range(8)),
    },
    "segment_hold": {
        "end_step": 5,
        "start_step": 2,
        "target_indices": list(range(8)),
    },
    "segment_zeroing": {
        "end_step": 5,
        "start_step": 2,
        "target_indices": list(range(8)),
    },
    "temporal_field_shift": {
        "fill_policy": "edge",
        "shift_steps": 1,
        "target_field": "all_components",
    },
}


class M6A1BridgeError(RuntimeError):
    """Raised when an M6A.1 bridge or readiness gate differs."""


def _fail(context: str, reason: str) -> NoReturn:
    raise M6A1BridgeError(f"{context}: {reason}")


def _exact_mapping(
    value: object, *, fields: set[str], context: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected an object with text keys")
    result = cast(Mapping[str, object], value)
    if set(result) != fields:
        _fail(context, "unexpected or missing fields")
    return result


def _digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        _fail(context, "expected a prefixed SHA-256 digest")
    return value


def content_digest(value: object) -> str:
    """Return a path-free canonical SHA-256 digest."""

    encoded = canonical_json_bytes(value, context="M6A.1 JSON")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def array_content_digest(value: NDArray[Any]) -> str:
    """Match the bridge's dtype-, shape-, and byte-bound array identity."""

    array = np.ascontiguousarray(value)
    byte_digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return content_digest(
        {
            "bytes_sha256": f"sha256:{byte_digest}",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    )


def make_envelope(
    schema_version: str, payload: Mapping[str, object]
) -> dict[str, object]:
    """Create the same strict canonical JSON envelope used by LangMani."""

    body = json.loads(canonical_json_bytes(dict(payload), context=schema_version))
    return {
        "content_digest": content_digest(
            {"payload": body, "schema_version": schema_version}
        ),
        "payload": body,
        "schema_version": schema_version,
    }


def validate_envelope(value: object, *, expected_schema: str) -> Mapping[str, object]:
    """Reject unknown envelope fields, semantic drift, and digest drift."""

    if not isinstance(value, Mapping) or set(value) != {
        "content_digest",
        "payload",
        "schema_version",
    }:
        _fail("bridge envelope", "unexpected or missing fields")
    if value["schema_version"] != expected_schema:
        _fail("bridge envelope.schema_version", "unexpected schema")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        _fail("bridge envelope.payload", "expected an object")
    expected = content_digest(
        {"payload": dict(payload), "schema_version": expected_schema}
    )
    if value["content_digest"] != expected:
        _fail("bridge envelope.content_digest", "content changed")
    return cast(Mapping[str, object], payload)


def read_envelope(path: Path, *, expected_schema: str) -> dict[str, object]:
    """Read one duplicate-free, finite canonical bridge envelope."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                _fail("JSON", f"duplicate field {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> NoReturn:
        _fail("JSON", f"non-finite constant {value!r}")

    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M6A1BridgeError(f"cannot read bridge JSON: {error}") from error
    if not isinstance(raw, dict):
        _fail("bridge JSON", "expected one object")
    validate_envelope(raw, expected_schema=expected_schema)
    return raw


def write_envelope(path: Path, envelope: Mapping[str, object]) -> None:
    """Write one canonical UTF-8 JSON message."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        canonical_json_bytes(dict(envelope), context="M6A.1 output") + b"\n"
    )


def _git_identity(root: Path, *, require_clean: bool) -> tuple[str, str]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            _fail("Git", result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    sha = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    if _SHA_RE.fullmatch(sha) is None:
        _fail("Git", "HEAD is not a full SHA")
    if require_clean and run("status", "--short"):
        _fail("Git", "operation requires a clean checkout")
    return sha, branch


def derive_probe_seed() -> int:
    """Return the shared deterministic bridge-only probe seed."""

    return int.from_bytes(
        hashlib.sha256(PROBE_SEED_SEMANTIC.encode()).digest()[:4], "big"
    ) & (2**31 - 1)


def m6b_source_seeds() -> tuple[int, int, int]:
    """Return three untouched M6B seeds without consulting outcomes."""

    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"sha256_prefix_u31_m6b_bounded_source_seed_{index}_v1".encode()
            ).digest()[:4],
            "big",
        )
        & (2**31 - 1)
        for index in range(3)
    )  # type: ignore[return-value]


def _layout() -> ActionLayout:
    return ActionLayout(
        action_dim=8,
        fields=(
            ActionField(
                name="all_components",
                indices=tuple(range(8)),
                semantic=ActionSemantic.UNSPECIFIED,
            ),
        ),
        description="LangMani pd_joint_pos eight-component bridge layout",
    )


def frozen_candidate_source() -> dict[str, object]:
    """Enumerate eligible existing corruptions and freeze the first three IDs."""

    layout = _layout()
    fixture = ActionChunk(
        actions=np.zeros((10, 8), dtype=np.float32),
        coordinate_frame="langmani_pd_joint_pos_action_targets_v1",
        control_period_s=0.05,
    )
    eligible: list[str] = []
    for name in sorted(default_corruption_registry().names):
        parameters = _FROZEN_CONFIGURATIONS.get(name)
        if parameters is None:
            continue
        try:
            corruption = create_corruption(name, parameters)
            corruption.validate_for(fixture, layout)
            transformed = corruption.apply(fixture, layout, seed=0)
        except (TypeError, ValueError):
            continue
        if (
            transformed.actions.shape == (10, 8)
            and transformed.actions.dtype == np.dtype("<f4")
            and bool(np.all(np.isfinite(transformed.actions)))
        ):
            eligible.append(name)
    selected = tuple(eligible[:3])
    if len(selected) != 3:
        _fail("candidate source", "fewer than three existing corruptions are eligible")
    configs = []
    for ordinal, name in enumerate(selected, start=1):
        parameters = dict(_FROZEN_CONFIGURATIONS[name])
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
        config = {
            "configuration_digest": content_digest(
                {"parameters": parameters, "seed": seed, "transformation_id": name}
            ),
            "parameters": parameters,
            "seed": seed,
            "transformation_id": name,
            "candidate_ordinal": ordinal,
        }
        configs.append(config)
    payload = {
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "eligible_transformation_ids": eligible,
        "identity_candidate_ordinal": 0,
        "selected_transformations": configs,
        "selection_semantic": "lexically_first_three_eligible_existing_corruptions_v1",
        "source_semantic": CANDIDATE_SOURCE_SEMANTIC,
    }
    return {**payload, "candidate_source_digest": content_digest(payload)}


def _proposal_prefix(proposal: Mapping[str, object]) -> NDArray[np.float32]:
    raw = proposal.get("raw_postprocessed_chunk")
    try:
        array = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise M6A1BridgeError("initial proposal raw chunk is malformed") from error
    if array.shape != (50, 8) or not bool(np.all(np.isfinite(array))):
        _fail("initial proposal", "raw chunk must be finite float32[50,8]")
    prefix = np.ascontiguousarray(array[:10], dtype=np.float32)
    if array_content_digest(prefix) != proposal.get("raw_prefix_digest"):
        _fail("initial proposal", "raw prefix digest differs")
    return prefix


def build_raw_candidates(
    proposal_envelope: Mapping[str, object],
    *,
    candidate_source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build four outcome-free raw prefixes with opaque content-bound IDs."""

    proposal = validate_envelope(
        proposal_envelope, expected_schema=LANGMANI_INITIAL_PROPOSAL_SCHEMA
    )
    if (
        proposal.get("environment_step_count") != 0
        or proposal.get("outcomes_generated") is not False
        or proposal.get("zero_action_execution") is not True
    ):
        _fail("initial proposal", "proposal is not zero-action/outcome-free evidence")
    source = dict(candidate_source or frozen_candidate_source())
    expected_source = frozen_candidate_source()
    if source != expected_source:
        _fail("candidate source", "configuration differs from the frozen source")
    prefix = _proposal_prefix(proposal)
    layout = _layout()
    source_chunk = ActionChunk(
        actions=prefix,
        coordinate_frame="langmani_pd_joint_pos_action_targets_v1",
        control_period_s=0.05,
    )
    transformed: list[tuple[str, str, NDArray[np.float32]]] = [
        ("identity", content_digest({"identity": True}), prefix)
    ]
    configurations = source["selected_transformations"]
    assert isinstance(configurations, list)
    for item in configurations:
        assert isinstance(item, Mapping)
        name = cast(str, item["transformation_id"])
        parameters = cast(Mapping[str, object], item["parameters"])
        seed = cast(int, item["seed"])
        corruption = create_corruption(name, parameters)
        result = corruption.apply(source_chunk, layout, seed=seed)
        values = np.ascontiguousarray(result.actions, dtype=np.float32)
        if values.shape != (10, 8) or not bool(np.all(np.isfinite(values))):
            _fail("candidate source", f"{name} produced an invalid candidate")
        transformed.append((name, cast(str, item["configuration_digest"]), values))
    pool_identity = {
        "candidate_source_digest": source["candidate_source_digest"],
        "policy_binding_digest": proposal["policy_binding_digest"],
        "proposal_digest": proposal_envelope["content_digest"],
        "raw_candidate_digests": [
            array_content_digest(item[2]) for item in transformed
        ],
    }
    pool_digest = content_digest(pool_identity)
    candidates = []
    for ordinal, (name, config_digest, actions) in enumerate(transformed):
        candidate_id = (
            "lgc-sha256-"
            + hashlib.sha256(
                canonical_json_bytes(
                    {"candidate_pool_digest": pool_digest, "ordinal": ordinal},
                    context="opaque candidate ID",
                )
            ).hexdigest()
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "raw_actions": actions.tolist(),
                "raw_candidate_digest": array_content_digest(actions),
                "transformation_config_digest": config_digest,
                "transformation_id": name,
            }
        )
    payload = {
        "action_dimension": 8,
        "candidate_count": 4,
        "candidate_pool_digest": pool_digest,
        "candidate_source_digest": source["candidate_source_digest"],
        "candidates": candidates,
        "environment_actions_executed": False,
        "outcomes_available": False,
        "policy_binding_digest": proposal["policy_binding_digest"],
        "proposal_digest": proposal_envelope["content_digest"],
        "source_horizon": 10,
    }
    return make_envelope(RAW_CANDIDATE_REQUEST_SCHEMA, payload)


def _projected_inputs(
    projected_envelope: Mapping[str, object],
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.bool_]]:
    payload = validate_envelope(
        projected_envelope, expected_schema=LANGMANI_PROJECTED_CANDIDATES_SCHEMA
    )
    _exact_mapping(
        payload,
        fields={
            "action_bounds_digest",
            "candidate_count",
            "candidate_pool_digest",
            "candidates",
            "environment_step_count",
            "outcomes_available",
            "policy_binding_digest",
            "projection_semantic",
            "raw_candidate_manifest_digest",
            "task_id",
            "zero_action_execution",
        },
        context="projected candidates",
    )
    _digest(payload.get("candidate_pool_digest"), context="projected candidate pool")
    if (
        payload.get("candidate_count") != 4
        or payload.get("environment_step_count") != 0
        or payload.get("outcomes_available") is not False
        or payload.get("zero_action_execution") is not True
    ):
        _fail("projected candidates", "zero-action four-candidate gate failed")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates, str | bytes
    ):
        _fail("projected candidates", "expected a candidate array")
    ids: list[str] = []
    actions = np.zeros((4, 16, 8), dtype=np.float64)
    masks = np.zeros((4, 16), dtype=np.bool_)
    masks[:, :10] = True
    for index, raw in enumerate(raw_candidates):
        raw = _exact_mapping(
            raw,
            fields={
                "candidate_id",
                "correction_count",
                "correction_mask",
                "nonfinite_count",
                "projected_actions",
                "projected_candidate_digest",
                "projection_semantic",
                "raw_candidate_digest",
                "transformation_config_digest",
                "transformation_id",
            },
            context=f"projected candidate[{index}]",
        )
        candidate_id = raw.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            _fail("projected candidates", "candidate ID is invalid")
        values = np.asarray(raw.get("projected_actions"), dtype=np.float32)
        if values.shape != (10, 8) or not bool(np.all(np.isfinite(values))):
            _fail("projected candidates", "projected actions must be finite [10,8]")
        if array_content_digest(values) != raw.get("projected_candidate_digest"):
            _fail("projected candidates", "projected candidate digest differs")
        mask = np.asarray(raw.get("correction_mask"), dtype=np.bool_)
        if mask.shape != (10, 8):
            _fail("projected candidates", "correction mask must have shape [10,8]")
        if (
            raw.get("correction_count") != int(mask.sum())
            or raw.get("nonfinite_count") != 0
        ):
            _fail("projected candidates", "projection count or finite gate differs")
        ids.append(candidate_id)
        actions[index, :10] = values.astype(np.float64)
    if len(ids) != 4 or len(set(ids)) != 4:
        _fail("projected candidates", "expected four unique candidate IDs")
    return tuple(ids), np.ascontiguousarray(actions), np.ascontiguousarray(masks)


def _prepared_action_only_batch(
    loaded: LoadedVerifierBundleV1,
    candidate_ids: tuple[str, ...],
    actions: NDArray[np.float64],
    masks: NDArray[np.bool_],
) -> NormalizedCandidateBatchV1:
    normalized = (
        actions.astype(np.float64, copy=True) - loaded.preprocessing.action_mean
    ) / loaded.preprocessing.action_standard_deviation
    normalized[~masks] = 0.0
    return NormalizedCandidateBatchV1(
        candidate_ids=candidate_ids,
        state_vectors=np.zeros(
            (len(candidate_ids), ACCEPTED_STATE_DIMENSION), dtype=np.float32
        ),
        action_chunks=np.asarray(normalized, dtype=np.float32, order="C"),
        action_masks=np.asarray(masks, dtype=np.bool_, order="C"),
    )


def _score_loaded(
    loaded: LoadedVerifierBundleV1,
    candidate_ids: tuple[str, ...],
    actions: NDArray[np.float64],
    masks: NDArray[np.bool_],
    *,
    device: str,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], tuple[str, ...]
]:
    batch = _prepared_action_only_batch(loaded, candidate_ids, actions, masks)
    logits: list[NDArray[np.float64]] = []
    probabilities: list[NDArray[np.float64]] = []
    for seed in loaded.seeds:
        raw = infer_prepared_seed_logits_once(seed.model, batch, device=device)
        calibrated = seed.calibration.apply(raw)
        logits.append(raw)
        probabilities.append(calibrated)
    raw_matrix = np.stack(logits).astype(np.float64, copy=False)
    probability_matrix = np.stack(probabilities).astype(np.float64, copy=False)
    mean = np.mean(probability_matrix, axis=0, dtype=np.float64)
    ranking = tuple(
        candidate_id
        for _, candidate_id in sorted(
            zip(mean.tolist(), candidate_ids, strict=True),
            key=lambda item: (item[0], item[1]),
        )
    )
    return raw_matrix, probability_matrix, mean, ranking


def score_projected_candidates(
    *,
    projected_envelope: Mapping[str, object],
    m3b_result_root: Path,
    m3b_runtime_root: Path,
    device: str,
) -> dict[str, object]:
    """Run only the accepted action-only five-seed ensemble and mask gate."""

    ids, actions, masks = _projected_inputs(projected_envelope)
    bundles, paths, selection_digest, preprocessing_digest = (
        build_verifier_bundle_identities(m3b_result_root, runtime_root=m3b_runtime_root)
    )
    bundle = bundles[ModelType.ACTION_ONLY_MLP.value]
    if bundle.content_digest != ACCEPTED_ACTION_ONLY_BUNDLE_DIGEST:
        _fail("action-only scorer", "accepted bundle identity differs")
    loaded = load_verifier_bundle(
        bundle, paths[ModelType.ACTION_ONLY_MLP.value], device=device
    )
    logits, probabilities, mean, ranking = _score_loaded(
        loaded, ids, actions, masks, device=device
    )
    altered = np.array(actions, copy=True)
    deterministic_padding = np.arange(4 * 6 * 8, dtype=np.float64).reshape(4, 6, 8)
    altered[:, 10:] = deterministic_padding + 1.0
    alt_logits, alt_probabilities, alt_mean, alt_ranking = _score_loaded(
        loaded, ids, altered, masks, device=device
    )
    mask_invariant = (
        np.array_equal(logits, alt_logits)
        and np.array_equal(probabilities, alt_probabilities)
        and np.array_equal(mean, alt_mean)
        and ranking == alt_ranking
    )
    if not mask_invariant:
        _fail("mask invariance", "masked finite padding changed accepted scorer output")
    payload = {
        "action_only_allowlist": ["projected_executable_actions", "action_mask"],
        "blind_ranking": list(ranking),
        "candidate_pool_digest": validate_envelope(
            projected_envelope, expected_schema=LANGMANI_PROJECTED_CANDIDATES_SCHEMA
        )["candidate_pool_digest"],
        "candidate_ids": list(ids),
        "ensemble_failure_probabilities": mean.tolist(),
        "mask_invariance": {
            "comparison_semantic": MODEL_EQUIVALENCE_SEMANTIC,
            "deterministic_nonzero_padding_checked": True,
            "passed": True,
        },
        "outcomes_available_during_selection": False,
        "per_seed_calibrated_failure_probabilities": probabilities.tolist(),
        "per_seed_raw_logits": logits.tolist(),
        "preprocessing_digest": preprocessing_digest,
        "projected_candidate_manifest_digest": projected_envelope["content_digest"],
        "scorer_bundle_digest": bundle.content_digest,
        "seed_order": [seed.identity.seed for seed in loaded.seeds],
        "selected_candidate_id": ranking[0],
        "selection_record_digest": selection_digest,
        "verifier_input": {
            "action_dimension": ACCEPTED_ACTION_DIMENSION,
            "dtype": "float32",
            "mask_semantic": MASK_SEMANTIC,
            "model_horizon": ACCEPTED_ACTION_HORIZON,
            "source_executable_horizon": EXPECTED_PREFIX_HORIZON,
            "state_input_source": (
                "constant_internal_placeholder_ignored_by_action_only_model"
            ),
            "verifier_input_semantic": VERIFIER_INPUT_SEMANTIC,
        },
    }
    return make_envelope(SCORE_RESULT_SCHEMA, payload)


def scorer_identity(selector_configuration: Path) -> dict[str, object]:
    """Load the committed accepted action-only identity without runtime weights."""

    report = load_strict_report(
        selector_configuration, expected_report_type="m3c_selector_configuration_v1"
    )
    digests = report.payload.get("verifier_bundle_digests")
    if not isinstance(digests, Mapping):
        _fail("selector configuration", "bundle digest inventory is absent")
    digest = digests.get(ModelType.ACTION_ONLY_MLP.value)
    if digest != ACCEPTED_ACTION_ONLY_BUNDLE_DIGEST:
        _fail("selector configuration", "action-only bundle identity differs")
    return {
        "architecture": ModelType.ACTION_ONLY_MLP.value,
        "bundle_digest": digest,
        "ensemble_seed_count": 5,
        "selector_configuration_report_digest": report.content_digest,
        "task_distribution_status": (
            "langmani_task_out_of_distribution_integration_smoke_only"
        ),
    }


def freeze_smoke_contract(
    *,
    binding_envelope: Mapping[str, object],
    selector_configuration: Path,
    repository_root: Path,
    expected_langmani_sha: str,
) -> dict[str, object]:
    """Freeze all bounded M6B choices before the proposal or any outcomes."""

    binding = validate_envelope(
        binding_envelope, expected_schema=LANGMANI_POLICY_BINDING_SCHEMA
    )
    latentguard_sha, latentguard_branch = _git_identity(
        repository_root, require_clean=True
    )
    if binding.get("bridge_git_commit") != expected_langmani_sha:
        _fail("policy binding", "LangMani implementation SHA differs")
    source = frozen_candidate_source()
    payload = {
        "blindness_requirements": {
            "candidate_manifest_before_outcomes": True,
            "outcome_inputs_prohibited": True,
            "reporting_metadata_inputs_prohibited": True,
        },
        "candidate_count": 4,
        "candidate_source": source,
        "cross_process_schemas": {
            "initial_proposal": LANGMANI_INITIAL_PROPOSAL_SCHEMA,
            "policy_binding": LANGMANI_POLICY_BINDING_SCHEMA,
            "projected_candidates": LANGMANI_PROJECTED_CANDIDATES_SCHEMA,
            "raw_candidates": RAW_CANDIDATE_REQUEST_SCHEMA,
        },
        "expected_m6b_candidate_outcome_count": 12,
        "expected_m6b_source_count": 3,
        "fixed_continuation_semantic": FIXED_CONTINUATION_SEMANTIC,
        "initial_reset_semantic": INITIAL_RESET_SEMANTIC,
        "langmani_sha": expected_langmani_sha,
        "latentguard_branch": latentguard_branch,
        "latentguard_sha": latentguard_sha,
        "m6b_source_seed_schedule": {
            "seeds": list(m6b_source_seeds()),
            "semantic": M6B_SOURCE_SEED_SEMANTIC,
        },
        "m6b_status": "conditionally_ready",
        "policy_binding": dict(binding),
        "policy_binding_digest": binding_envelope["content_digest"],
        "probe_seed": derive_probe_seed(),
        "probe_seed_semantic": PROBE_SEED_SEMANTIC,
        "scorer": scorer_identity(selector_configuration),
        "verifier_input": {
            "action_dimension": 8,
            "dtype": "float32",
            "mask_semantic": MASK_SEMANTIC,
            "model_equivalence_semantic": MODEL_EQUIVALENCE_SEMANTIC,
            "model_horizon": 16,
            "source_executable_horizon": 10,
            "verifier_input_semantic": VERIFIER_INPUT_SEMANTIC,
        },
    }
    return make_envelope(SMOKE_CONTRACT_SCHEMA, payload)


def audit_readiness(
    *,
    repository_root: Path,
    contract_envelope: Mapping[str, object] | None = None,
    proposal_envelope: Mapping[str, object] | None = None,
    raw_candidate_envelope: Mapping[str, object] | None = None,
    projected_envelope: Mapping[str, object] | None = None,
    score_envelope: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return honest structural or actual bounded-M6B readiness."""

    release_path = repository_root / "docs/release/release-manifest.json"
    observed_release = f"sha256:{hashlib.sha256(release_path.read_bytes()).hexdigest()}"
    gates: dict[str, bool] = {
        "candidate_source_frozen": len(
            cast(list[object], frozen_candidate_source()["selected_transformations"])
        )
        == 3,
        "fixed_continuation_frozen": True,
        "initial_state_only_frozen": True,
        "m5_release_manifest_unchanged": observed_release == M5_RELEASE_MANIFEST_SHA256,
        "m6a_historical_blockers_preserved": (
            repository_root / "docs/integrations/langmani/blockers.json"
        ).is_file(),
        "verifier_input_frozen": True,
    }
    actual = all(
        item is not None
        for item in (
            contract_envelope,
            proposal_envelope,
            raw_candidate_envelope,
            projected_envelope,
            score_envelope,
        )
    )
    if actual:
        assert contract_envelope is not None
        assert proposal_envelope is not None
        assert raw_candidate_envelope is not None
        assert projected_envelope is not None
        assert score_envelope is not None
        contract = validate_envelope(
            contract_envelope, expected_schema=SMOKE_CONTRACT_SCHEMA
        )
        proposal = validate_envelope(
            proposal_envelope, expected_schema=LANGMANI_INITIAL_PROPOSAL_SCHEMA
        )
        raw = validate_envelope(
            raw_candidate_envelope, expected_schema=RAW_CANDIDATE_REQUEST_SCHEMA
        )
        projected = validate_envelope(
            projected_envelope, expected_schema=LANGMANI_PROJECTED_CANDIDATES_SCHEMA
        )
        score = validate_envelope(score_envelope, expected_schema=SCORE_RESULT_SCHEMA)
        gates.update(
            {
                "actual_cross_process_bridge_passed": (
                    raw.get("policy_binding_digest")
                    == contract.get("policy_binding_digest")
                    and raw.get("proposal_digest")
                    == proposal_envelope["content_digest"]
                    and raw.get("candidate_source_digest")
                    == cast(
                        Mapping[str, object], contract.get("candidate_source", {})
                    ).get("candidate_source_digest")
                    and projected.get("raw_candidate_manifest_digest")
                    == raw_candidate_envelope["content_digest"]
                    and projected.get("candidate_pool_digest")
                    == raw.get("candidate_pool_digest")
                    and score.get("projected_candidate_manifest_digest")
                    == projected_envelope["content_digest"]
                    and score.get("candidate_pool_digest")
                    == raw.get("candidate_pool_digest")
                ),
                "blind_ranking_finalized": (
                    score.get("outcomes_available_during_selection") is False
                    and isinstance(score.get("selected_candidate_id"), str)
                ),
                "four_candidate_pool_passed": raw.get("candidate_count") == 4
                and projected.get("candidate_count") == 4,
                "mask_invariance_passed": cast(
                    Mapping[str, object], score.get("mask_invariance", {})
                ).get("passed")
                is True,
                "one_reset_zero_steps": proposal.get("reset_count") == 1
                and proposal.get("policy_query_count") == 1
                and proposal.get("environment_step_count") == 0,
                "zero_outcomes": proposal.get("outcomes_generated") is False
                and raw.get("outcomes_available") is False
                and projected.get("outcomes_available") is False,
            }
        )
    status = (
        "ready_for_bounded_m6b"
        if actual and all(gates.values())
        else "conditionally_ready"
    )
    payload = {
        "blocker_resolutions": {
            "M6A-B003": "resolved" if gates["candidate_source_frozen"] else "blocked",
            "M6A-B004": "scope_resolved_initial_state_only",
            "M6A-B005": (
                "resolved_for_bounded_m6b"
                if gates.get("mask_invariance_passed")
                else "pending_real_mask_gate"
            ),
            "M6A-B006": "scope_resolved_seeded_initial_reset_and_fixed_continuation",
            "M6A-B007": (
                "resolved"
                if gates.get("actual_cross_process_bridge_passed")
                else "pending_real_bridge"
            ),
            "M6A-B009": (
                "resolved" if contract_envelope is not None else "pending_exact_binding"
            ),
        },
        "future_limitations": [
            "no_intermediate_policy_snapshot",
            "no_intermediate_exact_replay",
            "no_same_policy_continuation_after_divergence",
            "no_closed_loop_langmani_shielding",
            "no_structured_langmani_verifier_state",
        ],
        "gates": gates,
        "m6b_executed": False,
        "readiness_status": status,
    }
    return make_envelope(READINESS_SCHEMA, payload)


__all__ = [
    "ACCEPTED_ACTION_ONLY_BUNDLE_DIGEST",
    "LANGMANI_INITIAL_PROPOSAL_SCHEMA",
    "LANGMANI_POLICY_BINDING_SCHEMA",
    "LANGMANI_PROJECTED_CANDIDATES_SCHEMA",
    "M5_RELEASE_MANIFEST_SHA256",
    "M6A1BridgeError",
    "RAW_CANDIDATE_REQUEST_SCHEMA",
    "READINESS_SCHEMA",
    "SCORE_RESULT_SCHEMA",
    "SMOKE_CONTRACT_SCHEMA",
    "array_content_digest",
    "audit_readiness",
    "build_raw_candidates",
    "content_digest",
    "derive_probe_seed",
    "freeze_smoke_contract",
    "frozen_candidate_source",
    "m6b_source_seeds",
    "make_envelope",
    "read_envelope",
    "score_projected_candidates",
    "validate_envelope",
    "write_envelope",
]
