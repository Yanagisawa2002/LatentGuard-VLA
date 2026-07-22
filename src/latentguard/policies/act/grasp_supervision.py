"""P0.2 temporal, phase, gripper, and staged-behavior contracts."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Any, NoReturn

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from latentguard.policies.act.types import PickCubeActGraspSupervisionConfig
from latentguard.replay.identity import canonical_json_bytes


class PickCubeGraspSupervisionError(ValueError):
    """Raised when P0.2 supervision or progress evidence is malformed."""


class PickCubeManipulationPhase(StrEnum):
    """Diagnostic-only phases that never enter policy observations."""

    RESET_OR_IDLE = "RESET_OR_IDLE"
    APPROACH = "APPROACH"
    PREGRASP = "PREGRASP"
    GRIPPER_CLOSING = "GRIPPER_CLOSING"
    CONTACT_OR_GRASP = "CONTACT_OR_GRASP"
    LIFT = "LIFT"
    TRANSPORT_OR_COMPLETION = "TRANSPORT_OR_COMPLETION"


class PickCubeGripperEvent(IntEnum):
    """Minimal target-side endpoint/transition taxonomy."""

    OPEN = 0
    CLOSING = 1
    CLOSED = 2


class PickCubeP02Result(StrEnum):
    """Exact mutually exclusive P0.2 terminal result."""

    RESULT_A_ACCEPTED = "RESULT_A_ACCEPTED_COMPATIBLE_ACT_POLICY"
    RESULT_B_GRASP_UNPROMOTED = "RESULT_B_GRASP_BEHAVIOR_ACQUIRED_UNPROMOTED"
    RESULT_C_NO_GRASP = "RESULT_C_ACT_STILL_FAILS_TO_ACQUIRE_GRASP"


class PickCubeRolloutFailure(StrEnum):
    """Earliest dominant manipulation failure after an honest rollout."""

    NO_INITIAL_MOTION = "no_initial_motion"
    NEVER_ENTERS_PREGRASP = "never_enters_pregrasp"
    NEVER_COMMANDS_CLOSE = "never_commands_close"
    CLOSE_COMMAND_TOO_EARLY = "close_command_too_early"
    CONTACT_WITHOUT_GRASP = "contact_without_grasp"
    GRASP_WITHOUT_LIFT = "grasp_without_lift"
    TIMEOUT_AFTER_PROGRESS = "timeout_after_progress"
    OTHER = "other"


_EXPERT_PHASE_MAP = MappingProxyType(
    {
        "REACH_PREGRASP": PickCubeManipulationPhase.APPROACH,
        "DESCEND_TO_GRASP": PickCubeManipulationPhase.PREGRASP,
        "CLOSE_GRIPPER": PickCubeManipulationPhase.GRIPPER_CLOSING,
        "TRANSPORT_TO_GOAL": PickCubeManipulationPhase.TRANSPORT_OR_COMPLETION,
    }
)


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeGraspSupervisionError(f"{context}: {reason}")


def phase_from_expert_label(value: str) -> PickCubeManipulationPhase:
    """Map the recorded expert phase without inventing contact or lift state."""
    try:
        return _EXPERT_PHASE_MAP[value]
    except (KeyError, TypeError) as exc:
        raise PickCubeGraspSupervisionError(
            f"expert phase: unsupported label {value!r}"
        ) from exc


def close_transition_index_in_window(
    phases: Sequence[str],
    *,
    window_start: int,
    window_length: int,
) -> int | None:
    """Locate the unique open-to-close phase boundary within one action chunk."""
    if not phases or any(value not in PickCubeManipulationPhase for value in phases):
        _fail("close transition window", "phases contain an unsupported value")
    if (
        type(window_start) is not int
        or not 0 <= window_start < len(phases)
        or type(window_length) is not int
        or window_length < 1
    ):
        _fail("close transition window", "window bounds are invalid")
    closing = PickCubeManipulationPhase.GRIPPER_CLOSING.value
    stop = min(len(phases), window_start + window_length)
    for absolute_index in range(window_start, stop):
        prior = phases[absolute_index - 1] if absolute_index > 0 else None
        if phases[absolute_index] == closing and prior != closing:
            return absolute_index - window_start
    return None


def predicted_close_index(
    actions: NDArray[Any], *, close_threshold: float
) -> int | None:
    """Return the first predicted close command in a finite native-action chunk."""
    values = np.asarray(actions)
    if (
        values.ndim != 2
        or values.shape[1:] != (8,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
    ):
        _fail("predicted close", "actions must be finite floating [T,8]")
    if not math.isfinite(close_threshold) or not -1.0 < close_threshold < 0.0:
        _fail("predicted close", "threshold differs from the bounded convention")
    indices = np.flatnonzero(values[:, 7] <= close_threshold)
    return None if not len(indices) else int(indices[0])


def aligned_frame_pairs(length: int, offset: int) -> tuple[tuple[int, int], ...]:
    """Return boundary-safe ``(observation, action)`` indices for one episode."""
    if type(length) is not int or length < 1:
        _fail("temporal alignment", "length must be positive")
    if type(offset) is not int or not -2 <= offset <= 2:
        _fail("temporal alignment", "offset must lie in [-2,2]")
    return tuple(
        (observation, observation + offset)
        for observation in range(length)
        if 0 <= observation + offset < length
    )


def action_chunk_at_offset(
    actions: NDArray[Any],
    observation_index: int,
    chunk_size: int,
    offset: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Build a target chunk without crossing an episode or adding terminal no-ops."""
    values = np.asarray(actions)
    if (
        values.ndim != 2
        or values.shape[1:] != (8,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
    ):
        _fail("offset action chunk", "actions must be finite floating [T,8]")
    if type(observation_index) is not int or not 0 <= observation_index < len(values):
        _fail("offset action chunk", "observation index is outside episode")
    if type(chunk_size) is not int or chunk_size < 1:
        _fail("offset action chunk", "chunk size must be positive")
    pairs = dict(aligned_frame_pairs(len(values), offset))
    if observation_index not in pairs:
        _fail("offset action chunk", "offset leaves the episode")
    start = pairs[observation_index]
    end = min(len(values), start + chunk_size)
    valid = np.array(values[start:end], dtype=np.float32, copy=True)
    chunk = np.zeros((chunk_size, 8), dtype=np.float32)
    chunk[: len(valid)] = valid
    padding = np.ones((chunk_size,), dtype=np.bool_)
    padding[: len(valid)] = False
    return chunk, padding


def derive_gripper_events(
    targets: Sequence[float] | NDArray[Any],
    *,
    close_threshold: float,
    open_threshold: float,
) -> NDArray[np.int64]:
    """Label endpoint commands and the first open-to-close command exactly once."""
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        _fail("gripper events", "targets must be a non-empty finite vector")
    if not -1.0 < close_threshold < 0.0 < open_threshold < 1.0:
        _fail("gripper events", "thresholds differ from the bounded convention")
    if np.any((values > close_threshold) & (values < open_threshold)):
        _fail("gripper events", "ambiguous continuous target is unsupported")
    closed = values <= close_threshold
    result = np.where(
        closed,
        int(PickCubeGripperEvent.CLOSED),
        int(PickCubeGripperEvent.OPEN),
    ).astype(np.int64)
    transitions = closed & np.concatenate(
        (np.array([False]), np.logical_not(closed[:-1]))
    )
    result[transitions] = int(PickCubeGripperEvent.CLOSING)
    return result


def phase_weights_from_train_labels(
    labels: Sequence[str],
    *,
    exponent: float,
    maximum: float,
) -> Mapping[str, float]:
    """Compute mean-one, inverse-frequency weights from train labels only."""
    if not labels or any(not isinstance(item, str) or not item for item in labels):
        _fail("phase weights", "labels must be non-empty strings")
    if not 0.0 <= exponent <= 1.0 or not 1.0 <= maximum <= 3.0:
        _fail("phase weights", "bounded weighting configuration is invalid")
    counts = Counter(labels)
    total = sum(counts.values())
    raw = {name: (total / count) ** exponent for name, count in counts.items()}
    mean = sum(raw[name] * counts[name] for name in raw) / total
    normalized = {name: min(maximum, value / mean) for name, value in raw.items()}
    renormalizer = sum(normalized[name] * counts[name] for name in normalized) / total
    result = {name: value / renormalizer for name, value in normalized.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in result.values()):
        _fail("phase weights", "non-finite weight was produced")
    if max(result.values()) > maximum + 1e-12:
        _fail("phase weights", "renormalization exceeded the declared cap")
    return MappingProxyType(dict(sorted(result.items())))


def supervision_identity(
    *,
    weights: Mapping[str, float],
    config: PickCubeActGraspSupervisionConfig,
) -> Mapping[str, object]:
    """Bind the actual train-derived weights into the training identity."""
    body: dict[str, object] = {
        "phase_weights": dict(sorted(weights.items())),
        "schema_version": "pickcube-act-grasp-supervision-identity-v1",
        "supervision": config.to_mapping(),
    }
    body["supervision_digest"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(body, context="PickCubeActGraspSupervision")
        ).hexdigest()
    )
    return MappingProxyType(body)


def phase_gripper_aware_loss(
    *,
    raw_actions: torch.Tensor,
    bounded_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_is_pad: torch.Tensor,
    phase_weights: torch.Tensor,
    gripper_events: torch.Tensor,
    config: PickCubeActGraspSupervisionConfig,
    mean: torch.Tensor | None,
    log_variance: torch.Tensor | None,
    kl_weight: float,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    """Compute separated bounded ACT losses with capped event weighting."""
    expected = target_actions.shape
    if (
        raw_actions.shape != expected
        or bounded_actions.shape != expected
        or expected[-1:] != (8,)
        or action_is_pad.shape != expected[:-1]
        or phase_weights.shape != expected[:-1]
        or gripper_events.shape != expected[:-1]
    ):
        _fail("phase-aware loss", "tensor shapes differ")
    if not all(
        torch.isfinite(value).all()
        for value in (raw_actions, bounded_actions, target_actions, phase_weights)
    ):
        _fail("phase-aware loss", "non-finite tensor")
    if torch.any(phase_weights <= 0.0) or torch.any(
        phase_weights > config.maximum_phase_weight + 1e-6
    ):
        _fail("phase-aware loss", "phase weight left its bounded interval")
    valid = torch.logical_not(action_is_pad)
    weights = phase_weights * valid.to(phase_weights.dtype)
    denominator = weights.sum().clamp_min(1.0)
    arm_per_step = torch.abs(bounded_actions[..., :7] - target_actions[..., :7]).mean(
        dim=-1
    )
    gripper_per_step = torch.abs(bounded_actions[..., 7] - target_actions[..., 7])
    arm_loss = (arm_per_step * weights).sum() / denominator
    gripper_loss = (gripper_per_step * weights).sum() / denominator
    close_target = (gripper_events != int(PickCubeGripperEvent.OPEN)).to(
        raw_actions.dtype
    )
    transition_multiplier = torch.where(
        gripper_events == int(PickCubeGripperEvent.CLOSING),
        torch.full_like(phase_weights, config.transition_weight_boost),
        torch.ones_like(phase_weights),
    )
    event_weights = weights * transition_multiplier
    event_denominator = event_weights.sum().clamp_min(1.0)
    event_per_step = F.binary_cross_entropy_with_logits(
        -raw_actions[..., 7], close_target, reduction="none"
    )
    transition_loss = (event_per_step * event_weights).sum() / event_denominator
    total = (
        config.arm_loss_weight * arm_loss
        + config.gripper_loss_weight * gripper_loss
        + config.transition_loss_weight * transition_loss
    )
    kl_loss = torch.zeros((), dtype=total.dtype, device=total.device)
    if mean is not None or log_variance is not None:
        if mean is None or log_variance is None or mean.shape != log_variance.shape:
            _fail("phase-aware loss", "VAE statistics differ")
        kl_loss = (
            (-0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp()))
            .sum(-1)
            .mean()
        )
        total = total + float(kl_weight) * kl_loss
    if not torch.isfinite(total):
        _fail("phase-aware loss", "total loss is non-finite")
    metrics = MappingProxyType(
        {
            "arm_action_loss": float(arm_loss.detach().cpu()),
            "gripper_action_loss": float(gripper_loss.detach().cpu()),
            "gripper_transition_loss": float(transition_loss.detach().cpu()),
            "kld_loss": float(kl_loss.detach().cpu()),
        }
    )
    return total, metrics


def forward_phase_aware_act(
    *,
    policy: Any,
    batch: Mapping[str, torch.Tensor],
    config: PickCubeActGraspSupervisionConfig,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    """Run the installed ACT model once and apply the P0.2 training-only loss."""
    model = getattr(policy, "model", None)
    policy_config = getattr(policy, "config", None)
    image_features = getattr(policy_config, "image_features", None)
    kl_weight = getattr(policy_config, "kl_weight", None)
    if not callable(model) or not isinstance(image_features, Mapping):
        _fail("phase-aware forward", "installed ACT public model contract changed")
    if not isinstance(kl_weight, (int, float)) or not math.isfinite(float(kl_weight)):
        _fail("phase-aware forward", "ACT KL weight is unavailable")
    prepared: dict[str, Any] = dict(batch)
    try:
        prepared["observation.images"] = [
            batch[key] for key in image_features
        ]  # installed LeRobot 0.6 public feature contract
        actions_hat, distribution = model(prepared)
        mean, log_variance = distribution
        head = getattr(model, "action_head", None)
        latest_outputs = getattr(head, "latest_outputs", None)
        if not callable(latest_outputs):
            _fail("phase-aware forward", "bounded action head is unavailable")
        raw_actions, bounded_actions = latest_outputs()
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise PickCubeGraspSupervisionError(
            "phase-aware forward: installed ACT interface changed"
        ) from exc
    if not isinstance(actions_hat, torch.Tensor) or actions_hat.data_ptr() != (
        bounded_actions.data_ptr()
    ):
        _fail("phase-aware forward", "bounded head output identity changed")
    return phase_gripper_aware_loss(
        raw_actions=raw_actions,
        bounded_actions=bounded_actions,
        target_actions=batch["action"],
        action_is_pad=batch["action_is_pad"],
        phase_weights=batch["p02_phase_weight"],
        gripper_events=batch["p02_gripper_event"],
        config=config,
        mean=mean,
        log_variance=log_variance,
        kl_weight=float(kl_weight),
    )


@dataclass(frozen=True, slots=True)
class PickCubeProgressSample:
    """One privileged diagnostic sample that is never a policy input."""

    step_index: int
    query_index: int
    action_index_in_chunk: int
    joint_positions: tuple[float, ...]
    commanded_action: tuple[float, ...]
    gripper_position: float
    commanded_gripper: float
    tcp_position: tuple[float, float, float]
    cube_position: tuple[float, float, float]
    tcp_to_cube_distance: float
    left_contact_force: float
    right_contact_force: float
    grasped: bool
    environment_step_latency_seconds: float

    def __post_init__(self) -> None:
        if (
            type(self.step_index) is not int
            or self.step_index < 0
            or type(self.query_index) is not int
            or self.query_index < 0
            or type(self.action_index_in_chunk) is not int
            or self.action_index_in_chunk < 0
        ):
            _fail("progress sample", "indices must be non-negative integers")
        if len(self.joint_positions) != 9 or len(self.commanded_action) != 8:
            _fail("progress sample", "joint/action dimensions changed")
        numeric = (
            *self.joint_positions,
            *self.commanded_action,
            self.gripper_position,
            self.commanded_gripper,
            *self.tcp_position,
            *self.cube_position,
            self.tcp_to_cube_distance,
            self.left_contact_force,
            self.right_contact_force,
            self.environment_step_latency_seconds,
        )
        if any(not math.isfinite(value) for value in numeric):
            _fail("progress sample", "numeric evidence must be finite")


@dataclass(frozen=True, slots=True)
class PickCubeRolloutProgressSummary:
    """Event timestamps and stage outcomes for one fixed-seed rollout."""

    first_arm_motion: int | None
    entered_pregrasp_region: int | None
    first_close_command: int | None
    first_valid_close: int | None
    first_contact: int | None
    first_grasp: int | None
    first_lift: int | None
    success_step: int | None
    failure: PickCubeRolloutFailure | None

    def to_mapping(self) -> Mapping[str, object]:
        """Return a compact machine-readable event contract."""
        return MappingProxyType(
            {
                "entered_pregrasp": self.entered_pregrasp_region is not None,
                "failure": None if self.failure is None else self.failure.value,
                "first_arm_motion": self.first_arm_motion,
                "first_close_command": self.first_close_command,
                "first_contact": self.first_contact,
                "first_grasp": self.first_grasp,
                "first_lift": self.first_lift,
                "first_valid_close": self.first_valid_close,
                "grasped": self.first_grasp is not None,
                "lifted": self.first_lift is not None,
                "success_step": self.success_step,
                "valid_close": self.first_valid_close is not None,
            }
        )


def analyze_progress_trace(
    samples: Sequence[PickCubeProgressSample],
    *,
    initial_joint_positions: Sequence[float],
    initial_cube_height: float,
    success: bool,
    pregrasp_distance: float = 0.08,
    arm_motion_threshold: float = 0.01,
    contact_force_threshold: float = 0.05,
    lift_height_delta: float = 0.03,
    close_threshold: float = -0.5,
) -> PickCubeRolloutProgressSummary:
    """Detect real progress events without using privilege as a policy input."""
    if not samples:
        _fail("progress trace", "at least one executed action is required")
    joints = np.asarray(initial_joint_positions, dtype=np.float64)
    if joints.shape != (9,) or not np.all(np.isfinite(joints)):
        _fail("progress trace", "initial joint positions must be finite [9]")
    thresholds = (
        initial_cube_height,
        pregrasp_distance,
        arm_motion_threshold,
        contact_force_threshold,
        lift_height_delta,
        close_threshold,
    )
    if any(not math.isfinite(value) for value in thresholds):
        _fail("progress trace", "thresholds must be finite")
    event: dict[str, int | None] = {
        "first_arm_motion": None,
        "entered_pregrasp_region": None,
        "first_close_command": None,
        "first_valid_close": None,
        "first_contact": None,
        "first_grasp": None,
        "first_lift": None,
    }
    for sample in samples:
        if (
            event["first_arm_motion"] is None
            and np.linalg.norm(np.asarray(sample.joint_positions[:7]) - joints[:7])
            >= arm_motion_threshold
        ):
            event["first_arm_motion"] = sample.step_index
        pregrasp = sample.tcp_to_cube_distance <= pregrasp_distance
        if event["entered_pregrasp_region"] is None and pregrasp:
            event["entered_pregrasp_region"] = sample.step_index
        close = sample.commanded_gripper <= close_threshold
        if event["first_close_command"] is None and close:
            event["first_close_command"] = sample.step_index
        if event["first_valid_close"] is None and close and pregrasp:
            event["first_valid_close"] = sample.step_index
        contact = (
            sample.left_contact_force >= contact_force_threshold
            or sample.right_contact_force >= contact_force_threshold
        )
        if event["first_contact"] is None and contact:
            event["first_contact"] = sample.step_index
        if event["first_grasp"] is None and sample.grasped:
            event["first_grasp"] = sample.step_index
        if event["first_lift"] is None and (
            sample.cube_position[2] >= initial_cube_height + lift_height_delta
        ):
            event["first_lift"] = sample.step_index
    success_step = samples[-1].step_index if success else None
    failure: PickCubeRolloutFailure | None = None
    if not success:
        if event["first_arm_motion"] is None:
            failure = PickCubeRolloutFailure.NO_INITIAL_MOTION
        elif event["entered_pregrasp_region"] is None:
            failure = PickCubeRolloutFailure.NEVER_ENTERS_PREGRASP
        elif event["first_close_command"] is None:
            failure = PickCubeRolloutFailure.NEVER_COMMANDS_CLOSE
        elif event["first_valid_close"] is None:
            failure = PickCubeRolloutFailure.CLOSE_COMMAND_TOO_EARLY
        elif event["first_contact"] is not None and event["first_grasp"] is None:
            failure = PickCubeRolloutFailure.CONTACT_WITHOUT_GRASP
        elif event["first_grasp"] is not None and event["first_lift"] is None:
            failure = PickCubeRolloutFailure.GRASP_WITHOUT_LIFT
        else:
            failure = PickCubeRolloutFailure.TIMEOUT_AFTER_PROGRESS
    return PickCubeRolloutProgressSummary(
        first_arm_motion=event["first_arm_motion"],
        entered_pregrasp_region=event["entered_pregrasp_region"],
        first_close_command=event["first_close_command"],
        first_valid_close=event["first_valid_close"],
        first_contact=event["first_contact"],
        first_grasp=event["first_grasp"],
        first_lift=event["first_lift"],
        success_step=success_step,
        failure=failure,
    )


@dataclass(frozen=True, slots=True)
class PickCubeStagedGateCounts:
    """Fixed 30-seed behavior counts and immutable thresholds."""

    episode_count: int
    action_integrity_failures: int
    pregrasp_count: int
    valid_close_count: int
    grasp_count: int
    lift_count: int
    success_count: int

    def __post_init__(self) -> None:
        if self.episode_count != 30:
            _fail("staged gates", "development requires exactly 30 episodes")
        values = (
            self.action_integrity_failures,
            self.pregrasp_count,
            self.valid_close_count,
            self.grasp_count,
            self.lift_count,
            self.success_count,
        )
        if any(type(value) is not int or not 0 <= value <= 30 for value in values):
            _fail("staged gates", "counts must lie in [0,30]")

    def to_mapping(self) -> Mapping[str, object]:
        """Return every independent gate without conflating progress and success."""
        return MappingProxyType(
            {
                "action_integrity_passed": self.action_integrity_failures == 0,
                "approach_gate_passed": self.pregrasp_count >= 24,
                "episode_count": self.episode_count,
                "grasp_gate_passed": self.grasp_count >= 18,
                "gripper_gate_passed": self.valid_close_count >= 21,
                "lift_gate_passed": self.lift_count >= 15,
                "pregrasp_count": self.pregrasp_count,
                "promotion_gate_passed": self.success_count / 30 >= 0.75,
                "success_count": self.success_count,
                "valid_close_count": self.valid_close_count,
                "grasp_count": self.grasp_count,
                "lift_count": self.lift_count,
            }
        )


def classify_p02_result(counts: PickCubeStagedGateCounts) -> PickCubeP02Result:
    """Classify A/B/C while keeping action integrity and promotion conjunctive."""
    gates = counts.to_mapping()
    if gates["action_integrity_passed"] is not True:
        _fail("P0.2 result", "action integrity failure is a hard stop")
    if gates["promotion_gate_passed"] is True:
        return PickCubeP02Result.RESULT_A_ACCEPTED
    if gates["approach_gate_passed"] is True and gates["grasp_gate_passed"] is True:
        return PickCubeP02Result.RESULT_B_GRASP_UNPROMOTED
    return PickCubeP02Result.RESULT_C_NO_GRASP


def final_access_authorized(result: PickCubeP02Result) -> bool:
    """Authorize sealed seeds and packaging only for an accepted Result A."""
    return result is PickCubeP02Result.RESULT_A_ACCEPTED


__all__ = [
    "PickCubeGraspSupervisionError",
    "PickCubeGripperEvent",
    "PickCubeManipulationPhase",
    "PickCubeP02Result",
    "PickCubeProgressSample",
    "PickCubeRolloutFailure",
    "PickCubeRolloutProgressSummary",
    "PickCubeStagedGateCounts",
    "action_chunk_at_offset",
    "analyze_progress_trace",
    "aligned_frame_pairs",
    "classify_p02_result",
    "close_transition_index_in_window",
    "derive_gripper_events",
    "final_access_authorized",
    "forward_phase_aware_act",
    "phase_from_expert_label",
    "phase_gripper_aware_loss",
    "phase_weights_from_train_labels",
    "predicted_close_index",
    "supervision_identity",
]
