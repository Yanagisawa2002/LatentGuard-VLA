"""CPU-only tests for native ACT collection authorization and alignment."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from latentguard.policies.act.collection import (
    PickCubeDemoCollectionError,
    PickCubePreActionCapture,
    validate_expert_gate_authorization,
)
from latentguard.policies.experts import (
    PICKCUBE_EXPERT_PHASES,
    ExpertEpisodeAudit,
    PickCubeExpertPhaseTracker,
    summarize_expert_evaluation,
)


def _authorized_gate() -> dict[str, object]:
    episodes = tuple(
        ExpertEpisodeAudit(
            seed=600000 + index,
            success=True,
            action_count=40,
            phase_action_counts={phase.value: 10 for phase in PICKCUBE_EXPERT_PHASES},
        )
        for index in range(100)
    )
    return summarize_expert_evaluation(
        episodes,
        source_audit_digest="sha256:" + "a" * 64,
    ).to_mapping()


def test_collection_gate_requires_complete_authorized_evidence() -> None:
    gate = _authorized_gate()
    digest = validate_expert_gate_authorization(gate)
    assert digest.startswith("sha256:")
    gate["authorized_for_demonstration_collection"] = False
    with pytest.raises(PickCubeDemoCollectionError, match="authorization is false"):
        validate_expert_gate_authorization(gate)


class _Factory:
    def extract_named_robot_state(
        self, environment: object
    ) -> tuple[tuple[str, ...], np.ndarray[Any, Any]]:
        del environment
        names = tuple(f"panda_joint{index}" for index in range(1, 8)) + (
            "panda_finger_joint1",
            "panda_finger_joint2",
        )
        return names, np.arange(18, dtype=np.float32)


class _Handle:
    def render_views(self) -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(
                camera_id=camera_id,
                rgb=np.full((224, 224, 3), index, dtype=np.uint8),
                camera_configuration_digest="sha256:" + "b" * 64,
            )
            for index, camera_id in enumerate(
                ("front_oblique", "overhead", "side_oblique")
            )
        )


class _Renderer:
    def prepare(self, environment: object, plan: object) -> _Handle:
        del environment, plan
        return _Handle()


def test_pre_action_capture_preserves_exact_alignment() -> None:
    tracker = PickCubeExpertPhaseTracker()
    capture = PickCubePreActionCapture(
        factory=cast(Any, _Factory()),
        renderer=cast(Any, _Renderer()),
        render_plan=cast(Any, object()),
        phase_tracker=tracker,
    )
    environment = object()
    capture.capture_boundary(environment, 0)
    tracker.transition(PICKCUBE_EXPERT_PHASES[0])
    action = np.arange(8, dtype=np.float64)
    capture.capture_pre_action(environment, 0, action)
    capture.capture_boundary(environment, 1)
    episode = capture.build_episode(
        scene_seed=700002,
        recorded_actions=(action,),
        compatibility_identity="sha256:" + "c" * 64,
        contract_digest="sha256:" + "d" * 64,
    )
    assert episode.frame_count == 1
    assert episode.rgb.dtype == np.uint8
    assert episode.proprioception.dtype == np.float32
    assert episode.actions.dtype == np.float64
    assert episode.phases == (PICKCUBE_EXPERT_PHASES[0].value,)


def test_pre_action_capture_rejects_action_content_drift() -> None:
    tracker = PickCubeExpertPhaseTracker()
    capture = PickCubePreActionCapture(
        factory=cast(Any, _Factory()),
        renderer=cast(Any, _Renderer()),
        render_plan=cast(Any, object()),
        phase_tracker=tracker,
    )
    capture.capture_boundary(object(), 0)
    tracker.transition(PICKCUBE_EXPERT_PHASES[0])
    action = np.zeros((8,), dtype=np.float32)
    capture.capture_pre_action(object(), 0, action)
    with pytest.raises(PickCubeDemoCollectionError, match="action 0 content changed"):
        capture.build_episode(
            scene_seed=700003,
            recorded_actions=(np.ones((8,), dtype=np.float32),),
            compatibility_identity="sha256:" + "c" * 64,
            contract_digest="sha256:" + "d" * 64,
        )
