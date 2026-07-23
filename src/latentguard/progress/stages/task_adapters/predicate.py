"""Task-specific stage labels derived from current privileged predicates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from latentguard.progress.models import StageLabel, compose_progress


@dataclass(frozen=True)
class TaskStageSpec:
    """Declarative stage inputs for one frozen LIBERO task."""

    adapter: str
    manipulated_objects: tuple[str, ...]
    goal_predicates: tuple[tuple[str, ...], ...]
    auxiliary_entities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the supported task-stage vocabulary."""

        supported = {
            "pick_place",
            "open_place",
            "toggle",
            "toggle_place",
            "multi_place",
            "place_close",
        }
        if self.adapter not in supported:
            raise ValueError(f"unsupported stage adapter: {self.adapter}")
        if not self.manipulated_objects:
            raise ValueError("at least one manipulated object is required")
        if not self.goal_predicates:
            raise ValueError("at least one goal predicate is required")


@dataclass(frozen=True)
class _Thresholds:
    approach: float = 0.18
    aligned: float = 0.08
    contact: float = 0.045
    lift_delta: float = 0.035
    target_near: float = 0.12


class DeclarativeTaskStageAdapter:
    """Label the five frozen LG-R1 task structures from current evidence."""

    def __init__(self, spec: TaskStageSpec) -> None:
        """Store one immutable task specification."""

        self._spec = spec

    def label_step(
        self,
        privileged_state: dict[str, Any],
        task_metadata: dict[str, Any],
    ) -> StageLabel:
        """Label one state using only current and episode-initial evidence."""

        thresholds = _thresholds(task_metadata)
        terminal_success = bool(privileged_state.get("terminal_success", False))
        terminal_failure = bool(privileged_state.get("terminal_failure", False))
        goal_flags = _goal_flags(privileged_state, self._spec.goal_predicates)
        initial = task_metadata.get("initial_privileged_state")
        initial_state = initial if isinstance(initial, dict) else {}

        if self._spec.adapter == "pick_place":
            return self._label_pick_place(
                privileged_state,
                initial_state,
                thresholds,
                goal_flags,
                terminal_success,
                terminal_failure,
            )
        if self._spec.adapter == "open_place":
            return self._label_open_place(
                privileged_state,
                initial_state,
                thresholds,
                goal_flags,
                terminal_success,
                terminal_failure,
            )
        if self._spec.adapter == "toggle":
            return self._label_toggle(
                privileged_state,
                thresholds,
                goal_flags,
                terminal_success,
                terminal_failure,
            )
        if self._spec.adapter == "toggle_place":
            return self._label_toggle_place(
                privileged_state,
                initial_state,
                thresholds,
                goal_flags,
                terminal_success,
                terminal_failure,
            )
        if self._spec.adapter == "multi_place":
            return self._label_multi_place(
                privileged_state,
                initial_state,
                thresholds,
                goal_flags,
                terminal_success,
                terminal_failure,
            )
        return self._label_place_close(
            privileged_state,
            initial_state,
            thresholds,
            goal_flags,
            terminal_success,
            terminal_failure,
        )

    def _label_toggle_place(
        self,
        state: dict[str, Any],
        initial: dict[str, Any],
        thresholds: _Thresholds,
        goals: list[bool],
        terminal_success: bool,
        terminal_failure: bool,
    ) -> StageLabel:
        names = (
            "approach_control",
            "activate_control",
            "approach_object",
            "grasp_contact",
            "lift",
            "transport",
            "place",
            "success",
        )
        object_name = self._spec.manipulated_objects[0]
        toggle_index = next(
            (
                index
                for index, predicate in enumerate(self._spec.goal_predicates)
                if predicate[0].lower() == "turnon"
            ),
            None,
        )
        object_index = next(
            (
                index
                for index, predicate in enumerate(self._spec.goal_predicates)
                if len(predicate) >= 3
                and predicate[0].lower() in {"in", "on"}
                and predicate[1] == object_name
            ),
            None,
        )
        if toggle_index is None or object_index is None:
            raise ValueError("toggle_place requires turnon and object-place goals")
        target_name = self._spec.goal_predicates[object_index][-1]
        control_name = self._spec.goal_predicates[toggle_index][1]
        features = _object_features(state, initial, object_name, target_name)
        features["control"] = control_name
        features["control_active"] = goals[toggle_index]
        features["goal_flags"] = goals
        if terminal_success:
            return _label(7, names, 1.0, True, terminal_failure, features)
        if goals[object_index]:
            return _label(6, names, 0.9, False, terminal_failure, features)
        if features["target_distance"] <= thresholds.target_near:
            return _label(6, names, 0.6, False, terminal_failure, features)
        if features["lift_delta"] > thresholds.lift_delta:
            stage = 5 if features["target_distance"] > thresholds.target_near else 6
            return _label(stage, names, 0.6, False, terminal_failure, features)
        if features["contact"]:
            return _label(3, names, 0.8, False, terminal_failure, features)
        if goals[toggle_index]:
            return _label(
                2,
                names,
                _bounded_inverse(
                    features["eef_distance"], thresholds.aligned, thresholds.approach
                ),
                False,
                terminal_failure,
                features,
            )
        control_distance = _eef_distance(state, control_name)
        features["control_eef_distance"] = control_distance
        if control_distance <= thresholds.aligned:
            return _label(1, names, 0.6, False, terminal_failure, features)
        return _label(
            0,
            names,
            _bounded_inverse(control_distance, thresholds.aligned, thresholds.approach),
            False,
            terminal_failure,
            features,
        )

    def _label_pick_place(
        self,
        state: dict[str, Any],
        initial: dict[str, Any],
        thresholds: _Thresholds,
        goals: list[bool],
        terminal_success: bool,
        terminal_failure: bool,
    ) -> StageLabel:
        names = (
            "approach_object",
            "pre_grasp_alignment",
            "grasp_contact",
            "lift",
            "transport",
            "place",
            "release",
            "success",
        )
        object_name = self._spec.manipulated_objects[0]
        features = _object_features(
            state,
            initial,
            object_name,
            self._spec.goal_predicates[0][-1],
        )
        if terminal_success:
            return _label(7, names, 1.0, True, terminal_failure, features)
        if goals[0]:
            return _label(6, names, 0.9, False, terminal_failure, features)
        if features["target_distance"] <= thresholds.target_near:
            return _label(5, names, 0.7, False, terminal_failure, features)
        if features["lift_delta"] > thresholds.lift_delta:
            completion = _bounded_inverse(
                features["target_distance"], thresholds.target_near, 0.5
            )
            stage = 4 if completion > 0.2 else 3
            return _label(
                stage,
                names,
                completion
                if stage == 4
                else _bounded_ratio(
                    features["lift_delta"], thresholds.lift_delta * 3.0
                ),
                False,
                terminal_failure,
                features,
            )
        if features["contact"]:
            return _label(2, names, 0.8, False, terminal_failure, features)
        if features["eef_distance"] <= thresholds.aligned:
            return _label(
                1,
                names,
                _bounded_inverse(
                    features["eef_distance"], thresholds.contact, thresholds.aligned
                ),
                False,
                terminal_failure,
                features,
            )
        return _label(
            0,
            names,
            _bounded_inverse(
                features["eef_distance"], thresholds.aligned, thresholds.approach
            ),
            False,
            terminal_failure,
            features,
        )

    def _label_open_place(
        self,
        state: dict[str, Any],
        initial: dict[str, Any],
        thresholds: _Thresholds,
        goals: list[bool],
        terminal_success: bool,
        terminal_failure: bool,
    ) -> StageLabel:
        names = (
            "approach_fixture",
            "open_fixture",
            "approach_object",
            "grasp_contact",
            "lift",
            "transport",
            "place",
            "success",
        )
        object_name = self._spec.manipulated_objects[0]
        object_goal_index = next(
            (
                index
                for index, predicate in enumerate(self._spec.goal_predicates)
                if len(predicate) >= 3
                and predicate[0].lower() in {"in", "on"}
                and predicate[1] == object_name
            ),
            None,
        )
        if object_goal_index is None:
            raise ValueError("open_place requires an object relation goal")
        target_name = self._spec.goal_predicates[object_goal_index][-1]
        features = _object_features(state, initial, object_name, target_name)
        fixture_name = _fixture_entity_name(
            state,
            self._spec.auxiliary_entities[0],
        )
        fixture = _entity(state, fixture_name)
        fixture_open = bool(fixture.get("open", False))
        features["fixture"] = fixture_name
        features["fixture_open"] = fixture_open
        if terminal_success:
            return _label(7, names, 1.0, True, terminal_failure, features)
        if goals[object_goal_index]:
            return _label(6, names, 0.9, False, terminal_failure, features)
        if features["lift_delta"] > thresholds.lift_delta:
            stage = 5 if features["target_distance"] > thresholds.target_near else 6
            return _label(stage, names, 0.6, False, terminal_failure, features)
        if features["contact"]:
            return _label(3, names, 0.8, False, terminal_failure, features)
        if fixture_open and features["eef_distance"] <= thresholds.aligned:
            return _label(2, names, 0.7, False, terminal_failure, features)
        if fixture_open:
            return _label(2, names, 0.2, False, terminal_failure, features)
        fixture_distance = _eef_distance(state, fixture_name)
        features["fixture_eef_distance"] = fixture_distance
        if fixture_distance <= thresholds.aligned:
            return _label(1, names, 0.5, False, terminal_failure, features)
        return _label(
            0,
            names,
            _bounded_inverse(fixture_distance, thresholds.aligned, thresholds.approach),
            False,
            terminal_failure,
            features,
        )

    def _label_toggle(
        self,
        state: dict[str, Any],
        thresholds: _Thresholds,
        goals: list[bool],
        terminal_success: bool,
        terminal_failure: bool,
    ) -> StageLabel:
        names = (
            "approach_control",
            "pre_contact_alignment",
            "control_contact",
            "activate",
            "success",
        )
        entity_name = self._spec.manipulated_objects[0]
        distance = _eef_distance(state, entity_name)
        entity = _entity(state, entity_name)
        contact = bool(entity.get("contact_with_gripper", False))
        evidence: dict[str, Any] = {
            "entity": entity_name,
            "eef_distance": distance,
            "contact": contact,
            "goal_satisfied": goals[0],
        }
        if terminal_success or goals[0]:
            return _label(4, names, 1.0, terminal_success, terminal_failure, evidence)
        if contact and bool(entity.get("turn_on", False)):
            return _label(3, names, 0.8, False, terminal_failure, evidence)
        if contact:
            return _label(2, names, 0.7, False, terminal_failure, evidence)
        if distance <= thresholds.aligned:
            return _label(1, names, 0.6, False, terminal_failure, evidence)
        return _label(
            0,
            names,
            _bounded_inverse(distance, thresholds.aligned, thresholds.approach),
            False,
            terminal_failure,
            evidence,
        )

    def _label_multi_place(
        self,
        state: dict[str, Any],
        initial: dict[str, Any],
        thresholds: _Thresholds,
        goals: list[bool],
        terminal_success: bool,
        terminal_failure: bool,
    ) -> StageLabel:
        names = (
            "approach_first_object",
            "manipulate_first_object",
            "first_subgoal_complete",
            "approach_second_object",
            "manipulate_second_object",
            "second_subgoal_complete",
            "final_arrangement",
            "success",
        )
        object_goal_indices = []
        for object_name in self._spec.manipulated_objects:
            matches = [
                index
                for index, predicate in enumerate(self._spec.goal_predicates)
                if len(predicate) >= 3
                and predicate[0].lower() in {"in", "on"}
                and predicate[1] == object_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"multi_place requires one goal for {object_name}, got {matches}"
                )
            object_goal_indices.append(matches[0])
        object_goals = [goals[index] for index in object_goal_indices]
        completed = sum(object_goals)
        active_index = 0 if not object_goals[0] else 1
        object_name = self._spec.manipulated_objects[active_index]
        predicate = self._spec.goal_predicates[object_goal_indices[active_index]]
        features = _object_features(state, initial, object_name, predicate[-1])
        features["goal_flags"] = goals
        features["object_goal_flags"] = object_goals
        if terminal_success:
            return _label(7, names, 1.0, True, terminal_failure, features)
        if completed == 2:
            return _label(6, names, 0.8, False, terminal_failure, features)
        if completed == 1:
            if features["target_distance"] <= thresholds.target_near:
                return _label(5, names, 0.8, False, terminal_failure, features)
            if features["contact"] or features["lift_delta"] > thresholds.lift_delta:
                return _label(4, names, 0.6, False, terminal_failure, features)
            return _label(
                3,
                names,
                _bounded_inverse(
                    features["eef_distance"], thresholds.aligned, thresholds.approach
                ),
                False,
                terminal_failure,
                features,
            )
        if features["contact"] or features["lift_delta"] > thresholds.lift_delta:
            return _label(1, names, 0.6, False, terminal_failure, features)
        return _label(
            0,
            names,
            _bounded_inverse(
                features["eef_distance"], thresholds.aligned, thresholds.approach
            ),
            False,
            terminal_failure,
            features,
        )

    def _label_place_close(
        self,
        state: dict[str, Any],
        initial: dict[str, Any],
        thresholds: _Thresholds,
        goals: list[bool],
        terminal_success: bool,
        terminal_failure: bool,
    ) -> StageLabel:
        names = (
            "approach_fixture",
            "open_fixture",
            "approach_object",
            "grasp_contact",
            "transport",
            "object_placed",
            "close_fixture",
            "success",
        )
        object_name = self._spec.manipulated_objects[0]
        target_name = self._spec.auxiliary_entities[0]
        fixture_name = _fixture_entity_name(
            state,
            target_name,
        )
        object_goal_index = next(
            (
                index
                for index, predicate in enumerate(self._spec.goal_predicates)
                if predicate[0].lower() == "in"
            ),
            0,
        )
        close_goal_index = next(
            (
                index
                for index, predicate in enumerate(self._spec.goal_predicates)
                if predicate[0].lower() == "close"
            ),
            0,
        )
        features = _object_features(state, initial, object_name, target_name)
        fixture = _entity(state, fixture_name)
        fixture_open = bool(fixture.get("open", False))
        features["fixture"] = fixture_name
        features["fixture_open"] = fixture_open
        features["goal_flags"] = goals
        if terminal_success:
            return _label(7, names, 1.0, True, terminal_failure, features)
        if goals[object_goal_index] and not goals[close_goal_index]:
            return _label(6, names, 0.6, False, terminal_failure, features)
        if goals[object_goal_index]:
            return _label(5, names, 0.9, False, terminal_failure, features)
        if features["lift_delta"] > thresholds.lift_delta:
            return _label(4, names, 0.6, False, terminal_failure, features)
        if features["contact"]:
            return _label(3, names, 0.7, False, terminal_failure, features)
        if fixture_open:
            return _label(2, names, 0.5, False, terminal_failure, features)
        fixture_distance = _eef_distance(state, fixture_name)
        features["fixture_eef_distance"] = fixture_distance
        if fixture_distance <= thresholds.aligned:
            return _label(1, names, 0.5, False, terminal_failure, features)
        return _label(
            0,
            names,
            _bounded_inverse(fixture_distance, thresholds.aligned, thresholds.approach),
            False,
            terminal_failure,
            features,
        )


def _thresholds(metadata: dict[str, Any]) -> _Thresholds:
    raw = metadata.get("distance_thresholds", {})
    values = raw if isinstance(raw, dict) else {}
    return _Thresholds(
        approach=float(values.get("approach_meters", 0.18)),
        aligned=float(values.get("aligned_meters", 0.08)),
        contact=float(values.get("contact_meters", 0.045)),
        lift_delta=float(values.get("lift_delta_meters", 0.035)),
        target_near=float(values.get("target_near_meters", 0.12)),
    )


def _goal_flags(
    state: dict[str, Any], expected: tuple[tuple[str, ...], ...]
) -> list[bool]:
    raw = state.get("goal_predicates", [])
    records = raw if isinstance(raw, list) else []
    by_key: dict[tuple[str, ...], bool] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        predicate = record.get("predicate")
        if isinstance(predicate, list) and all(
            isinstance(item, str) for item in predicate
        ):
            by_key[tuple(item.lower() for item in predicate)] = bool(
                record.get("satisfied", False)
            )
    flags = []
    for predicate in expected:
        key = tuple(item.lower() for item in predicate)
        if key not in by_key:
            raise ValueError(f"missing privileged goal predicate: {predicate}")
        flags.append(by_key[key])
    return flags


def _objects(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("objects")
    if not isinstance(raw, dict):
        raise ValueError("privileged state is missing objects")
    return raw


def _entity(state: dict[str, Any], name: str) -> dict[str, Any]:
    entity = _objects(state).get(name)
    if not isinstance(entity, dict):
        raise ValueError(f"privileged state is missing entity {name}")
    return entity


def _fixture_entity_name(state: dict[str, Any], configured_name: str) -> str:
    """Resolve a LIBERO drawer region to its articulated fixture body."""

    objects = _objects(state)
    for suffix in ("_top_region", "_bottom_region", "_heating_region"):
        if configured_name.endswith(suffix):
            fixture_name = configured_name[: -len(suffix)]
            if fixture_name in objects:
                return fixture_name
    if configured_name in objects:
        return configured_name
    raise ValueError(f"privileged state is missing fixture {configured_name}")


def _position(entity: dict[str, Any], name: str) -> npt.NDArray[np.float64]:
    raw = entity.get("position")
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"invalid position for {name}: {value}")
    return value


def _eef_position(state: dict[str, Any]) -> npt.NDArray[np.float64]:
    value = np.asarray(state.get("eef_position"), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"invalid eef_position: {value}")
    return value


def _eef_distance(state: dict[str, Any], entity_name: str) -> float:
    return float(
        np.linalg.norm(
            _eef_position(state) - _position(_entity(state, entity_name), entity_name)
        )
    )


def _object_features(
    state: dict[str, Any],
    initial: dict[str, Any],
    object_name: str,
    target_name: str,
) -> dict[str, Any]:
    entity = _entity(state, object_name)
    target = _entity(state, target_name)
    current_position = _position(entity, object_name)
    initial_entity = _entity(initial, object_name) if initial else entity
    initial_position = _position(initial_entity, object_name)
    features: dict[str, Any] = {
        "object": object_name,
        "target": target_name,
        "eef_distance": float(np.linalg.norm(_eef_position(state) - current_position)),
        "target_distance": float(
            np.linalg.norm(current_position - _position(target, target_name))
        ),
        "lift_delta": float(current_position[2] - initial_position[2]),
        "contact": bool(entity.get("contact_with_gripper", False)),
    }
    return features


def _bounded_ratio(value: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("ratio denominator must be positive")
    return min(max(value / denominator, 0.0), 1.0)


def _bounded_inverse(value: float, lower: float, upper: float) -> float:
    if not all(math.isfinite(item) for item in (value, lower, upper)):
        raise ValueError("distance values must be finite")
    if upper <= lower:
        raise ValueError("upper distance must exceed lower distance")
    return min(max((upper - value) / (upper - lower), 0.0), 1.0)


def _label(
    stage_id: int,
    stage_names: tuple[str, ...],
    completion: float,
    terminal_success: bool,
    terminal_failure: bool,
    evidence: dict[str, Any],
) -> StageLabel:
    bounded = min(max(float(completion), 0.0), 1.0)
    return StageLabel(
        stage_id=stage_id,
        stage_name=stage_names[stage_id],
        stage_completion=bounded,
        overall_progress=compose_progress(
            stage_id,
            len(stage_names),
            bounded,
            terminal_success=terminal_success,
        ),
        terminal_success=terminal_success,
        terminal_failure=terminal_failure,
        evidence=evidence,
    )
