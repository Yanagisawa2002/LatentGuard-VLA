"""Deterministic, non-physical reference adapter for M6A infrastructure tests."""

from __future__ import annotations

from dataclasses import dataclass

from latentguard.integrations.langmani.models import (
    LangManiActionProposalV1,
    LangManiObservationEnvelopeV1,
    LangManiObservationFieldV1,
    LangManiOutcomeEvidenceV1,
    LangManiPolicyBindingV1,
    LangManiReplaySnapshotV1,
    LangManiTaskContextV1,
    ObservationRole,
    OutcomeStatus,
    ProposalStage,
    content_digest,
)

SYNTHETIC_WARNING = (
    "Synthetic LangMani adapter validation does not establish compatibility with the "
    "real LangMani simulator or policy."
)
_ZERO_DIGEST = "sha256:" + "0" * 64


@dataclass(slots=True)
class SyntheticLangManiAdapter:
    """Small deterministic adapter that demonstrates separation and resume."""

    seed: int = 0
    step: int = 0
    completed: bool = False
    execution_count: int = 0

    def task_context(self) -> LangManiTaskContextV1:
        """Return a deterministic synthetic task context."""

        return LangManiTaskContextV1(
            task_family="synthetic_pick_place",
            instruction_text="Pick up the red cube and place it in the left bin.",
            normalized_task_id="synthetic:red_cube:left_bin:v1",
            object_id="red_cube",
            goal_id="left_bin",
            conditioning_id="synthetic-onehot-v1:index-0",
            conditioning_vector=(1.0, 0.0),
            simulator_task_configuration={"object": "red_cube", "goal": "left_bin"},
            identity_only_fields=("instruction_text", "normalized_task_id"),
            model_input_fields=("conditioning_vector",),
            reporting_only_fields=("reporting_labels",),
            reporting_labels={"fixture": True},
        )

    def policy_binding(self) -> LangManiPolicyBindingV1:
        """Return a deterministic synthetic policy identity."""

        return LangManiPolicyBindingV1(
            policy_family="synthetic_act_fixture",
            variant="per_task_fixture",
            checkpoint_fingerprint="sha256:" + "1" * 64,
            model_component_fingerprint="sha256:" + "2" * 64,
            preprocessor_fingerprint="sha256:" + "3" * 64,
            postprocessor_fingerprint="sha256:" + "4" * 64,
            producer_git_commit="5" * 40,
            conditioning_id=self.task_context().conditioning_id,
            image_shape_chw=(3, 256, 256),
            state_dimension=9,
            action_dimension=8,
            policy_chunk_size=50,
            execution_horizon=2,
            observation_steps=1,
            control_frequency_hz=20,
            normalization={
                "visual": "MEAN_STD",
                "state": "MEAN_STD",
                "action": "MEAN_STD",
            },
        )

    def observation(self) -> LangManiObservationEnvelopeV1:
        """Return a role-separated synthetic observation."""

        fields = (
            LangManiObservationFieldV1(
                name="base_camera_rgb_digest",
                dtype="uint8",
                shape=(256, 256, 3),
                role=ObservationRole.POLICY_INPUT,
                units="intensity_0_255",
                coordinate_frame="base_camera",
                normalization="policy_preprocessor_mean_std",
                update_cadence="20_hz",
                source_location="synthetic.fixture",
                determinism="deterministic_fixture",
                reconstructable_after_restore=True,
            ),
            LangManiObservationFieldV1(
                name="panda_qpos",
                dtype="float32",
                shape=(9,),
                role=ObservationRole.VERIFIER_INPUT,
                units="radian_or_meter",
                coordinate_frame="joint_coordinates",
                normalization="none",
                update_cadence="20_hz",
                source_location="synthetic.fixture",
                determinism="deterministic_fixture",
                reconstructable_after_restore=True,
            ),
            LangManiObservationFieldV1(
                name="outcome_hidden",
                dtype="bool",
                shape=(1,),
                role=ObservationRole.PROHIBITED_LEARNED,
                units="boolean",
                coordinate_frame="not_applicable",
                normalization="none",
                update_cadence="terminal_only",
                source_location="synthetic.fixture",
                determinism="deterministic_fixture",
                reconstructable_after_restore=False,
            ),
            LangManiObservationFieldV1(
                name="fixture_label",
                dtype="string",
                shape=(1,),
                role=ObservationRole.REPORTING_ONLY,
                units="not_applicable",
                coordinate_frame="not_applicable",
                normalization="none",
                update_cadence="episode",
                source_location="synthetic.fixture",
                determinism="deterministic_fixture",
                reconstructable_after_restore=True,
            ),
        )
        return LangManiObservationEnvelopeV1(
            observation_id=f"synthetic-observation-{self.step}",
            task_context_digest=self.task_context().digest,
            fields=fields,
            policy_inputs={"base_camera_rgb_digest": _ZERO_DIGEST},
            verifier_inputs={"panda_qpos": (float(self.step),) + (0.0,) * 8},
            privileged_verifier_inputs={},
            restoration_state_digest=self.snapshot().simulator_state_digest,
            reporting_metadata={"fixture_label": "infrastructure_only"},
        )

    def propose(self) -> LangManiActionProposalV1:
        """Return the same raw proposal for the same state and seed."""

        value = 1.25 if self.step == 0 else 0.25
        actions = tuple(tuple([0.0] * 7 + [value]) for _ in range(2))
        return LangManiActionProposalV1(
            proposal_id=f"synthetic-raw-{self.seed}-{self.step}",
            stage=ProposalStage.RAW,
            actions=actions,
            mask=(True, True),
            action_dtype="float32",
            horizon=2,
            action_dimension=8,
            control_mode="pd_joint_pos",
            control_period_s=0.05,
            action_lower_bounds=(-1.0,) * 8,
            action_upper_bounds=(1.0,) * 8,
            normalization="mean_std_inverted_by_saved_postprocessor",
            projection_status=("not_applied", "not_applied"),
            policy_binding_digest=self.policy_binding().digest,
            task_context_digest=self.task_context().digest,
            proposal_generation_seed=self.seed,
            source_proposal_digest=None,
        )

    def executable_proposal(
        self, proposal: LangManiActionProposalV1
    ) -> LangManiActionProposalV1:
        """Project finite values and create a distinct proposal identity."""

        if proposal.stage is not ProposalStage.RAW:
            raise ValueError("only a raw proposal may be projected")
        actions = tuple(
            tuple(
                max(low, min(value, high))
                for value, low, high in zip(
                    action,
                    proposal.action_lower_bounds,
                    proposal.action_upper_bounds,
                    strict=True,
                )
            )
            for action in proposal.actions
        )
        statuses = tuple(
            "projected" if raw != projected else "unchanged"
            for raw, projected in zip(proposal.actions, actions, strict=True)
        )
        return LangManiActionProposalV1(
            proposal_id=proposal.proposal_id.replace("-raw-", "-projected-"),
            stage=ProposalStage.PROJECTED,
            actions=actions,
            mask=proposal.mask,
            action_dtype=proposal.action_dtype,
            horizon=proposal.horizon,
            action_dimension=proposal.action_dimension,
            control_mode=proposal.control_mode,
            control_period_s=proposal.control_period_s,
            action_lower_bounds=proposal.action_lower_bounds,
            action_upper_bounds=proposal.action_upper_bounds,
            normalization=proposal.normalization,
            projection_status=statuses,
            policy_binding_digest=proposal.policy_binding_digest,
            task_context_digest=proposal.task_context_digest,
            proposal_generation_seed=proposal.proposal_generation_seed,
            source_proposal_digest=proposal.digest,
        )

    def snapshot(self) -> LangManiReplaySnapshotV1:
        """Capture fixture simulator and policy-history state."""

        payload: dict[str, object] = {
            "step": self.step,
            "completed": self.completed,
            "history": (self.step,),
        }
        return LangManiReplaySnapshotV1(
            snapshot_id=f"synthetic-snapshot-{self.step}-{int(self.completed)}",
            task_context_digest=self.task_context().digest,
            simulator_state_digest=content_digest({"step": self.step}),
            simulator_state_schema="synthetic-complete-state-v1",
            component_count=1,
            comparison_semantic="byte_exact_synthetic_v1",
            maximum_absolute_tolerance=0.0,
            restoration_mode="complete_fixture_state",
            policy_state_digest=content_digest({"history": [self.step]}),
            observation_history_digest=content_digest([self.step]),
            wrapper_state_digest=content_digest({"completed": self.completed}),
            rng_state_digest=content_digest({"seed": self.seed}),
            renderer_state_digest=None,
            state_payload=payload,  # type: ignore[arg-type]
        )

    def restore(self, snapshot: LangManiReplaySnapshotV1) -> None:
        """Restore the complete fixture state from a safe JSON snapshot."""

        payload = snapshot.state_payload
        step = payload.get("step")
        completed = payload.get("completed")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not isinstance(completed, bool)
        ):
            raise ValueError("synthetic snapshot payload is malformed")
        self.step = step
        self.completed = completed

    def evaluate_outcome(
        self, proposal: LangManiActionProposalV1
    ) -> LangManiOutcomeEvidenceV1:
        """Execute once, or return the prior zero-work completion."""

        before = self.snapshot()
        if self.completed:
            return LangManiOutcomeEvidenceV1(
                evidence_id="synthetic-zero-work-resume",
                proposal_digest=proposal.digest,
                replay_snapshot_digest=before.digest,
                status=OutcomeStatus.SUCCESS,
                official_success=True,
                official_failure=False,
                horizon_exhausted=False,
                unsafe_proxy=False,
                execution_error=None,
                executed_action_count=0,
                continuation_semantic="zero_work_resume_v1",
                simulator_verified=False,
                reporting_metadata={"resumed_without_execution": True},
            )
        if proposal.stage is not ProposalStage.PROJECTED:
            raise ValueError("synthetic execution requires a projected proposal")
        self.execution_count += 1
        self.step += sum(proposal.mask)
        self.completed = True
        return LangManiOutcomeEvidenceV1(
            evidence_id="synthetic-success",
            proposal_digest=proposal.digest,
            replay_snapshot_digest=before.digest,
            status=OutcomeStatus.SUCCESS,
            official_success=True,
            official_failure=False,
            horizon_exhausted=False,
            unsafe_proxy=False,
            execution_error=None,
            executed_action_count=sum(proposal.mask),
            continuation_semantic="execute_bounded_chunk_then_stop_v1",
            simulator_verified=False,
            reporting_metadata={"fixture": True},
        )


def validate_synthetic_adapter() -> dict[str, object]:
    """Exercise deterministic projection, blindness, restore, and zero-work resume."""

    adapter = SyntheticLangManiAdapter(seed=7)
    snapshot = adapter.snapshot()
    raw = adapter.propose()
    projected = adapter.executable_proposal(raw)
    if raw.digest == projected.digest:
        raise RuntimeError("raw and projected proposals must have distinct identities")
    if "outcome_hidden" in adapter.observation().verifier_inputs:
        raise RuntimeError("outcome metadata leaked into verifier inputs")
    first = adapter.evaluate_outcome(projected)
    executions = adapter.execution_count
    resumed = adapter.evaluate_outcome(projected)
    if adapter.execution_count != executions or resumed.executed_action_count != 0:
        raise RuntimeError("completed fixture resume executed duplicate work")
    adapter.restore(snapshot)
    if adapter.propose().digest != raw.digest:
        raise RuntimeError("restored fixture proposal changed identity")
    return {
        "passed": True,
        "raw_proposal_digest": raw.digest,
        "projected_proposal_digest": projected.digest,
        "first_status": first.status.value,
        "zero_work_resume": True,
        "blindness_validated": True,
        "warning": SYNTHETIC_WARNING,
    }


__all__ = [
    "SYNTHETIC_WARNING",
    "SyntheticLangManiAdapter",
    "validate_synthetic_adapter",
]
