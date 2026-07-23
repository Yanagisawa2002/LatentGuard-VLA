"""Read-only exposure of VLA-JEPA's training-time world-model tensors."""

from __future__ import annotations

from typing import Any

from latentguard.adapters.vla_jepa.schemas import VLAJepaWorldModelOutput


class WorldModelUnavailableError(RuntimeError):
    """Raised when a requested official world-model path is unavailable."""


class ExternalCandidateUnsupportedError(RuntimeError):
    """Raised because the official predictor has no numeric candidate-action API."""


def reject_external_action_candidates() -> None:
    """Fail explicitly instead of fabricating candidate-conditioned predictions."""

    raise ExternalCandidateUnsupportedError(
        "VLA-JEPA 0.6 conditions its video predictor on Qwen special-token "
        "representations, not externally supplied numeric action chunks"
    )


def inspect_training_world_model(
    policy: Any,
    batch: dict[str, Any],
) -> VLAJepaWorldModelOutput:
    """Expose exact tensors from the official offline training-style WM path.

    This function intentionally mirrors ``VLAJEPAModel._world_model_loss``. It
    does not call ``backward``, build an optimizer, synthesize a score, or accept
    external numeric action candidates.
    """

    import torch
    import torch.nn.functional as functional

    native_model = getattr(policy, "model", None)
    if native_model is None:
        raise WorldModelUnavailableError("policy has no native VLA-JEPA model")
    if not native_model.config.enable_world_model:
        return VLAJepaWorldModelOutput(
            current_latent=None,
            predicted_future_latents=None,
            target_future_latents=None,
            predictor_metadata={
                "available": False,
                "reason": "world_model_disabled",
                "external_numeric_candidates_supported": False,
            },
        )
    prepared = native_model_policy_inputs(policy, batch)
    videos = prepared.get("videos")
    if videos is None:
        return VLAJepaWorldModelOutput(
            current_latent=None,
            predicted_future_latents=None,
            target_future_latents=None,
            predictor_metadata={
                "available": False,
                "reason": "temporal_video_not_provided",
                "external_numeric_candidates_supported": False,
            },
        )

    with torch.inference_mode():
        _, action_tokens = native_model._encode_qwen(
            prepared["images"],
            prepared["instructions"],
            need_action_tokens=True,
        )
        if action_tokens is None:
            raise WorldModelUnavailableError("Qwen action tokens were not emitted")

        expected_views = native_model.config.jepa_tubelet_size
        if videos.shape[1] < expected_views:
            missing = expected_views - videos.shape[1]
            videos = torch.cat(
                [videos, videos[:, :1].repeat(1, missing, 1, 1, 1, 1)],
                dim=1,
            )
        elif videos.shape[1] > expected_views:
            videos = videos[:, :expected_views]

        batch_size, view_count, frame_count, channels, height, width = videos.shape
        flat = videos.reshape(
            batch_size * view_count,
            frame_count,
            channels,
            height,
            width,
        )
        video_pixels = native_model.video_processor(
            videos=list(flat),
            return_tensors="pt",
            device=native_model.video_encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]
        embeddings = native_model.video_encoder.get_vision_features(
            pixel_values_videos=video_pixels
        )
        embeddings = torch.cat(torch.chunk(embeddings, chunks=view_count, dim=0), dim=2)

        tubelet_size = native_model.video_encoder.config.tubelet_size
        encoded_steps = native_model.config.num_video_frames // tubelet_size
        if encoded_steps < 2:
            raise WorldModelUnavailableError(
                "fewer than two encoded temporal positions"
            )
        context_steps = encoded_steps - 1
        tokens_per_step = embeddings.shape[1] // encoded_steps
        current = embeddings[:, : tokens_per_step * context_steps, :]
        target = embeddings[:, tokens_per_step:, :]
        expected_action_tokens = (
            context_steps * native_model.config.num_action_tokens_per_timestep
        )
        if action_tokens.shape[1] < expected_action_tokens:
            pad = action_tokens[:, -1:].repeat(
                1,
                expected_action_tokens - action_tokens.shape[1],
                1,
            )
            action_tokens = torch.cat([action_tokens, pad], dim=1)
        condition = action_tokens[:, :expected_action_tokens].float()
        predicted = native_model.video_predictor(current.float(), condition)
        l1 = functional.l1_loss(predicted, target.float(), reduction="mean")
        predicted_temporal = predicted.float().reshape(
            batch_size,
            context_steps,
            tokens_per_step,
            predicted.shape[-1],
        )
        temporal_change = (
            (predicted_temporal[:, 1:] - predicted_temporal[:, :-1]).abs().mean()
            if context_steps > 1
            else torch.zeros((), device=predicted.device)
        )
        current_predicted_l1 = functional.l1_loss(
            predicted, current.float(), reduction="mean"
        )

    return VLAJepaWorldModelOutput(
        current_latent=current,
        predicted_future_latents=predicted,
        target_future_latents=target,
        predictor_metadata={
            "available": True,
            "path_semantic": "official_training_style_offline_diagnostic",
            "inference_default": False,
            "condition_tensor": "qwen_special_action_token_hidden_states",
            "numeric_action_chunk_consumed": False,
            "external_numeric_candidates_supported": False,
            "encoded_temporal_positions": encoded_steps,
            "context_temporal_positions": context_steps,
            "tokens_per_temporal_position": int(tokens_per_step),
            "action_token_shape": list(condition.shape),
            "action_token_dtype": str(condition.dtype),
            "action_token_norm_mean": float(condition.norm(dim=-1).mean().item()),
            "current_latent_norm_mean": float(
                current.float().norm(dim=-1).mean().item()
            ),
            "predicted_latent_norm_mean": float(
                predicted.float().norm(dim=-1).mean().item()
            ),
            "current_predicted_l1": float(current_predicted_l1.item()),
            "predicted_temporal_change_mean": float(temporal_change.item()),
            "l1_distance": float(l1.item()),
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
    )


def native_model_policy_inputs(policy: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Build the exact native training-style inputs through the official adapter."""

    prepare = getattr(policy, "_prepare_model_inputs", None)
    if prepare is None:
        raise WorldModelUnavailableError("policy does not expose _prepare_model_inputs")
    return dict(prepare(batch, training=True))
