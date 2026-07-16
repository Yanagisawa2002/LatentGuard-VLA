"""Metadata-isolated neural inference over one preserved dataset split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latentguard.action_verifier import DatasetSplit
from latentguard.training.batching import iter_numpy_batches, to_torch_batch
from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
from latentguard.training.evaluation import EvaluationMetadataV1
from latentguard.training.preprocessing import PreprocessingStateV1


@dataclass(frozen=True, slots=True, eq=False)
class SplitInferenceResultV1:
    """Ordered logits, labels, and separate reporting joins for one split."""

    split: DatasetSplit
    logits: NDArray[np.float64]
    failure_targets: NDArray[np.int64]
    metadata: tuple[EvaluationMetadataV1, ...]

    def __post_init__(self) -> None:
        """Freeze arrays and validate exact prediction/report alignment."""

        if not isinstance(self.split, DatasetSplit):
            raise ValueError("SplitInferenceResultV1.split: invalid split")
        if (
            not isinstance(self.logits, np.ndarray)
            or self.logits.dtype != np.dtype("<f8")
            or self.logits.ndim != 1
            or not bool(np.all(np.isfinite(self.logits)))
        ):
            raise ValueError("SplitInferenceResultV1.logits: invalid array")
        if (
            not isinstance(self.failure_targets, np.ndarray)
            or self.failure_targets.dtype != np.dtype("<i8")
            or self.failure_targets.shape != self.logits.shape
            or not bool(np.all(np.isin(self.failure_targets, (0, 1))))
        ):
            raise ValueError("SplitInferenceResultV1.failure_targets: invalid array")
        metadata = tuple(self.metadata)
        if len(metadata) != self.logits.size or tuple(
            item.sample_index for item in metadata
        ) != tuple(range(self.logits.size)):
            raise ValueError("SplitInferenceResultV1.metadata: alignment differs")
        logits = np.array(self.logits, copy=True, order="C")
        targets = np.array(self.failure_targets, copy=True, order="C")
        object.__setattr__(
            self,
            "logits",
            np.frombuffer(logits.tobytes(order="C"), dtype=logits.dtype),
        )
        object.__setattr__(
            self,
            "failure_targets",
            np.frombuffer(targets.tobytes(order="C"), dtype=targets.dtype),
        )
        object.__setattr__(self, "metadata", metadata)


def infer_action_verifier_split(
    model: nn.Module,
    dataset: AcceptedActionVerifierDatasetV1,
    preprocessing: PreprocessingStateV1,
    *,
    split: DatasetSplit,
    batch_size: int,
    device: str,
) -> SplitInferenceResultV1:
    """Infer one split without ever supplying reporting metadata to the model."""

    if not isinstance(model, nn.Module):
        raise TypeError("infer_action_verifier_split.model: expected torch module")
    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        raise TypeError("infer_action_verifier_split.dataset: expected accepted data")
    if not isinstance(preprocessing, PreprocessingStateV1):
        raise TypeError("infer_action_verifier_split.preprocessing: invalid state")
    if not isinstance(split, DatasetSplit):
        raise TypeError("infer_action_verifier_split.split: invalid split")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("infer_action_verifier_split.batch_size: expected positive")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("infer_action_verifier_split.device: CUDA is unavailable")
    model.to(target_device)
    model.eval()
    logits_parts: list[NDArray[np.float64]] = []
    targets_parts: list[NDArray[np.int64]] = []
    dataset_index_parts: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for numpy_batch in iter_numpy_batches(
            dataset,
            dataset.indices_for_split(split),
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            preprocessing=preprocessing,
        ):
            converted = to_torch_batch(numpy_batch, device=str(target_device))
            state = cast(Tensor, converted.state_vectors)
            actions = cast(Tensor, converted.action_chunks)
            mask = cast(Tensor, converted.action_masks)
            logits = model(state, actions, mask)
            if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
                raise ValueError("infer_action_verifier_split: invalid model logits")
            logits_parts.append(
                np.asarray(logits.detach().cpu().numpy(), dtype=np.float64)
            )
            targets_parts.append(
                np.asarray(numpy_batch.failure_targets, dtype=np.int64)
            )
            dataset_index_parts.append(
                np.asarray(numpy_batch.sample_indices, dtype=np.int64)
            )
    logits_array = np.concatenate(logits_parts)
    targets_array = np.concatenate(targets_parts)
    dataset_indices = np.concatenate(dataset_index_parts)
    metadata: list[EvaluationMetadataV1] = []
    for prediction_index, dataset_index_value in enumerate(dataset_indices):
        report = dataset.reporting_for(int(dataset_index_value))
        metadata.append(
            EvaluationMetadataV1(
                sample_index=prediction_index,
                sample_id=report.sample_id,
                group_id=report.group_id,
                source_trajectory_id=report.source_trajectory,
                split=report.dataset_split.value,
                candidate_type=report.candidate_type.value,
                corruption_family=report.corruption_family,
                corruption_severity=report.severity,
                anchor_selection_reason=(
                    report.anchor_selection_reason
                    if report.anchor_selection_reason is not None
                    else "unavailable"
                ),
            )
        )
    return SplitInferenceResultV1(
        split=split,
        logits=np.asarray(logits_array, dtype=np.float64),
        failure_targets=np.asarray(targets_array, dtype=np.int64),
        metadata=tuple(metadata),
    )


__all__ = ["SplitInferenceResultV1", "infer_action_verifier_split"]
