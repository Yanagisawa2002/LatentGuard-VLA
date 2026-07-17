"""Exact cross-dataset leakage detection for M3A and M3C visual data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, TypeVar, cast


class VisualLeakageError(ValueError):
    """Raised when development and external source identities overlap."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualLeakageError(f"{context}: {reason}")


@dataclass(frozen=True, slots=True)
class CrossDatasetLeakageReportV1:
    """Exact overlap inventories; no perceptual similarity is used."""

    overlapping_reset_seeds: tuple[int, ...]
    overlapping_source_trajectory_ids: tuple[str, ...]
    overlapping_split_group_ids: tuple[str, ...]
    overlapping_anchor_ids: tuple[str, ...]
    overlapping_state_digests: tuple[str, ...]
    overlapping_verifier_state_digests: tuple[str, ...]
    overlapping_packet_ids: tuple[str, ...]
    overlapping_candidate_ids: tuple[str, ...]
    overlapping_image_digests: tuple[str, ...]
    constant_image_digests_requiring_review: tuple[str, ...]
    repeated_images_within_development: tuple[str, ...]
    repeated_images_within_external: tuple[str, ...]

    @property
    def cross_dataset_leakage_absent(self) -> bool:
        """Return whether every prohibited exact-overlap inventory is empty."""

        return not any(
            (
                self.overlapping_reset_seeds,
                self.overlapping_source_trajectory_ids,
                self.overlapping_split_group_ids,
                self.overlapping_anchor_ids,
                self.overlapping_state_digests,
                self.overlapping_verifier_state_digests,
                self.overlapping_packet_ids,
                self.overlapping_candidate_ids,
                self.overlapping_image_digests,
            )
        )


@dataclass(frozen=True, slots=True)
class _PacketLeakageInventory:
    source_trajectory_ids: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    state_digests: tuple[str, ...]
    verifier_state_digests: tuple[str, ...]
    packet_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    image_digests: tuple[str, ...]


def _packet_inventory(dataset: object) -> _PacketLeakageInventory:
    packets = tuple(getattr(dataset, "packets", ()))
    bindings = tuple(getattr(dataset, "candidate_bindings", ()))
    image_digests = tuple(
        cast(str, cast(Any, view).pixel_sha256)
        for packet in packets
        for view in tuple(getattr(cast(Any, packet), "views", ()))
    )
    return _PacketLeakageInventory(
        source_trajectory_ids=tuple(
            cast(str, cast(Any, item).source_trajectory_id) for item in packets
        ),
        split_group_ids=tuple(
            cast(str, cast(Any, item).split_group_id) for item in packets
        ),
        anchor_ids=tuple(cast(str, cast(Any, item).anchor_id) for item in packets),
        state_digests=tuple(
            cast(str, cast(Any, item).expected_state_digest) for item in packets
        ),
        verifier_state_digests=tuple(
            cast(str, cast(Any, item).verifier_state_digest) for item in packets
        ),
        packet_ids=tuple(cast(str, cast(Any, item).packet_id) for item in packets),
        candidate_ids=tuple(
            cast(str, cast(Any, item).candidate_sample_id) for item in bindings
        ),
        image_digests=image_digests,
    )


_Comparable = TypeVar("_Comparable", str, int)


def _intersection(
    first: tuple[_Comparable, ...], second: tuple[_Comparable, ...]
) -> tuple[_Comparable, ...]:
    return tuple(sorted(set(first).intersection(second)))


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for item in values:
        if item in seen:
            repeated.add(str(item))
        seen.add(item)
    return tuple(sorted(repeated))


def inspect_cross_dataset_leakage(
    development_dataset: object,
    external_dataset: object,
    *,
    development_reset_seeds: tuple[int, ...],
    external_reset_seeds: tuple[int, ...],
    constant_image_digests: tuple[str, ...] = (),
) -> CrossDatasetLeakageReportV1:
    """Compute exact overlap inventories without silently authorizing any."""

    left = _packet_inventory(development_dataset)
    right = _packet_inventory(external_dataset)
    image_overlap = tuple(
        str(item) for item in _intersection(left.image_digests, right.image_digests)
    )
    constants = tuple(sorted(set(image_overlap).intersection(constant_image_digests)))
    return CrossDatasetLeakageReportV1(
        overlapping_reset_seeds=tuple(
            int(item)
            for item in _intersection(development_reset_seeds, external_reset_seeds)
        ),
        overlapping_source_trajectory_ids=tuple(
            str(item)
            for item in _intersection(
                left.source_trajectory_ids, right.source_trajectory_ids
            )
        ),
        overlapping_split_group_ids=tuple(
            str(item)
            for item in _intersection(left.split_group_ids, right.split_group_ids)
        ),
        overlapping_anchor_ids=tuple(
            str(item) for item in _intersection(left.anchor_ids, right.anchor_ids)
        ),
        overlapping_state_digests=tuple(
            str(item) for item in _intersection(left.state_digests, right.state_digests)
        ),
        overlapping_verifier_state_digests=tuple(
            str(item)
            for item in _intersection(
                left.verifier_state_digests, right.verifier_state_digests
            )
        ),
        overlapping_packet_ids=tuple(
            str(item) for item in _intersection(left.packet_ids, right.packet_ids)
        ),
        overlapping_candidate_ids=tuple(
            str(item) for item in _intersection(left.candidate_ids, right.candidate_ids)
        ),
        overlapping_image_digests=image_overlap,
        constant_image_digests_requiring_review=constants,
        repeated_images_within_development=_duplicates(left.image_digests),
        repeated_images_within_external=_duplicates(right.image_digests),
    )


def validate_cross_dataset_leakage(
    development_dataset: object,
    external_dataset: object,
    *,
    development_reset_seeds: tuple[int, ...],
    external_reset_seeds: tuple[int, ...],
    constant_image_digests: tuple[str, ...] = (),
) -> CrossDatasetLeakageReportV1:
    """Return an exact report or reject any cross-dataset identity overlap."""

    report = inspect_cross_dataset_leakage(
        development_dataset,
        external_dataset,
        development_reset_seeds=development_reset_seeds,
        external_reset_seeds=external_reset_seeds,
        constant_image_digests=constant_image_digests,
    )
    if not report.cross_dataset_leakage_absent:
        fields = [
            name
            for name in report.__dataclass_fields__
            if name.startswith("overlapping_") and getattr(report, name)
        ]
        _fail("cross-dataset leakage", "overlap in " + ", ".join(fields))
    return report


__all__ = [
    "CrossDatasetLeakageReportV1",
    "VisualLeakageError",
    "inspect_cross_dataset_leakage",
    "validate_cross_dataset_leakage",
]
