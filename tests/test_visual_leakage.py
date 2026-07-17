from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from latentguard.vision_data.leakage import (
    VisualLeakageError,
    inspect_cross_dataset_leakage,
    validate_cross_dataset_leakage,
)


@dataclass(frozen=True, slots=True)
class _View:
    pixel_sha256: str


@dataclass(frozen=True, slots=True)
class _Packet:
    source_trajectory_id: str
    split_group_id: str
    anchor_id: str
    expected_state_digest: str
    verifier_state_digest: str
    packet_id: str
    views: tuple[_View, ...]


@dataclass(frozen=True, slots=True)
class _Binding:
    candidate_sample_id: str


@dataclass(frozen=True, slots=True)
class _Dataset:
    packets: tuple[_Packet, ...]
    candidate_bindings: tuple[_Binding, ...]


def _packet(prefix: str) -> _Packet:
    return _Packet(
        source_trajectory_id=f"{prefix}-trajectory",
        split_group_id=f"{prefix}-split-group",
        anchor_id=f"{prefix}-anchor",
        expected_state_digest=f"{prefix}-state",
        verifier_state_digest=f"{prefix}-verifier",
        packet_id=f"{prefix}-packet",
        views=tuple(_View(f"{prefix}-image-{index}") for index in range(3)),
    )


def _datasets(overlap: str | None = None) -> tuple[_Dataset, _Dataset]:
    development_packet = _packet("development")
    external_packet = _packet("external")
    development_binding = _Binding("development-candidate")
    external_binding = _Binding("external-candidate")
    if overlap in {
        "source_trajectory_id",
        "split_group_id",
        "anchor_id",
        "expected_state_digest",
        "verifier_state_digest",
        "packet_id",
    }:
        external_packet = replace(
            external_packet,
            **{overlap: getattr(development_packet, overlap)},
        )
    elif overlap == "candidate_sample_id":
        external_binding = replace(
            external_binding,
            candidate_sample_id=development_binding.candidate_sample_id,
        )
    elif overlap == "pixel_sha256":
        external_packet = replace(
            external_packet,
            views=(development_packet.views[0], *external_packet.views[1:]),
        )
    return (
        _Dataset((development_packet,), (development_binding,)),
        _Dataset((external_packet,), (external_binding,)),
    )


@pytest.mark.parametrize(
    ("overlap", "report_field"),
    (
        ("reset_seed", "overlapping_reset_seeds"),
        ("source_trajectory_id", "overlapping_source_trajectory_ids"),
        ("split_group_id", "overlapping_split_group_ids"),
        ("anchor_id", "overlapping_anchor_ids"),
        ("expected_state_digest", "overlapping_state_digests"),
        ("verifier_state_digest", "overlapping_verifier_state_digests"),
        ("packet_id", "overlapping_packet_ids"),
        ("candidate_sample_id", "overlapping_candidate_ids"),
        ("pixel_sha256", "overlapping_image_digests"),
    ),
)
def test_each_cross_dataset_identity_field_is_independently_rejected(
    overlap: str,
    report_field: str,
) -> None:
    development, external = _datasets(None if overlap == "reset_seed" else overlap)
    development_seeds = (17,)
    external_seeds = (17,) if overlap == "reset_seed" else (23,)

    report = inspect_cross_dataset_leakage(
        development,
        external,
        development_reset_seeds=development_seeds,
        external_reset_seeds=external_seeds,
    )
    assert getattr(report, report_field)
    assert not report.cross_dataset_leakage_absent
    with pytest.raises(VisualLeakageError, match=report_field):
        validate_cross_dataset_leakage(
            development,
            external,
            development_reset_seeds=development_seeds,
            external_reset_seeds=external_seeds,
        )


def test_constant_image_overlap_is_reported_but_never_authorized() -> None:
    development, external = _datasets("pixel_sha256")
    constant_digest = development.packets[0].views[0].pixel_sha256

    report = inspect_cross_dataset_leakage(
        development,
        external,
        development_reset_seeds=(17,),
        external_reset_seeds=(23,),
        constant_image_digests=(constant_digest,),
    )
    assert report.overlapping_image_digests == (constant_digest,)
    assert report.constant_image_digests_requiring_review == (constant_digest,)
    with pytest.raises(VisualLeakageError, match="overlapping_image_digests"):
        validate_cross_dataset_leakage(
            development,
            external,
            development_reset_seeds=(17,),
            external_reset_seeds=(23,),
            constant_image_digests=(constant_digest,),
        )


def test_repeated_pixels_across_distinct_development_states_are_reported() -> None:
    development, external = _datasets()
    first = development.packets[0]
    second = replace(
        first,
        source_trajectory_id="development-trajectory-2",
        split_group_id="development-split-group-2",
        anchor_id="development-anchor-2",
        expected_state_digest="development-state-2",
        verifier_state_digest="development-verifier-2",
        packet_id="development-packet-2",
        views=(
            first.views[0],
            _View("development-other-1"),
            _View("development-other-2"),
        ),
    )
    repeated = _Dataset(
        packets=(first, second),
        candidate_bindings=development.candidate_bindings,
    )

    report = validate_cross_dataset_leakage(
        repeated,
        external,
        development_reset_seeds=(17, 19),
        external_reset_seeds=(23,),
    )
    assert report.cross_dataset_leakage_absent
    assert report.repeated_images_within_development == (first.views[0].pixel_sha256,)
    assert report.repeated_images_within_external == ()
