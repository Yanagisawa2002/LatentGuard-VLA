"""Official torchvision ResNet-18 preparation and frozen feature extraction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.visual_training.config import BackboneConfigV1


class VisualBackboneError(ValueError):
    """Raised when the reviewed official backbone cannot be verified."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualBackboneError(f"{context}: {reason}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _semantic_digest(value: object, context: str) -> str:
    payload = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class BackboneManifestV1:
    """Content identity for one locally verified official weight file."""

    configuration_digest: str
    torchvision_version: str
    weight_enum_identity: str
    weight_content_digest: str
    expected_input_resolution: int
    normalization_mean: tuple[float, float, float]
    normalization_standard_deviation: tuple[float, float, float]
    output_feature_dimension: int
    preprocessing_semantic: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for name in ("configuration_digest", "weight_content_digest"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 71
                or not value.startswith("sha256:")
            ):
                _fail(f"BackboneManifestV1.{name}", "invalid digest")
        if (
            self.expected_input_resolution != 224
            or self.output_feature_dimension != 512
        ):
            _fail("BackboneManifestV1", "backbone dimensions changed")
        if self.schema_version != "1.0":
            _fail("BackboneManifestV1.schema_version", "unsupported version")

    def as_mapping(self, *, include_digest: bool = True) -> dict[str, object]:
        """Return the path-independent manifest mapping."""
        result: dict[str, object] = {
            "configuration_digest": self.configuration_digest,
            "expected_input_resolution": self.expected_input_resolution,
            "normalization_mean": list(self.normalization_mean),
            "normalization_standard_deviation": list(
                self.normalization_standard_deviation
            ),
            "output_feature_dimension": self.output_feature_dimension,
            "preprocessing_semantic": self.preprocessing_semantic,
            "schema_version": self.schema_version,
            "torchvision_version": self.torchvision_version,
            "weight_content_digest": self.weight_content_digest,
            "weight_enum_identity": self.weight_enum_identity,
        }
        if include_digest:
            result["content_digest"] = self.content_digest
        return result

    @property
    def content_digest(self) -> str:
        """Return the complete semantic backbone identity."""
        return _semantic_digest(
            self.as_mapping(include_digest=False), "M4BBackboneManifestV1"
        )


def _write_manifest(path: Path, manifest: BackboneManifestV1) -> Path:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(manifest.as_mapping(), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        loaded = load_backbone_manifest(destination)
        if loaded != manifest:
            _fail("backbone manifest", "existing content differs")
        return destination
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_backbone_manifest(path: Path) -> BackboneManifestV1:
    """Strictly reload and authenticate a backbone manifest."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualBackboneError(f"backbone manifest: {exc}") from exc
    expected = {
        "configuration_digest",
        "content_digest",
        "expected_input_resolution",
        "normalization_mean",
        "normalization_standard_deviation",
        "output_feature_dimension",
        "preprocessing_semantic",
        "schema_version",
        "torchvision_version",
        "weight_content_digest",
        "weight_enum_identity",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        _fail("backbone manifest", "unexpected or missing fields")
    mean, std = raw["normalization_mean"], raw["normalization_standard_deviation"]
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 3
        or len(std) != 3
    ):
        _fail("backbone manifest", "invalid normalization")
    result = BackboneManifestV1(
        configuration_digest=raw["configuration_digest"],
        torchvision_version=raw["torchvision_version"],
        weight_enum_identity=raw["weight_enum_identity"],
        weight_content_digest=raw["weight_content_digest"],
        expected_input_resolution=raw["expected_input_resolution"],
        normalization_mean=cast(tuple[float, float, float], tuple(mean)),
        normalization_standard_deviation=cast(tuple[float, float, float], tuple(std)),
        output_feature_dimension=raw["output_feature_dimension"],
        preprocessing_semantic=raw["preprocessing_semantic"],
        schema_version=raw["schema_version"],
    )
    if raw["content_digest"] != result.content_digest:
        _fail("backbone manifest", "content digest mismatch")
    return result


@dataclass(slots=True)
class FrozenBackboneRuntime:
    """Runtime-only frozen feature extractor; paths never enter its identity."""

    model: Any
    device: str
    mean: Any
    standard_deviation: Any
    manifest: BackboneManifestV1

    def extract(self, rgb_batch: NDArray[Any]) -> NDArray[np.float32]:
        """Extract deterministic float32 [B,512] features from uint8 RGB224."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise VisualBackboneError(
                "frozen backbone: PyTorch is unavailable"
            ) from exc
        if (
            not isinstance(rgb_batch, np.ndarray)
            or rgb_batch.dtype != np.dtype("uint8")
            or rgb_batch.ndim != 4
            or tuple(rgb_batch.shape[1:]) != (224, 224, 3)
        ):
            _fail("frozen backbone input", "expected uint8 [B,224,224,3]")
        values = torch.from_numpy(np.array(rgb_batch, copy=True)).to(self.device)
        values = values.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
        values = (values - self.mean) / self.standard_deviation
        with torch.inference_mode():
            features = self.model(values)
        if tuple(features.shape) != (rgb_batch.shape[0], 512) or not bool(
            torch.isfinite(features).all()
        ):
            _fail("frozen backbone output", "expected finite [B,512]")
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)


def prepare_backbone(
    config: BackboneConfigV1,
    manifest_path: Path,
    *,
    device: str = "cpu",
) -> FrozenBackboneRuntime:
    """Download at most once, digest, freeze, and return official ResNet-18."""
    if not isinstance(config, BackboneConfigV1):
        _fail("prepare backbone", "invalid configuration")
    try:
        import torch
        from torch import nn
        from torchvision.models import (  # type: ignore
            ResNet18_Weights,
            resnet18,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise VisualBackboneError(
            "prepare backbone: install the visual-training optional dependencies"
        ) from exc
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        _fail("prepare backbone", "CUDA is unavailable")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    checkpoint_name = Path(urlparse(weights.url).path).name
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not weight_path.is_file():
        _fail("prepare backbone", "official downloaded weight file is unavailable")
    manifest = BackboneManifestV1(
        configuration_digest=config.content_digest,
        torchvision_version=importlib.metadata.version("torchvision"),
        weight_enum_identity=config.weight_enum,
        weight_content_digest=_sha256_file(weight_path),
        expected_input_resolution=config.expected_input_resolution,
        normalization_mean=config.normalization_mean,
        normalization_standard_deviation=config.normalization_standard_deviation,
        output_feature_dimension=config.output_feature_dimension,
        preprocessing_semantic=config.preprocessing_semantic,
    )
    _write_manifest(manifest_path, manifest)
    model.fc = nn.Identity()
    model.requires_grad_(False)
    model.to(target)
    model.eval()
    mean = torch.tensor(config.normalization_mean, device=target).view(1, 3, 1, 1)
    std = torch.tensor(config.normalization_standard_deviation, device=target).view(
        1, 3, 1, 1
    )
    return FrozenBackboneRuntime(model, str(target), mean, std, manifest)


__all__ = [
    "BackboneManifestV1",
    "FrozenBackboneRuntime",
    "VisualBackboneError",
    "load_backbone_manifest",
    "prepare_backbone",
]
