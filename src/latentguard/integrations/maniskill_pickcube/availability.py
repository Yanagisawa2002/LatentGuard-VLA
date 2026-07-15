"""Lazy dependency discovery for the optional ManiSkill PickCube integration."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from latentguard.integrations.maniskill_pickcube.configuration import (
    REQUIRED_MANISKILL_VERSION,
)

VersionResolver = Callable[[str], str]
ModuleImporter = Callable[[str], ModuleType]


class ManiSkillAvailabilityError(ImportError):
    """Raised when the optional pinned simulator runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class ManiSkillDistributionVersions:
    """Installed versions required by the compatibility probe."""

    mani_skill: str
    torch: str
    sapien: str
    mplib: str
    gymnasium: str


@dataclass(frozen=True, slots=True)
class ManiSkillRuntimeModules:
    """Lazily imported modules needed only by a real simulator probe or run."""

    gymnasium: ModuleType
    mani_skill: ModuleType
    mani_skill_envs: ModuleType
    torch: ModuleType
    sapien: ModuleType
    mplib: ModuleType


def discover_distribution_versions(
    *, version_resolver: VersionResolver = importlib.metadata.version
) -> ManiSkillDistributionVersions:
    """Resolve package metadata without importing simulator or CUDA modules."""
    versions: dict[str, str] = {}
    distributions = {
        "mani_skill": "mani_skill",
        "torch": "torch",
        "sapien": "sapien",
        "mplib": "mplib",
        "gymnasium": "gymnasium",
    }
    for field_name, distribution in distributions.items():
        try:
            value = version_resolver(distribution)
        except Exception as exc:
            raise ManiSkillAvailabilityError(
                "ManiSkill PickCube integration is unavailable: missing installed "
                f"distribution {distribution!r}; install the isolated M2C probe "
                "requirements before remote execution"
            ) from exc
        if not isinstance(value, str) or not value.strip():
            raise ManiSkillAvailabilityError(
                "ManiSkill PickCube integration is unavailable: distribution "
                f"{distribution!r} returned an invalid version"
            )
        versions[field_name] = value

    mani_skill_version = versions["mani_skill"]
    if mani_skill_version != REQUIRED_MANISKILL_VERSION:
        raise ManiSkillAvailabilityError(
            "ManiSkill PickCube integration requires mani_skill=="
            f"{REQUIRED_MANISKILL_VERSION}, found {mani_skill_version}"
        )
    return ManiSkillDistributionVersions(
        mani_skill=mani_skill_version,
        torch=versions["torch"],
        sapien=versions["sapien"],
        mplib=versions["mplib"],
        gymnasium=versions["gymnasium"],
    )


def load_runtime_modules(
    *,
    module_importer: ModuleImporter = importlib.import_module,
) -> ManiSkillRuntimeModules:
    """Import optional runtime modules lazily with one descriptive error boundary."""
    loaded: dict[str, ModuleType] = {}
    names = (
        "gymnasium",
        "mani_skill",
        "mani_skill.envs",
        "torch",
        "sapien",
        "mplib",
    )
    for name in names:
        try:
            module = module_importer(name)
        except Exception as exc:
            raise ManiSkillAvailabilityError(
                "ManiSkill PickCube integration could not import optional runtime "
                f"module {name!r}; the core LatentGuard package remains usable"
            ) from exc
        if not isinstance(module, ModuleType):
            raise ManiSkillAvailabilityError(
                "ManiSkill PickCube integration module loader returned an invalid "
                f"object for {name!r}"
            )
        loaded[name] = module
    return ManiSkillRuntimeModules(
        gymnasium=loaded["gymnasium"],
        mani_skill=loaded["mani_skill"],
        mani_skill_envs=loaded["mani_skill.envs"],
        torch=loaded["torch"],
        sapien=loaded["sapien"],
        mplib=loaded["mplib"],
    )


def require_maniskill_pickcube_runtime(
    *,
    version_resolver: VersionResolver = importlib.metadata.version,
    module_importer: ModuleImporter = importlib.import_module,
) -> tuple[ManiSkillDistributionVersions, ManiSkillRuntimeModules]:
    """Verify the exact ManiSkill version, then import the optional runtime."""
    versions = discover_distribution_versions(version_resolver=version_resolver)
    modules = load_runtime_modules(module_importer=module_importer)
    return versions, modules


__all__ = [
    "ManiSkillAvailabilityError",
    "ManiSkillDistributionVersions",
    "ManiSkillRuntimeModules",
    "ModuleImporter",
    "VersionResolver",
    "discover_distribution_versions",
    "load_runtime_modules",
    "require_maniskill_pickcube_runtime",
]
