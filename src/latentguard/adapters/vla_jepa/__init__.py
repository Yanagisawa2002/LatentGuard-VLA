"""Read-only integration surface for the frozen LeRobot VLA-JEPA policy."""

from latentguard.adapters.vla_jepa.constants import (
    CHECKPOINT_REVISION,
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
    LIBERO_ASSET_REVISION,
    QWEN_REVISION,
    VJEPA_REVISION,
    VLA_JEPA_COMMIT,
)
from latentguard.adapters.vla_jepa.identity import (
    FileIdentity,
    IdentityValidationError,
    validate_file_identities,
)
from latentguard.adapters.vla_jepa.policy_adapter import VLAJepaAdapter
from latentguard.adapters.vla_jepa.schemas import (
    VLAJepaPolicyOutput,
    VLAJepaWorldModelOutput,
)
from latentguard.adapters.vla_jepa.world_model_adapter import (
    ExternalCandidateUnsupportedError,
    WorldModelUnavailableError,
)

__all__ = [
    "CHECKPOINT_REVISION",
    "LEROBOT_COMMIT",
    "LEROBOT_VERSION",
    "LIBERO_ASSET_REVISION",
    "QWEN_REVISION",
    "VJEPA_REVISION",
    "VLA_JEPA_COMMIT",
    "ExternalCandidateUnsupportedError",
    "FileIdentity",
    "IdentityValidationError",
    "VLAJepaAdapter",
    "VLAJepaPolicyOutput",
    "VLAJepaWorldModelOutput",
    "WorldModelUnavailableError",
    "validate_file_identities",
]
