"""Report native candidate-conditioning availability without synthetic fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r0_runtime import read_json, write_json

from latentguard.adapters.vla_jepa.world_model_adapter import (
    ExternalCandidateUnsupportedError,
    reject_external_action_candidates,
)


def main() -> None:
    """Emit the explicit unsupported result required by the frozen source audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    try:
        reject_external_action_candidates()
    except ExternalCandidateUnsupportedError as exc:
        payload = {
            "schema_version": "latentguard.lg_r0.real_candidate_probe.v1",
            "status": config["expected_if_unsupported"],
            "native_external_candidate_api": False,
            "native_multi_candidate_sampling": False,
            "synthetic_candidates_used": False,
            "reason": str(exc),
            "source_fact": (
                "the predictor receives Qwen special action-token hidden states; "
                "the official API does not accept external numeric action chunks"
            ),
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
