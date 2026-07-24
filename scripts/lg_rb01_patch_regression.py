"""CPU-safe regression probes for the RoboLab recorded-config overlay patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any


def _predicate(*, env: object | None = None, object: str | None = None) -> bool:
    return env is not None and object is not None


class _Subtask:
    def __init__(self) -> None:
        self.conditions = {"banana": [(partial(_predicate, object="banana"), 1.0)]}


class _EnvCfg:
    def __init__(self, *, live_metadata: bool) -> None:
        self.instruction: object = {"default": "Live instruction"}
        if live_metadata:
            self._instruction_variants = {"default": "Live instruction"}


def _category_present(skips: list[str], category: str, namespace: str) -> bool:
    return any(
        item.startswith(f"[{category}] ") and namespace in item for item in skips
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_env_config(robolab_root: Path) -> tuple[Any, Path]:
    """Load only the overlay module with a CPU-safe Isaac string shim."""
    source = robolab_root / "robolab/core/replay/env_config.py"
    constants_source = robolab_root / "robolab/constants.py"
    if not source.is_file() or not constants_source.is_file():
        raise ValueError("RoboLab root does not contain the expected source files")

    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []  # type: ignore[attr-defined]
    isaac_utils = types.ModuleType("isaaclab.utils")
    isaac_utils.__path__ = []  # type: ignore[attr-defined]
    isaac_string = types.ModuleType("isaaclab.utils.string")

    def unavailable_resolver(value: str) -> Any:
        raise ImportError(f"CPU regression resolver unavailable: {value}")

    isaac_string.string_to_callable = unavailable_resolver  # type: ignore[attr-defined]
    robolab = types.ModuleType("robolab")
    robolab.__path__ = [str(robolab_root / "robolab")]  # type: ignore[attr-defined]
    constants = types.ModuleType("robolab.constants")
    constants.ASSET_DIR = str(robolab_root / "assets")  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.utils": isaac_utils,
            "isaaclab.utils.string": isaac_string,
            "robolab": robolab,
            "robolab.constants": constants,
        }
    )
    specification = importlib.util.spec_from_file_location(
        "lg_rb01_env_config",
        source,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not construct the overlay module specification")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, source


def run_regressions(expect: str, robolab_root: Path) -> dict[str, Any]:
    """Run the same cases against an unpatched or patched RoboLab import."""
    env_config, source = _load_env_config(robolab_root)

    cases: dict[str, dict[str, Any]] = {}

    live_conditions = {"subtasks": [_Subtask()]}
    recorded_conditions = {
        "subtasks": [
            {
                "conditions": {
                    "banana": [
                        [
                            "functools.partial(<function _predicate at 0x1>, "
                            "object='banana')",
                            1.0,
                        ]
                    ]
                }
            }
        ]
    }
    existing_skips = env_config.apply_recorded_env_cfg(
        live_conditions, recorded_conditions
    )
    callable_preserved = callable(
        live_conditions["subtasks"][0].conditions["banana"][0][0]
    )
    cases["existing_callable_container"] = {
        "callable_preserved": callable_preserved,
        "skips": existing_skips,
    }

    live_missing = {"subtasks": [_Subtask()]}
    recorded_missing = {
        "subtasks": [
            {
                "conditions": {
                    "orange": [["example.module.predicate", 1.0]],
                }
            }
        ]
    }
    missing_skips = env_config.apply_recorded_env_cfg(live_missing, recorded_missing)
    missing_rejected = "orange" not in live_missing["subtasks"][0].conditions
    cases["missing_callable_key"] = {
        "recorded_key_rejected": missing_rejected,
        "skips": missing_skips,
    }

    original_resolver = env_config.string_to_callable
    try:
        env_config.string_to_callable = lambda _: _predicate
        live_callable = {"terminations": {"success": {"func": _predicate}}}
        callable_skips = env_config.apply_recorded_env_cfg(
            live_callable,
            {"terminations": {"success": {"func": "tests.predicate"}}},
        )
        cases["safe_importable_callable"] = {
            "live_identity_preserved": (
                live_callable["terminations"]["success"]["func"] is _predicate
            ),
            "skips": callable_skips,
        }

        def _raise(_: str) -> Any:
            raise ImportError("not importable")

        env_config.string_to_callable = _raise
        live_unimportable = {"terminations": {"success": {"func": _predicate}}}
        invalid_skips = env_config.apply_recorded_env_cfg(
            live_unimportable,
            {"terminations": {"success": {"func": "missing.predicate"}}},
        )
        cases["unimportable_callable"] = {
            "failed_closed": (
                live_unimportable["terminations"]["success"]["func"] is _predicate
            ),
            "skips": invalid_skips,
        }
    finally:
        env_config.string_to_callable = original_resolver

    live_data = {"events": {"probe": {"params": {}}}, "unknown": {}}
    data_skips = env_config.apply_recorded_env_cfg(
        live_data,
        {
            "events": {"probe": {"params": {"threshold": 3}}},
            "unknown": {"new_key": 3},
        },
    )
    cases["recorded_only_data"] = {
        "allowlisted_data_restored": (
            live_data["events"]["probe"]["params"].get("threshold") == 3
        ),
        "unknown_key_rejected": "new_key" not in live_data["unknown"],
        "skips": data_skips,
    }

    live_instruction = _EnvCfg(live_metadata=True)
    instruction_skips = env_config.apply_recorded_env_cfg(
        live_instruction,
        {
            "_instruction_variants": {"default": "Lossy recorded metadata"},
            "instruction": "Recorded resolved instruction",
        },
    )
    cases["instruction_variants"] = {
        "runtime_metadata_preserved": (
            live_instruction._instruction_variants == {"default": "Live instruction"}
        ),
        "resolved_instruction_restored": (
            live_instruction.instruction == "Recorded resolved instruction"
        ),
        "skips": instruction_skips,
    }

    live_missing_metadata = _EnvCfg(live_metadata=False)
    missing_metadata_skips = env_config.apply_recorded_env_cfg(
        live_missing_metadata,
        {
            "_instruction_variants": {"default": "Lossy recorded metadata"},
            "instruction": "Recorded resolved instruction",
        },
    )
    cases["missing_instruction_metadata"] = {
        "recorded_metadata_not_injected": not hasattr(
            live_missing_metadata, "_instruction_variants"
        ),
        "resolved_instruction_restored": (
            live_missing_metadata.instruction == "Recorded resolved instruction"
        ),
        "skips": missing_metadata_skips,
    }

    patched_expectations = {
        "existing_callable_container": callable_preserved
        and _category_present(
            existing_skips,
            "EXPECTED_RUNTIME_PRESERVATION",
            "/subtasks[0]/conditions/banana",
        ),
        "missing_callable_key": missing_rejected
        and _category_present(
            missing_skips,
            "SCHEMA_DRIFT",
            "/subtasks[0]/conditions/orange",
        ),
        "safe_importable_callable": cases["safe_importable_callable"][
            "live_identity_preserved"
        ]
        and not callable_skips,
        "unimportable_callable": cases["unimportable_callable"]["failed_closed"]
        and _category_present(
            invalid_skips,
            "INVALID_RECORDED_VALUE",
            "/terminations/success/func",
        ),
        "recorded_only_data": cases["recorded_only_data"]["allowlisted_data_restored"]
        and cases["recorded_only_data"]["unknown_key_rejected"]
        and _category_present(
            data_skips,
            "SCHEMA_DRIFT",
            "/unknown/new_key",
        ),
        "instruction_variants": cases["instruction_variants"][
            "runtime_metadata_preserved"
        ]
        and cases["instruction_variants"]["resolved_instruction_restored"]
        and _category_present(
            instruction_skips,
            "EXPECTED_RUNTIME_PRESERVATION",
            "/_instruction_variants",
        ),
        "missing_instruction_metadata": cases["missing_instruction_metadata"][
            "recorded_metadata_not_injected"
        ]
        and cases["missing_instruction_metadata"]["resolved_instruction_restored"]
        and _category_present(
            missing_metadata_skips,
            "EXPECTED_RUNTIME_PRESERVATION",
            "/_instruction_variants",
        ),
    }
    unpatched_expectations = {
        "existing_callable_container": not callable_preserved,
        "missing_callable_key": not missing_rejected,
        "safe_importable_callable": cases["safe_importable_callable"][
            "live_identity_preserved"
        ],
        "unimportable_callable": cases["unimportable_callable"]["failed_closed"],
        "recorded_only_data": cases["recorded_only_data"]["allowlisted_data_restored"]
        and not cases["recorded_only_data"]["unknown_key_rejected"],
        "instruction_variants": not cases["instruction_variants"][
            "runtime_metadata_preserved"
        ]
        and cases["instruction_variants"]["resolved_instruction_restored"],
        "missing_instruction_metadata": cases["missing_instruction_metadata"][
            "recorded_metadata_not_injected"
        ]
        and cases["missing_instruction_metadata"]["resolved_instruction_restored"],
    }
    expectations = (
        patched_expectations if expect == "patched" else unpatched_expectations
    )
    return {
        "schema_version": "lg_rb01_patch_regression_v1",
        "status": "pass" if all(expectations.values()) else "fail",
        "expected_source_state": expect,
        "env_config_sha256": _sha256(source),
        "cases": cases,
        "expectations": expectations,
        "eval_or_exec_used": False,
    }


def main() -> None:
    """Run overlay regressions and write compact evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect",
        choices=["unpatched", "patched"],
        required=True,
    )
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_regressions(args.expect, args.robolab_root.resolve())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
