"""Run the bounded P0.1 static-screen survivors through fixed development gates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

from evaluate_pickcube_act import evaluate

from latentguard.control.serialization import write_atomic_json


class BoundedDevelopmentError(RuntimeError):
    """Raised when immutable screen, role, or development contracts differ."""


def _fail(context: str, reason: str) -> NoReturn:
    raise BoundedDevelopmentError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundedDevelopmentError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _ranking(summary: Mapping[str, object]) -> tuple[float, ...]:
    """Return the immutable best-first 10/30-episode ranking."""
    return (
        -float(cast(int, summary["success_count"])),
        float(cast(int, summary["action_contract_violation_count"])),
        float(cast(int, summary["simulator_error_count"])),
        float(cast(int, summary["workspace_violation_count"])),
        float(cast(int, summary["timeout_count"])),
        float(cast(str, summary["checkpoint_id"]).removeprefix("step-")),
    )


def _eligible(summary: Mapping[str, object], threshold: float) -> bool:
    return bool(
        float(cast(float, summary["success_rate"])) >= threshold
        and summary["action_contract_violation_count"] == 0
        and summary["simulator_error_count"] == 0
        and summary["workspace_violation_count"] == 0
        and summary["fixed_seed_reproduction"]["reproducible"] is True
    )


def run(
    *,
    config_path: Path,
    screen_path: Path,
    compatibility_path: Path,
    contract_path: Path,
    training_run: Path,
    output_dir: Path,
) -> Mapping[str, object]:
    """Evaluate all safe checkpoints for 10 episodes and selected ones for 30."""
    config = _mapping(config_path, "development config")
    screen = _mapping(screen_path, "static screen")
    if (
        config.get("schema_version") != "pickcube-act-bounded-development-config-v1"
        or screen.get("schema_version")
        != "pickcube-act-bounded-checkpoint-screen-report-v1"
        or screen.get("passed") is not True
    ):
        _fail("development", "config or static screening is incomplete")
    initial_count = config.get("initial_episode_count_per_valid_checkpoint")
    extended_count = config.get("extended_episode_count_per_selected_checkpoint")
    if initial_count != 10 or extended_count != 30:
        _fail("development", "episode budget changed")
    checkpoints = screen.get("checkpoints")
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)):
        _fail("static screen", "checkpoint inventory is missing")
    valid_ids = tuple(
        cast(str, item["checkpoint"])
        for item in checkpoints
        if isinstance(item, Mapping) and item.get("valid_for_development") is True
    )
    if not valid_ids:
        _fail("static screen", "no checkpoint may enter development")
    root = Path(output_dir).absolute()
    initial: dict[str, Mapping[str, object]] = {}
    for checkpoint_id in valid_ids:
        summary = evaluate(
            config_path=config_path,
            compatibility_path=compatibility_path,
            contract_path=contract_path,
            training_run=training_run,
            checkpoint=Path(training_run) / "checkpoints" / checkpoint_id,
            output_dir=root / checkpoint_id,
            evaluation_kind="development",
            episode_count_override=10,
        )
        initial[checkpoint_id] = summary

    roles = _mapping(Path(training_run) / "checkpoint_roles.json", "checkpoint roles")
    raw_roles = config.get("always_extend_roles")
    if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (str, bytes)):
        _fail("development config", "extension roles are missing")
    selected: set[str] = set()
    for role in raw_roles:
        relative = roles.get(str(role))
        if not isinstance(relative, str):
            _fail("checkpoint roles", f"role {role} is missing")
        checkpoint_id = Path(relative).name
        if checkpoint_id in valid_ids:
            selected.add(checkpoint_id)
    top_initial = min(initial, key=lambda item: _ranking(initial[item]))
    selected.add(top_initial)

    extended: dict[str, Mapping[str, object]] = {}
    for checkpoint_id in sorted(selected):
        extended[checkpoint_id] = evaluate(
            config_path=config_path,
            compatibility_path=compatibility_path,
            contract_path=contract_path,
            training_run=training_run,
            checkpoint=Path(training_run) / "checkpoints" / checkpoint_id,
            output_dir=root / checkpoint_id,
            evaluation_kind="development",
            episode_count_override=30,
        )
    threshold = float(cast(float, config["promotion_success_rate"]))
    eligible = tuple(
        checkpoint_id
        for checkpoint_id, summary in extended.items()
        if _eligible(summary, threshold)
    )
    promoted = (
        min(eligible, key=lambda item: _ranking(extended[item])) if eligible else None
    )
    report = {
        "development_gate_passed": promoted is not None,
        "extended_checkpoint_ids": sorted(extended),
        "extended_results": dict(sorted(extended.items())),
        "initial_checkpoint_count": len(initial),
        "initial_results": dict(sorted(initial.items())),
        "promoted_checkpoint_id": promoted,
        "promotion_success_rate": threshold,
        "schema_version": "pickcube-act-bounded-development-report-v1",
        "static_screen": dict(screen),
        "top_initial_checkpoint_id": top_initial,
    }
    write_atomic_json(root / "development_summary.json", report)
    write_atomic_json(
        root / "final_evaluation.json",
        {
            "promoted_checkpoint_id": promoted,
            "schema_version": "pickcube-act-bounded-final-status-v1",
            "status": (
                "AUTHORIZED_NOT_RUN"
                if promoted is not None
                else "NOT_RUN_PROMOTION_GATE_FAILED"
            ),
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pickcube_act_bounded/eval_development.yaml"),
    )
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        screen_path=args.screen,
        compatibility_path=args.compatibility_report,
        contract_path=args.contract,
        training_run=args.training_run,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
