"""Run the standalone CPU/GPU property gate for P0.1 action bounds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

import torch

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.actions import (
    ActionTransformError,
    BoundedActionTransform,
    action_bounds_from_contract,
)


class BoundedTransformValidationError(RuntimeError):
    """Raised when a bounded-output property does not hold."""


def _fail(reason: str) -> NoReturn:
    raise BoundedTransformValidationError(reason)


def _mapping(path: Path) -> Mapping[str, object]:
    value = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    if not isinstance(value, Mapping):
        _fail("expected JSON object")
    return cast(Mapping[str, object], value)


def validate(
    *, config_path: Path, contract_path: Path, output: Path, device: str
) -> Mapping[str, object]:
    config = _mapping(config_path)
    contract = _mapping(contract_path)
    action = contract.get("action")
    if config.get(
        "schema_version"
    ) != "pickcube-act-bounded-transform-validation-v1" or not isinstance(
        action, Mapping
    ):
        _fail("configuration or action contract differs")
    transform = BoundedActionTransform(
        action_bounds_from_contract(action), eps=float(cast(float, config["eps"]))
    ).to(torch.device(device))
    count = config.get("random_finite_case_count")
    extremes = config.get("extreme_values")
    if type(count) is not int or count < 10000 or not isinstance(extremes, Sequence):
        _fail("property coverage was weakened")
    generator = torch.Generator(device=device).manual_seed(0)
    random_values = (
        torch.randn((count, transform.dimension), generator=generator, device=device)
        * 25.0
    )
    extreme_values = torch.tensor(list(extremes), dtype=torch.float32, device=device)
    extreme_grid = extreme_values[:, None].repeat(1, transform.dimension)
    raw = torch.cat((random_values, extreme_grid), dim=0).requires_grad_(True)
    canonical = transform.squash(raw)
    native = transform.to_environment(canonical)
    if not torch.all(torch.abs(canonical) < 1.0):
        _fail("finite logits did not remain strictly interior")
    if not torch.all(native > transform.lower) or not torch.all(
        native < transform.upper
    ):
        _fail("finite logits did not remain inside native bounds")
    native.square().mean().backward()
    if raw.grad is None or not torch.isfinite(raw.grad).all():
        _fail("backward pass produced nonfinite gradients")
    failures = 0
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            transform.squash(torch.full((1, transform.dimension), value, device=device))
        except ActionTransformError:
            failures += 1
    if failures != 3:
        _fail("nonfinite inputs did not fail closed")
    result = {
        "case_count": int(raw.shape[0]),
        "device": device,
        "gradient_finite": True,
        "maximum_canonical_absolute_value": float(
            torch.abs(canonical).max().detach().cpu()
        ),
        "native_boundary_violation_count": 0,
        "nonfinite_fail_closed_count": failures,
        "passed": True,
        "schema_version": "pickcube-act-bounded-transform-validation-report-v1",
        "transform": transform.to_mapping(),
    }
    write_atomic_json(Path(output).absolute(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pickcube_act_bounded/action_transform.yaml"),
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                config_path=args.config,
                contract_path=args.contract,
                output=args.output,
                device=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
