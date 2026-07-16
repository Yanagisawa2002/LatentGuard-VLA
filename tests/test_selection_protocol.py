"""CPU-only checks for the frozen M3C execution protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentguard.selection.protocol import (
    SelectionProtocolError,
    load_selection_protocol,
)

_PROTOCOL = Path("configs/selection/m3c/protocol-v1.json")


def test_checked_in_protocol_freezes_disjoint_smoke_and_full_scope() -> None:
    protocol = load_selection_protocol(_PROTOCOL)
    assert protocol.smoke_requested_success_count == 6
    assert protocol.full_requested_success_count == 60
    assert protocol.smoke_starting_seed != protocol.full_starting_seed
    assert protocol.bootstrap_replicates == 2000
    assert protocol.outcomes_available_during_selection is False
    assert protocol.content_digest.startswith("sha256:")


def test_protocol_rejects_scope_drift_and_duplicate_fields(tmp_path: Path) -> None:
    raw = json.loads(_PROTOCOL.read_text(encoding="utf-8"))
    raw["full_requested_success_count"] = 59
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SelectionProtocolError, match="source contract changed"):
        load_selection_protocol(changed)

    duplicate = tmp_path / "duplicate.json"
    original = _PROTOCOL.read_text(encoding="utf-8").rstrip()
    duplicate.write_text(original[:-1] + ', "schema_version": "1.0"}', encoding="utf-8")
    with pytest.raises(SelectionProtocolError, match="duplicate field"):
        load_selection_protocol(duplicate)
