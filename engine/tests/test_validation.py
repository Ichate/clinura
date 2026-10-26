from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clinura.replay import compare_to_reference


def test_compare_to_reference_uses_wall_time(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    analysis.write_text(
        json.dumps({"timestamp": 12.0, "wall_time": 1000.0, "bpm": 70.0}) + "\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        json.dumps({"timestamp": 1001.0, "bpm": 68.0}) + "\n",
        encoding="utf-8",
    )
    metrics = compare_to_reference(analysis, reference, tolerance_bpm=5.0)
    assert metrics["n"] == 1.0
    assert abs(metrics["mae"] - 2.0) < 1e-9
    assert metrics["within_tolerance_pct"] == 100.0


def test_compare_to_reference_without_matches(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    analysis.write_text(json.dumps({"timestamp": 1.0, "bpm": None}) + "\n", encoding="utf-8")
    reference = tmp_path / "reference.jsonl"
    reference.write_text(json.dumps({"timestamp": 1.0, "bpm": 70.0}) + "\n", encoding="utf-8")
    metrics = compare_to_reference(analysis, reference)
    assert metrics["n"] == 0.0
    assert np.isnan(metrics["mae"])
