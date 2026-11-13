"""Unit tests for golden evaluation script and dataset schema (tests.test_eval_golden_lite).

Validates:
- Schema, integrity, and constraints of evaluation/golden_lite_queries.json
- Metric calculation functions (@1, @3, mIoU@1, R@1(tIoU>=0.5), median latency) using mock data
- Override parser and CLI flag handling without requiring heavy models or video files.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.eval_golden_lite import (
    best_tiou,
    build_arg_parser,
    compute_mode_metrics,
    is_timestamp_hit,
    load_golden_queries,
    parse_override_dict,
)

_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_JSON = _ROOT / "evaluation" / "golden_lite_queries.json"


def test_r08_runner_resolves_only_explicit_dark_threshold(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from retrieval.temporal import TemporalConfig
    from scripts import eval_golden_lite as runner
    factory = Mock()
    monkeypatch.setattr(runner, "TemporalBaselineRetriever", factory)
    service = SimpleNamespace(dark_frame_luma_max=5., _get_dark_frame_ids=Mock(return_value={"black"}))
    off = TemporalConfig(dark_frame_luma_max=None)
    runner.build_temporal_retriever(service, "entry", {}, "v", "path", off)
    service._get_dark_frame_ids.assert_not_called()
    assert factory.call_args.kwargs["config"] == off
    assert factory.call_args.kwargs["dark_frame_ids"] == frozenset()
    on = TemporalConfig(dark_frame_luma_max=5.)
    runner.build_temporal_retriever(service, "entry", {}, "v", "path", on)
    service._get_dark_frame_ids.assert_called_once_with("v", "path", "entry", 5.)
    assert factory.call_args.kwargs["config"] == on
    assert factory.call_args.kwargs["dark_frame_ids"] == {"black"}


# ---------------------------------------------------------------------------
# 1. Dataset Schema & Invariant Tests
# ---------------------------------------------------------------------------

def test_golden_queries_file_exists_and_valid():
    assert _GOLDEN_JSON.is_file(), f"Missing {_GOLDEN_JSON}"
    queries = load_golden_queries(_GOLDEN_JSON)
    assert len(queries) == 14, f"Expected exactly 14 golden queries, got {len(queries)}"

    expected_ids = [f"TC{i:02d}" for i in range(14)]
    actual_ids = [q["id"] for q in queries]
    assert actual_ids == expected_ids, f"Mismatch in query IDs: {actual_ids}"

    valid_vids = {
        "cd004f04834e466791e25c202ef4756f",
        "959c2799bf8c4ca0b380399c813dc051",
    }

    for item in queries:
        assert isinstance(item["id"], str) and item["id"].strip()
        assert item["video_id"] in valid_vids, f"Invalid video_id {item['video_id']!r} in {item['id']}"
        assert isinstance(item["query_vi"], str) and len(item["query_vi"].strip()) > 0
        assert isinstance(item["gt_windows"], list)
        assert isinstance(item["note"], str) and len(item["note"].strip()) > 0

        # Validate intervals
        for win in item["gt_windows"]:
            assert isinstance(win, list) and len(win) == 2
            start, end = float(win[0]), float(win[1])
            assert 0.0 <= start < end, f"Invalid interval [{start}, {end}] in {item['id']}"

    # TC13 must be a negative/absent query with empty gt_windows
    tc13 = queries[-1]
    assert tc13["id"] == "TC13"
    assert tc13["gt_windows"] == []
    assert "con mèo" in tc13["query_vi"]

    # All other 13 queries must have at least one ground-truth window
    for q in queries[:-1]:
        assert len(q["gt_windows"]) >= 1, f"Query {q['id']} expected non-empty gt_windows"


# ---------------------------------------------------------------------------
# 2. Pure Unit Tests for Evaluation Math & Parsers
# ---------------------------------------------------------------------------

def test_is_timestamp_hit():
    gt = [[10.0, 20.0], [50.0, 60.0]]
    # Exact inside
    assert is_timestamp_hit(15.0, gt) is True
    # Boundary with 1.0s tolerance
    assert is_timestamp_hit(9.0, gt, tol=1.0) is True
    assert is_timestamp_hit(21.0, gt, tol=1.0) is True
    assert is_timestamp_hit(49.0, gt, tol=1.0) is True
    # Outside tolerance
    assert is_timestamp_hit(8.9, gt, tol=1.0) is False
    assert is_timestamp_hit(21.1, gt, tol=1.0) is False
    assert is_timestamp_hit(35.0, gt, tol=1.0) is False
    # Empty ground truth
    assert is_timestamp_hit(15.0, []) is False


def test_best_tiou():
    gt = [[10.0, 20.0], [50.0, 60.0]]
    # Exact match: [10, 20] vs [10, 20] -> IoU = 1.0
    assert best_tiou((10.0, 20.0), gt) == pytest.approx(1.0)
    # Half overlap: [10, 20] vs [15, 25] -> inter=5, union=15 -> 1/3
    assert best_tiou((15.0, 25.0), gt) == pytest.approx(5.0 / 15.0)
    # Disjoint -> 0.0
    assert best_tiou((25.0, 30.0), gt) == pytest.approx(0.0)
    # Empty GT -> 0.0
    assert best_tiou((10.0, 20.0), []) == 0.0


def test_parse_override_dict():
    raw = [
        "relative_peak_threshold=0.0",
        "smoothing_window=5",
        "enabled=true",
        "include_local_peaks=false",
        "min_score_floor=none",
        "custom_str=hello_world",
    ]
    parsed = parse_override_dict(raw)
    assert parsed["relative_peak_threshold"] == pytest.approx(0.0)
    assert isinstance(parsed["relative_peak_threshold"], float)
    assert parsed["smoothing_window"] == 5
    assert isinstance(parsed["smoothing_window"], int)
    assert parsed["enabled"] is True
    assert parsed["include_local_peaks"] is False
    assert parsed["min_score_floor"] is None
    assert parsed["custom_str"] == "hello_world"

    with pytest.raises(ValueError, match="Invalid override format"):
        parse_override_dict(["invalid_string_without_equal"])


def test_compute_mode_metrics_synthetic():
    # 3 evaluable queries + 1 negative query (empty gt_windows)
    eval_rows = [
        {
            "id": "Q1",
            "gt_windows": [[10.0, 20.0]],
            "hit_at_1": True,
            "hit_at_3": True,
            "tiou_at_1": 0.8,
            "latency_ms": 20.0,
        },
        {
            "id": "Q2",
            "gt_windows": [[30.0, 40.0]],
            "hit_at_1": False,
            "hit_at_3": True,
            "tiou_at_1": 0.4,
            "latency_ms": 30.0,
        },
        {
            "id": "Q3",
            "gt_windows": [[50.0, 60.0]],
            "hit_at_1": True,
            "hit_at_3": True,
            "tiou_at_1": 0.6,
            "latency_ms": 40.0,
        },
        {
            "id": "Q4_absent",
            "gt_windows": [],  # negative query
            "hit_at_1": False,
            "hit_at_3": False,
            "tiou_at_1": None,
            "latency_ms": 25.0,
        },
    ]

    metrics = compute_mode_metrics(eval_rows)
    assert metrics["n_evaluable"] == 3
    # Hit@1: 2 / 3 = 0.667
    assert metrics["hit_at_1"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert metrics["hit_at_1_count"] == 2
    # Hit@3: 3 / 3 = 1.0
    assert metrics["hit_at_3"] == pytest.approx(1.0)
    assert metrics["hit_at_3_count"] == 3
    # mIoU@1: (0.8 + 0.4 + 0.6) / 3 = 0.60
    assert metrics["miou_at_1"] == pytest.approx(0.60, abs=1e-3)
    # R@1(tIoU>=0.5): 2 out of 3 (Q1=0.8, Q3=0.6 >= 0.5)
    assert metrics["r1_tiou_05"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert metrics["r1_tiou_05_count"] == 2
    # Median latency across all 4 queries: median([20, 30, 40, 25]) = 27.5
    assert metrics["median_latency_ms"] == pytest.approx(27.5, abs=1e-1)


def test_cli_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.queries == "evaluation/golden_lite_queries.json"
    assert args.out is None
    assert args.no_yolo is False
    assert args.temporal == []
    assert args.rerank == []
    assert args.top_k == 5
