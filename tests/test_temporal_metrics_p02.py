"""P02 interval scoring; all expected values are hand calculated."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from evaluation.temporal_metrics import (
    evaluate_temporal_predictions, prepare_temporal_dataset, temporal_iou,
)


def _query(query_id="q1", *, split="dev", windows=((2.0, 6.0),)):
    return {"query_id": query_id, "video_id": "v", "query_vi": "người đi bộ",
            "split": split, "duration_s": 12.0,
            "time_windows": [list(window) for window in windows]}


def _prediction(start, end, rank=1):
    return {"video_id": "v", "t_start": start, "t_end": end, "rank": rank}


def test_temporal_iou_uses_union_and_touching_boundary_has_zero_overlap():
    assert temporal_iou((3, 7), (2, 6)) == pytest.approx(3 / 5)
    assert temporal_iou((0, 2), (2, 6)) == 0.0


@pytest.mark.parametrize("invalid", [(2, 2), (3, 2), (-1, 1),
                                      (float("nan"), 2), (True, 2)])
def test_temporal_iou_rejects_invalid_intervals(invalid):
    with pytest.raises(ValueError, match="interval"):
        temporal_iou(invalid, (0, 2))


def test_multiple_ground_truth_windows_choose_best_match_per_prediction():
    queries = [_query(windows=((0, 2), (8, 10)))]
    predictions = {"q1": [_prediction(3, 4), _prediction(8.5, 9.5, 2)]}
    report = evaluate_temporal_predictions(queries, predictions)
    row = report["per_query"][0]
    assert [p["best_tiou"] for p in row["predictions"]] == [0.0, 0.5]
    assert row["r@1_tiou_0.3"] == 0
    assert row["r@5_tiou_0.5"] == 1
    assert row["miou"] == 0.0  # rank-1 IoU, not oracle over Top-5


def test_empty_predictions_count_as_zero_in_primary_denominator():
    queries = [_query("q1"), _query("q2", split="test")]
    report = evaluate_temporal_predictions(
        queries, {"q1": [_prediction(3, 7)], "q2": []})
    assert report["summary"]["all"] == {
        "n_queries": 2, "n_no_prediction": 1,
        "r@1_tiou_0.3": 0.5, "r@1_tiou_0.5": 0.5,
        "r@5_tiou_0.3": 0.5, "r@5_tiou_0.5": 0.5,
        "miou": pytest.approx(0.3),
    }
    assert report["summary"]["test"]["miou"] == 0.0


@pytest.mark.parametrize("predictions", [
    {"q1": [_prediction(4, 4)]},
    {"q1": [_prediction(11, 13)]},
    {"q1": [{**_prediction(3, 4), "video_id": "other"}]},
    {"q1": [_prediction(3, 4, rank=2)]},
    {"unknown": []},
])
def test_invalid_predictions_are_rejected_not_silently_scored(predictions):
    with pytest.raises(ValueError):
        evaluate_temporal_predictions([_query()], predictions)


def test_preparation_reuses_source_validator_and_preserves_phase1_labels(tmp_path):
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (32, 32))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    manifest = {"schema_version": 1, "videos": [{
        "video_id": "v", "path": "v.mp4",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duration_s": 1.0, "fps": 10.0, "frame_count": 10,
        "source": "self filmed", "rights": {"status": "confirmed", "basis": "owner"},
        "split": "dev", "group_id": "v"}]}
    label = {"query_id": "q1", "video_id": "v", "query": "đi bộ",
             "time_windows": [[0.0, 0.5]], "category": "action",
             "split": "dev", "language": "vi", "paraphrase_group": "q1",
             "annotator": "A", "review_status": "verified",
             "reviewer": "B", "reviewed_at": "2026-11-09"}
    original = dict(label)
    prepared = prepare_temporal_dataset(manifest, [label], root=tmp_path)
    assert prepared["n_evaluable"] == 1
    assert prepared["n_unreviewed"] == 0
    assert prepared["queries"][0] == {
        "query_id": "q1", "video_id": "v", "query_vi": "đi bộ",
        "time_windows": [[0.0, 0.5]], "duration_s": 1.0, "split": "dev"}
    assert label == original  # no T06 label migration or mutation
    label["time_windows"] = [[0.9, 1.1]]
    with pytest.raises(ValueError, match="window_out_of_bounds"):
        prepare_temporal_dataset(manifest, [label], root=tmp_path)


def test_temporal_cli_scores_predictions_from_validated_files(tmp_path):
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (32, 32))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    manifest = {"schema_version": 1, "videos": [{
        "video_id": "v", "path": "v.mp4",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duration_s": 1.0, "fps": 10.0, "frame_count": 10,
        "source": "self filmed", "rights": {"status": "confirmed", "basis": "owner"},
        "split": "dev", "group_id": "v"}]}
    label = {"query_id": "q1", "video_id": "v", "query": "đi bộ",
             "time_windows": [[0.0, 0.5]], "category": "action",
             "split": "dev", "language": "vi", "paraphrase_group": "q1",
             "annotator": "A", "review_status": "verified",
             "reviewer": "B", "reviewed_at": "2026-11-09"}
    for name, payload in (("manifest", manifest), ("labels", [label]),
                          ("predictions", {"q1": [_prediction(0.0, 0.5)]})):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    result = subprocess.run([
        sys.executable, "-m", "evaluation.temporal_metrics",
        "--manifest", str(tmp_path / "manifest.json"),
        "--labels", str(tmp_path / "labels.json"),
        "--root", str(tmp_path),
        "--predictions", str(tmp_path / "predictions.json"),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["dataset"]["n_evaluable"] == 1
    assert output["metrics"]["summary"]["dev"]["r@1_tiou_0.5"] == 1.0
