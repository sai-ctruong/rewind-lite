"""Per-video scoring and dev-only absence calibration for corpus v2."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evaluation.corpus_v2_benchmark import calibrate_threshold, score_keyframes
import evaluation.corpus_v2_benchmark as benchmark


def _hit(frame_id, timestamp, score):
    return SimpleNamespace(frame_id=frame_id, video_id="v", timestamp=timestamp,
                           score=score, rank=1)


def test_keyframe_score_uses_only_matching_video_window():
    entry = SimpleNamespace(
        raws={"a": _hit("a", 4.0, 0), "b": _hit("b", 8.0, 0)},
        index=SimpleNamespace(ids=["a", "b"]),
    )
    label = {"video_id": "v", "time_windows": [[7.0, 9.0]]}
    scored = score_keyframes(entry, label, [_hit("a", 4.0, .9),
                                           _hit("b", 8.0, .8)])
    assert scored["n_relevant"] == 1
    assert scored["hit@1"] == 0
    assert scored["hit@5"] == 1
    assert scored["mrr"] == .5
    assert scored["top_score"] == .9


def test_dev_threshold_is_unaffected_by_test_rows():
    rows = [
        {"split": "dev", "expected_presence": "present", "top_score": .90},
        {"split": "dev", "expected_presence": "present", "top_score": .80},
        {"split": "dev", "expected_presence": "absent", "top_score": .70},
        {"split": "dev", "expected_presence": "absent", "top_score": .20},
        {"split": "test", "expected_presence": "absent", "top_score": .99},
    ]
    assert calibrate_threshold(rows)["threshold"] == pytest.approx(.80)


def test_threshold_requires_both_classes_on_dev():
    with pytest.raises(ValueError, match="both"):
        calibrate_threshold([{"split": "dev", "expected_presence": "present",
                              "top_score": .8}])


def test_report_timestamp_uses_utc_without_system_timezone_database():
    timestamp = benchmark.report_timestamp()
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)
