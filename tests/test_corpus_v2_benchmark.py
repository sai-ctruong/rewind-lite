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


def test_dev_experiment_excludes_all_test_queries():
    positive = [{"query_id": "p-dev", "split": "dev"},
                {"query_id": "p-test", "split": "test"}]
    absent = [{"query_id": "a-dev", "split": "dev"},
              {"query_id": "a-test", "split": "test"}]
    selected_positive, selected_absent = benchmark.select_queries(
        positive, absent, split="dev")
    assert [row["query_id"] for row in selected_positive] == ["p-dev"]
    assert [row["query_id"] for row in selected_absent] == ["a-dev"]


def test_full_benchmark_keeps_both_splits():
    positive = [{"split": "dev"}, {"split": "test"}]
    selected_positive, selected_absent = benchmark.select_queries(
        positive, [], split="all")
    assert selected_positive == positive
    assert selected_absent == []


def test_dev_experiment_can_quarantine_a_confirmed_bad_label():
    positive = [{"query_id": "p", "split": "dev"}]
    absent = [{"query_id": "bad", "split": "dev"},
              {"query_id": "good", "split": "dev"}]
    selected_positive, selected_absent = benchmark.select_queries(
        positive, absent, split="dev", exclude_query_ids={"bad"})
    assert selected_positive == positive
    assert [row["query_id"] for row in selected_absent] == ["good"]


def test_exclusions_cannot_modify_full_held_out_measurement():
    with pytest.raises(ValueError, match="dev"):
        benchmark.select_queries([], [], split="all", exclude_query_ids={"bad"})


def test_exclusion_must_identify_an_existing_dev_query():
    positive = [{"query_id": "p-test", "split": "test"}]
    absent = [{"query_id": "a-dev", "split": "dev"}]
    with pytest.raises(ValueError, match="unknown dev"):
        benchmark.select_queries(positive, absent, split="dev",
                                 exclude_query_ids={"p-test"})


def test_dev_cli_summary_reports_dev_instead_of_empty_test_metrics():
    report = {"keyframe": {"by_split": {"dev": {"hit@1": .4},
                                        "test": {"hit@1": 0.0}}},
              "temporal": {"summary": {"dev": {"miou": .2},
                                       "test": {"miou": 0.0}}},
              "presence": {"dev": {"n_present": 64},
                           "test": {"n_present": 0}}}
    summary = benchmark.cli_summary(report, split="dev")
    assert summary == {"keyframe_dev": {"hit@1": .4},
                       "temporal_dev": {"miou": .2},
                       "presence_dev": {"n_present": 64}}
