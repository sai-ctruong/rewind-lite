"""Negative-query evidence is kept separate from temporal positive scoring."""

from evaluation.absent_dataset import validate_absent_queries


def _manifest():
    return {"schema_version": 1, "videos": [
        {"video_id": "a", "split": "dev", "group_id": "p1"},
        {"video_id": "b", "split": "test", "group_id": "p2"},
    ]}


def _negative(**changes):
    row = {
        "query_id": "abs-001", "video_id": "a",
        "query": "Chiếc ô màu tím đặt trên bàn",
        "expected_presence": "absent", "time_windows": [],
        "category": "object", "split": "dev", "language": "vi",
        "paraphrase_group": "purple-umbrella",
        "annotator": "A", "review_status": "verified",
        "reviewer": "B", "reviewed_at": "2026-11-19",
    }
    row.update(changes)
    return row


def test_valid_absent_query_has_no_temporal_window():
    assert validate_absent_queries(_manifest(), [_negative()], []) == []


def test_rejects_absent_query_with_timestamp_or_positive_collision():
    positive = [{"query_id": "pos-001", "video_id": "a",
                 "query": "Chiếc ô màu tím đặt trên bàn", "split": "dev"}]
    bad = _negative(time_windows=[[0, 1]])
    issues = validate_absent_queries(_manifest(), [bad], positive)
    assert {issue.code for issue in issues} >= {"invalid_windows", "positive_collision"}


def test_rejects_invalid_video_split_and_duplicate_id():
    bad = _negative(video_id="b", split="dev")
    issues = validate_absent_queries(_manifest(), [bad, bad], [])
    assert {issue.code for issue in issues} >= {"split_mismatch", "duplicate_query_id"}


def test_requires_review_metadata_for_verified_absence():
    issues = validate_absent_queries(_manifest(), [_negative(reviewer="")], [])
    assert "missing_reviewer" in {issue.code for issue in issues}
