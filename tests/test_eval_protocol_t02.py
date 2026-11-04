"""T02: annotation validation and scoring after sampling/dedup."""
from types import SimpleNamespace

import pytest

from evaluation.bench_retrieval import Label, evaluate_labeled, load_labels, relevant_ids
from evaluation.dataset import validate_labels


def _entry(sampled, searchable):
    raws = {name: SimpleNamespace(video_id="v", timestamp=ts)
            for name, ts in sampled.items()}
    return SimpleNamespace(raws=raws, index=SimpleNamespace(ids=searchable))


class StaticEngine:
    def __init__(self, ids):
        self.ids = ids

    def search(self, entry, query, top_k, rerank):
        return [SimpleNamespace(keyframe_id=i) for i in self.ids[:top_k]]


def test_validator_rejects_empty_overlap_bad_interval_missing_video_and_duplicate_id():
    labels = [
        {"id": "a", "query": "q", "video_id": "v", "time_windows": [[1, 2], [1.5, 3]]},
        {"id": "a", "query": "q2", "video_id": "absent", "time_window": [2, 2]},
        {"id": "c", "query": "", "video_id": "v", "time_window": [0, 1]},
    ]
    issues = validate_labels(labels, video_ids={"v"})
    assert {i.code for i in issues} >= {
        "overlapping_windows", "duplicate_id", "missing_video_id", "invalid_interval", "empty_query"
    }


def test_validator_accepts_disjoint_multiple_windows_and_half_open_boundaries():
    assert validate_labels([{"id": "a", "query": "q", "video_id": "v",
                             "time_windows": [[0, 1], [1, 2]]}], video_ids={"v"}) == []
    entry = _entry({"left": 1.0, "right": 1.5}, ["left", "right"])
    label = Label("q", "v", (0, 1), ((1.5, 2),))
    assert relevant_ids(entry, label) == {"right"}


def test_validator_rejects_empty_windows_and_nonfinite_boundary():
    labels = [
        {"query": "q", "video_id": "v", "time_windows": []},
        {"query": "q", "video_id": "v", "time_window": [0, float("inf")]},
    ]
    assert [issue.code for issue in validate_labels(labels)] == [
        "invalid_windows", "invalid_interval"]


def test_loader_accepts_disjoint_windows_and_keeps_stable_id(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text('[{"id":"q-1","query":"q","video_id":"v",'
                    '"time_windows":[[0,1],[2,3]]}]', encoding="utf-8")
    loaded = load_labels(path)
    assert loaded[0].id == "q-1"
    assert loaded[0].windows == ((0, 1), (2, 3))


def test_no_sampled_positive_counts_as_uncovered_zero_hit():
    report = evaluate_labeled(StaticEngine(["other"]), _entry({"other": 4}, ["other"]),
                              [Label("q", "v", (0, 1))], ks=(1,))
    row, agg = report["per_query"][0], report["aggregate"]
    assert (row["annotation_validity"], row["index_coverage"], row["retrieval_hit"]) == (
        "valid", "no_sampled_positive", False)
    assert agg["n_valid"] == 1 and agg["n_covered"] == 0
    assert agg["coverage"] == 0 and agg["hit@1"] == 0 and agg["mrr"] == 0


def test_dedup_removed_all_positives_is_not_rank_error():
    entry = _entry({"positive": 0.5, "other": 4}, ["other"])
    report = evaluate_labeled(StaticEngine(["other"]), entry,
                              [Label("q", "v", (0, 1))], ks=(1,))
    row = report["per_query"][0]
    assert row["n_sampled_relevant"] == 1 and row["n_relevant"] == 0
    assert row["index_coverage"] == "dedup_removed_positive"
    assert row["retrieval_hit"] is False


def test_multiple_searchable_positives_score_recall_and_rank():
    entry = _entry({"p1": 0.5, "p2": 1.5, "wrong": 4}, ["p1", "p2", "wrong"])
    report = evaluate_labeled(StaticEngine(["wrong", "p2"]), entry,
                              [Label("q", "v", (0, 1), ((1, 2),))], ks=(1, 2))
    row, agg = report["per_query"][0], report["aggregate"]
    assert row["index_coverage"] == "covered" and row["retrieval_hit"] is True
    assert row["hit@1"] == 0 and row["hit@2"] == 1
    assert row["recall@2"] == 0.5 and row["mrr"] == 0.5
    assert agg["coverage"] == 1 and agg["hit@2_covered"] == 1


def test_invalid_annotation_is_reported_without_search():
    report = evaluate_labeled(StaticEngine([]), _entry({}, []),
                              [Label("q", "v", (1, 1))], ks=(1,))
    assert report["per_query"][0]["annotation_validity"] == "invalid"
    assert report["aggregate"]["n_valid"] == 0
    assert report["aggregate"]["n_invalid"] == 1


def test_covered_positive_not_retrieved_is_rank_miss():
    entry = _entry({"positive": 0.5, "wrong": 4}, ["positive", "wrong"])
    row = evaluate_labeled(StaticEngine(["wrong"]), entry,
                           [Label("q", "v", (0, 1))], ks=(1,))["per_query"][0]
    assert row["index_coverage"] == "covered"
    assert row["retrieval_hit"] is False
    assert row["outcome"] == "retrieval_rank_miss"


def test_empty_label_set_has_zero_denominators():
    agg = evaluate_labeled(StaticEngine([]), _entry({}, []), [], ks=(1,))["aggregate"]
    assert agg["n_queries"] == 0 and agg["n_valid"] == 0
    assert agg["n_covered"] == 0 and agg["coverage"] == 0
