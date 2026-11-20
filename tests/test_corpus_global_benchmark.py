"""Global retrieval evaluation never treats unjudged videos as negatives."""

import pytest

from evaluation.corpus_global_benchmark import (build_dev_pool, score_pool,
                                                validate_judgments)


def _hit(rank, video_id, timestamp):
    return {"rank": rank, "video_id": video_id,
            "frame_id": f"{video_id}/{rank}", "timestamp": timestamp,
            "score": 1.0 / rank}


def test_dev_pool_uses_all_videos_and_keeps_other_video_unjudged():
    labels = [
        {"query_id": "dev-1", "split": "dev", "video_id": "source",
         "query": "walking", "category": "action", "time_windows": [[10, 15]]},
        {"query_id": "test-1", "split": "test", "video_id": "source",
         "query": "held out", "category": "action", "time_windows": [[10, 15]]},
    ]
    calls = []

    def search(query, top_k):
        calls.append((query, top_k))
        return {"hits": [_hit(1, "other", 12), _hit(2, "source", 12)],
                "total_videos": 2, "searched_videos": 2, "skipped": []}

    pool, judgments = build_dev_pool(labels, search, expected_videos=2, top_k=2)
    assert calls == [("walking", 2)]
    assert len(pool["queries"]) == 1
    assert [item["judgment"] for item in judgments["queries"][0]["hits"]] == ["unjudged", "relevant"]
    assert judgments["queries"][0]["hits"][1]["basis"] == "source_label"
    metrics = score_pool(pool, judgments)
    assert metrics["hit@1"] == {"lower": 0.0, "upper": 1.0, "resolved": 0}
    assert metrics["hit@2"] == {"lower": 1.0, "upper": 1.0, "resolved": 1}


def test_judging_other_video_resolves_hit_bounds():
    labels = [{"query_id": "q", "split": "dev", "video_id": "source",
               "query": "x", "category": "object", "time_windows": [[10, 15]]}]
    pool, judgments = build_dev_pool(
        labels, lambda *_: {"hits": [_hit(1, "other", 3), _hit(2, "source", 11)],
                            "total_videos": 2, "searched_videos": 2, "skipped": []},
        expected_videos=2, top_k=2)
    judgments["queries"][0]["hits"][0].update(
        judgment="irrelevant", basis="visual_review", reviewer="Codex",
        reviewed_at="2026-11-20", review_method="checked sampled frames")
    assert score_pool(pool, judgments)["hit@1"] == {"lower": 0.0, "upper": 0.0, "resolved": 1}
    judgments["queries"][0]["hits"][0]["judgment"] = "relevant"
    assert score_pool(pool, judgments)["hit@1"] == {"lower": 1.0, "upper": 1.0, "resolved": 1}


def test_pool_rejects_incomplete_search_coverage():
    labels = [{"query_id": "q", "split": "dev", "video_id": "source",
               "query": "x", "category": "object", "time_windows": [[1, 2]]}]
    with pytest.raises(ValueError, match="coverage"):
        build_dev_pool(labels, lambda *_: {"hits": [], "total_videos": 2,
                                           "searched_videos": 1, "skipped": []},
                       expected_videos=2)


def test_judgments_must_match_predictions_and_have_review_evidence():
    labels = [{"query_id": "q", "split": "dev", "video_id": "source",
               "query": "x", "category": "object", "time_windows": [[1, 2]]}]
    pool, judgments = build_dev_pool(
        labels, lambda *_: {"hits": [_hit(1, "other", 3)], "total_videos": 2,
                            "searched_videos": 2, "skipped": []}, expected_videos=2)
    judgments["queries"][0]["hits"][0]["judgment"] = "irrelevant"
    with pytest.raises(ValueError, match="review evidence"):
        validate_judgments(pool, judgments)
    judgments["queries"][0]["hits"][0].update(
        basis="visual_review", reviewer="Codex", reviewed_at="2026-11-20",
        review_method="checked sampled frames")
    validate_judgments(pool, judgments)
    judgments["queries"][0]["hits"][0]["review_method"] = ""
    with pytest.raises(ValueError, match="review evidence"):
        validate_judgments(pool, judgments)
    judgments["queries"][0]["hits"][0]["review_method"] = "checked sampled frames"
    judgments["queries"][0]["hits"][0]["video_id"] = "different"
    with pytest.raises(ValueError, match="does not match"):
        validate_judgments(pool, judgments)


def test_unjudged_hit_cannot_claim_review_or_source_label():
    labels = [{"query_id": "q", "split": "dev", "video_id": "source",
               "query": "x", "category": "object", "time_windows": [[1, 2]]}]
    pool, judgments = build_dev_pool(
        labels, lambda *_: {"hits": [_hit(1, "other", 3)], "total_videos": 2,
                            "searched_videos": 2, "skipped": []}, expected_videos=2)
    judgments["queries"][0]["hits"][0]["basis"] = "source_label"
    with pytest.raises(ValueError, match="unjudged"):
        validate_judgments(pool, judgments)
