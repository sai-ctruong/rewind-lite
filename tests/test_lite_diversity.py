from retrieval.lite_core import SearchResult
from ui.lite_app import diversify_keyframe_results


def _hit(rank, second):
    return SearchResult(frame_id=f"f{rank}", video_id="v", timestamp=second,
                        score=1 / rank, rank=rank, image_path=None,
                        score_kind="cosine", encoder_versions={"clip": "test"})


def test_diversity_keeps_best_hit_and_spreads_remaining_results():
    ranked = [_hit(i, second) for i, second in enumerate(
        (0.0, 0.5, 1.0, 5.0, 5.5, 10.0, 15.0), 1)]
    selected = diversify_keyframe_results(ranked, top_k=4, min_gap_s=5.0)
    assert [hit.timestamp for hit in selected] == [0.0, 5.0, 10.0, 15.0]
    assert [hit.rank for hit in selected] == [1, 2, 3, 4]
    assert selected[0].score == ranked[0].score


def test_diversity_backfills_when_video_is_too_short():
    ranked = [_hit(i, second) for i, second in enumerate((0.0, 0.5, 1.0), 1)]
    selected = diversify_keyframe_results(ranked, top_k=3, min_gap_s=5.0)
    assert [hit.frame_id for hit in selected] == ["f1", "f2", "f3"]


def test_diversity_does_not_mix_videos_or_duplicate_results():
    ranked = [_hit(1, 0.0), _hit(2, 0.5), _hit(3, 5.0)]
    other = _hit(4, 0.0)
    other = SearchResult(**{**other.to_dict(), "video_id": "other"})
    selected = diversify_keyframe_results(ranked + [other], top_k=3,
                                          min_gap_s=5.0)
    assert [hit.frame_id for hit in selected] == ["f1", "f3", "f4"]
