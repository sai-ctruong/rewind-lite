"""Unit tests for P03 temporal moment localization (retrieval.temporal).

Covers:
- TemporalMoment, FrameScore, TemporalConfig schema & validation.
- Synthetic timeline proposals:
  * zero candidate / empty input
  * neighbor merging within max_merge_gap_s
  * split cluster when span exceeds max_duration_s
  * boundary clipping at 0.0 and video_duration
  * minimum & maximum duration enforcement
  * deterministic ordering and Temporal NMS
- TemporalBaselineRetriever exact scoring & RRF integration.
- Compatibility of TemporalMoment output with evaluation.temporal_metrics.
- Phase 1 LiteRetriever regression safety.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from ingestion.build_index import KeyframeIndex
from ingestion.schemas import KeyframeRecord, RawKeyframe
from retrieval.lite_core import EncoderSpec, LiteRetriever
from retrieval.temporal import (
    FrameScore,
    TemporalBaselineRetriever,
    TemporalConfig,
    TemporalMoment,
    propose_temporal_moments,
    score_video_frames,
    smooth_frame_scores,
)
from retrieval.video_engine import VideoIndexEntry


def test_r08_none_is_disabled_with_service_entry():
    from types import SimpleNamespace
    entry = _make_synthetic_entry("v_synth")
    cfg = TemporalConfig(dark_frame_luma_max=None)
    retriever = TemporalBaselineRetriever(entry, {"clip": EncoderSpec("model-a", "rev-a")}, config=cfg)
    vectors = {"clip": np.array([1., 0.])}
    baseline = retriever.search_temporal_vectors(vectors, video_duration=20.)
    entry._lite_service = SimpleNamespace(dark_frame_luma_max=5.,
        _video_path=lambda vid: "unused", _get_dark_frame_ids=lambda *args: set(entry.index.ids))
    assert retriever.search_temporal_vectors(vectors, video_duration=20.) == baseline
    assert retriever.search_temporal_vectors(vectors, video_duration=20., config=cfg) == baseline
from evaluation.temporal_metrics import evaluate_temporal_predictions, temporal_iou


# ---------------------------------------------------------------------------
# 1. Contract & Invariant Validation Tests
# ---------------------------------------------------------------------------

def test_frame_score_contract():
    fs = FrameScore(frame_id="f1", video_id="v1", timestamp=1.5, score=0.85)
    assert fs.frame_id == "f1"
    assert fs.video_id == "v1"
    assert fs.timestamp == 1.5
    assert fs.score == 0.85

    # Rejection of invalid values
    with pytest.raises(ValueError, match="frame_id"):
        FrameScore(frame_id="", video_id="v1", timestamp=1.0, score=0.5)
    with pytest.raises(ValueError, match="video_id"):
        FrameScore(frame_id="f1", video_id="", timestamp=1.0, score=0.5)
    with pytest.raises(ValueError, match="timestamp"):
        FrameScore(frame_id="f1", video_id="v1", timestamp=-0.1, score=0.5)
    with pytest.raises(ValueError, match="timestamp"):
        FrameScore(frame_id="f1", video_id="v1", timestamp=True, score=0.5)  # bool
    with pytest.raises(ValueError, match="timestamp"):
        FrameScore(frame_id="f1", video_id="v1", timestamp=float("nan"), score=0.5)
    with pytest.raises(ValueError, match="score"):
        FrameScore(frame_id="f1", video_id="v1", timestamp=1.0, score=float("inf"))


def test_temporal_moment_contract():
    tm = TemporalMoment(
        video_id="v1",
        t_start=1.0,
        t_end=5.0,
        rank=1,
        score=0.92,
        evidence_frame_ids=("f1", "f2"),
    )
    assert tm.video_id == "v1"
    assert tm.t_start == 1.0
    assert tm.t_end == 5.0
    assert tm.rank == 1
    assert tm.score == pytest.approx(0.92)
    assert tm.evidence_frame_ids == ("f1", "f2")
    assert tm.score_type == "temporal_relevance"
    assert tm.detections is None

    # Serialization matches evaluation schema
    d = tm.to_dict()
    assert d["video_id"] == "v1"
    assert d["t_start"] == 1.0
    assert d["t_end"] == 5.0
    assert d["rank"] == 1
    assert d["score"] == pytest.approx(0.92)
    assert d["evidence_frame_ids"] == ["f1", "f2"]
    assert d["detections"] is None

    # Rejection of invalid moments
    with pytest.raises(ValueError, match="video_id"):
        TemporalMoment(video_id="", t_start=1.0, t_end=2.0, rank=1, score=0.5, evidence_frame_ids=("f1",))
    with pytest.raises(ValueError, match="invalid interval"):
        TemporalMoment(video_id="v1", t_start=5.0, t_end=2.0, rank=1, score=0.5, evidence_frame_ids=("f1",))
    with pytest.raises(ValueError, match="invalid interval"):
        TemporalMoment(video_id="v1", t_start=2.0, t_end=2.0, rank=1, score=0.5, evidence_frame_ids=("f1",))
    with pytest.raises(ValueError, match="invalid interval"):
        TemporalMoment(video_id="v1", t_start=-0.5, t_end=2.0, rank=1, score=0.5, evidence_frame_ids=("f1",))
    with pytest.raises(ValueError, match="rank"):
        TemporalMoment(video_id="v1", t_start=1.0, t_end=2.0, rank=0, score=0.5, evidence_frame_ids=("f1",))
    with pytest.raises(ValueError, match="rank"):
        TemporalMoment(video_id="v1", t_start=1.0, t_end=2.0, rank=True, score=0.5, evidence_frame_ids=("f1",))
    with pytest.raises(ValueError, match="evidence_frame_ids"):
        TemporalMoment(video_id="v1", t_start=1.0, t_end=2.0, rank=1, score=0.5, evidence_frame_ids=())
    with pytest.raises(ValueError, match="duplicates"):
        TemporalMoment(video_id="v1", t_start=1.0, t_end=2.0, rank=1, score=0.5, evidence_frame_ids=("f1", "f1"))
    with pytest.raises(ValueError, match="detections"):
        TemporalMoment(video_id="v1", t_start=1.0, t_end=2.0, rank=1, score=0.5, evidence_frame_ids=("f1",), detections=["not-a-dict"])


def test_temporal_config_validation():
    cfg = TemporalConfig()
    assert cfg.smoothing_window == 3
    assert cfg.score_quantile == 0.70
    assert cfg.top_k == 5

    with pytest.raises(ValueError, match="smoothing_window"):
        TemporalConfig(smoothing_window=0)
    with pytest.raises(ValueError, match="score_quantile"):
        TemporalConfig(score_quantile=1.5)
    with pytest.raises(ValueError, match="min_duration_s"):
        TemporalConfig(min_duration_s=-1.0)
    with pytest.raises(ValueError, match="max_duration_s"):
        TemporalConfig(min_duration_s=10.0, max_duration_s=5.0)
    with pytest.raises(ValueError, match="nms_tiou_threshold"):
        TemporalConfig(nms_tiou_threshold=1.5)
    with pytest.raises(ValueError, match="relative_peak_threshold"):
        TemporalConfig(relative_peak_threshold=1.5)
    with pytest.raises(ValueError, match="relative_peak_threshold"):
        TemporalConfig(relative_peak_threshold=-0.1)
    with pytest.raises(ValueError, match="top_k"):
        TemporalConfig(top_k=0)


def test_r01_relative_peak_threshold_default_and_tail_trimming():
    # Default tuned on golden lite sweep (thr=0.50 achieves mIoU@1=0.581 > thr=0.0 at 0.542)
    default_cfg = TemporalConfig()
    assert default_cfg.relative_peak_threshold == pytest.approx(0.50)

    # Cluster with a long low-score tail (t=2..5 around 0.22) connected to a high peak (t=6..8 around 0.90)
    frames = [
        FrameScore("f0", "v", 0.0, 0.02),
        FrameScore("f1", "v", 1.0, 0.02),
        FrameScore("f2", "v", 2.0, 0.22),
        FrameScore("f3", "v", 3.0, 0.23),
        FrameScore("f4", "v", 4.0, 0.24),
        FrameScore("f5", "v", 5.0, 0.25),
        FrameScore("f6", "v", 6.0, 0.85),
        FrameScore("f7", "v", 7.0, 0.95),
        FrameScore("f8", "v", 8.0, 0.88),
        FrameScore("f12", "v", 12.0, 0.02),
        FrameScore("f15", "v", 15.0, 0.02),
    ]

    # When thr=0.0 (disabled), the merged cluster keeps the low-score tail [f2..f8]
    ms_untrimmed = propose_temporal_moments(
        frames,
        video_duration=20.0,
        config=TemporalConfig(smoothing_window=1, score_quantile=0.35, relative_peak_threshold=0.0),
    )
    assert len(ms_untrimmed) >= 1
    top_untrimmed = ms_untrimmed[0]
    assert top_untrimmed.t_start <= 2.0
    assert "f2" in top_untrimmed.evidence_frame_ids
    assert top_untrimmed.peak_frame_id == "f7"
    assert top_untrimmed.t_start <= top_untrimmed.peak_timestamp <= top_untrimmed.t_end

    # When thr=0.5 (default), the low-score tail (< 0.5 * 0.95 = 0.475) is trimmed off
    ms_trimmed = propose_temporal_moments(
        frames,
        video_duration=20.0,
        config=TemporalConfig(smoothing_window=1, score_quantile=0.35, relative_peak_threshold=0.5),
    )
    assert len(ms_trimmed) >= 1
    top_trimmed = ms_trimmed[0]
    assert top_trimmed.t_start >= 5.0
    assert "f2" not in top_trimmed.evidence_frame_ids
    assert ("f6", "f7", "f8") == top_trimmed.evidence_frame_ids
    assert top_trimmed.peak_frame_id == "f7"
    assert top_trimmed.t_start <= top_trimmed.peak_timestamp <= top_trimmed.t_end


# ---------------------------------------------------------------------------
# 2. Synthetic Timeline & Proposal Algorithm Tests
# ---------------------------------------------------------------------------

def test_empty_or_zero_candidate():
    moments = propose_temporal_moments([], video_duration=30.0)
    assert moments == []

    # All frames below min_score_floor
    frames = [
        FrameScore("f1", "v", 1.0, 0.1),
        FrameScore("f2", "v", 2.0, 0.12),
        FrameScore("f3", "v", 3.0, 0.08),
    ]
    cfg = TemporalConfig(min_score_floor=0.5)
    moments = propose_temporal_moments(frames, video_duration=10.0, config=cfg)
    assert moments == []


def test_neighbor_merging_and_peak_selection():
    # Frames: event at 5.0, 6.0, 7.0 (gap = 1.0s <= 2.5s -> should merge)
    # Another event at 20.0, 21.0
    frames = [
        FrameScore("f1", "v", 1.0, 0.1),
        FrameScore("f2", "v", 2.0, 0.15),
        FrameScore("f5", "v", 5.0, 0.85),
        FrameScore("f6", "v", 6.0, 0.95),
        FrameScore("f7", "v", 7.0, 0.80),
        FrameScore("f12", "v", 12.0, 0.2),
        FrameScore("f20", "v", 20.0, 0.75),
        FrameScore("f21", "v", 21.0, 0.70),
        FrameScore("f25", "v", 25.0, 0.1),
    ]
    cfg = TemporalConfig(
        score_quantile=0.6,
        max_merge_gap_s=2.5,
        min_duration_s=1.0,
        max_duration_s=10.0,
        top_k=3,
    )
    moments = propose_temporal_moments(frames, video_duration=30.0, config=cfg)
    assert len(moments) >= 2
    top1 = moments[0]
    # Peak at f6 (6.0s) has highest score -> should be rank 1
    assert top1.rank == 1
    assert "f6" in top1.evidence_frame_ids
    assert 4.0 <= top1.t_start <= 5.5
    assert 6.5 <= top1.t_end <= 8.5
    assert 0.0 <= top1.t_start < top1.t_end <= 30.0

    # Rank 2 should cover the second event around 20-21s
    top2 = moments[1]
    assert top2.rank == 2
    assert "f20" in top2.evidence_frame_ids or "f21" in top2.evidence_frame_ids
    assert 18.0 <= top2.t_start <= 20.5
    assert 20.5 <= top2.t_end <= 23.0


def test_long_cluster_split_at_max_duration():
    # Sustained high scores from t=2.0 to t=25.0 (span 23s > max_duration_s 10s)
    # with two distinct peaks at 5.0 and 20.0
    frames = []
    for t in range(1, 30):
        # Double peak Gaussian-like
        s1 = 0.9 * math.exp(-0.5 * ((t - 5.0) / 1.5) ** 2)
        s2 = 0.85 * math.exp(-0.5 * ((t - 20.0) / 1.5) ** 2)
        score = max(s1, s2, 0.05)
        frames.append(FrameScore(f"f{t}", "v", float(t), float(score)))

    cfg = TemporalConfig(
        score_quantile=0.5,
        max_merge_gap_s=2.0,
        min_duration_s=2.0,
        max_duration_s=8.0,
        nms_tiou_threshold=0.4,
        top_k=5,
    )
    moments = propose_temporal_moments(frames, video_duration=35.0, config=cfg)
    assert len(moments) >= 2
    for m in moments:
        duration = m.t_end - m.t_start
        assert duration <= 8.0 + 1e-5, f"Interval duration {duration} exceeds max_duration 8.0"
        assert duration >= 2.0 - 1e-5, f"Interval duration {duration} below min_duration 2.0"
        assert 0.0 <= m.t_start < m.t_end <= 35.0


def test_boundary_clipping_at_start_and_end():
    # Frame near 0.0
    frames = [
        FrameScore("f0", "v", 0.1, 0.9),
        FrameScore("f1", "v", 0.8, 0.8),
        FrameScore("f2", "v", 5.0, 0.1),
        FrameScore("f3", "v", 9.8, 0.95),  # near end of 10s video
        FrameScore("f4", "v", 10.0, 0.85),
    ]
    cfg = TemporalConfig(
        score_quantile=0.5,
        min_duration_s=2.0,
        max_duration_s=5.0,
        top_k=5,
    )
    moments = propose_temporal_moments(frames, video_duration=10.0, config=cfg)
    assert len(moments) >= 2
    for m in moments:
        assert m.t_start >= 0.0
        assert m.t_end <= 10.0
        assert m.t_end - m.t_start >= 2.0 - 1e-5


def test_deterministic_results_and_nms_suppression():
    frames = [
        FrameScore("f1", "v", 2.0, 0.8),
        FrameScore("f2", "v", 2.5, 0.85),
        FrameScore("f3", "v", 3.0, 0.82),
        FrameScore("f4", "v", 3.5, 0.79),
        FrameScore("f10", "v", 15.0, 0.75),
    ]
    cfg = TemporalConfig(
        nms_tiou_threshold=0.3,
        top_k=5,
    )
    run1 = propose_temporal_moments(frames, video_duration=20.0, config=cfg)
    run2 = propose_temporal_moments(frames, video_duration=20.0, config=cfg)
    assert len(run1) == len(run2)
    for m1, m2 in zip(run1, run2):
        assert m1.rank == m2.rank
        assert m1.t_start == pytest.approx(m2.t_start)
        assert m1.t_end == pytest.approx(m2.t_end)
        assert m1.score == pytest.approx(m2.score)
        assert m1.evidence_frame_ids == m2.evidence_frame_ids

    # Verify NMS: no two returned intervals have IoU >= nms_tiou_threshold
    for i in range(len(run1)):
        for j in range(i + 1, len(run1)):
            iou = temporal_iou((run1[i].t_start, run1[i].t_end),
                               (run1[j].t_start, run1[j].t_end))
            assert iou < 0.3, f"Moments {i} and {j} overlap with IoU {iou} >= 0.3"


# ---------------------------------------------------------------------------
# 3. Integration with VideoIndexEntry and evaluate_temporal_predictions
# ---------------------------------------------------------------------------

def _make_synthetic_entry(video_id="v_synth"):
    # 5 frames across 10 seconds with 2-dim embeddings
    # f1 @ 1.0s, f2 @ 2.0s, f3 @ 3.0s, f4 @ 7.0s, f5 @ 8.0s
    records = [
        KeyframeRecord("f1", video_id, 1.0, np.array([1.0, 0.0], dtype=np.float32),
                       siglip_embedding=np.array([1.0, 0.0], dtype=np.float32)),
        KeyframeRecord("f2", video_id, 2.0, np.array([0.9, 0.1], dtype=np.float32),
                       siglip_embedding=np.array([0.9, 0.1], dtype=np.float32)),
        KeyframeRecord("f3", video_id, 3.0, np.array([0.8, 0.2], dtype=np.float32),
                       siglip_embedding=np.array([0.8, 0.2], dtype=np.float32)),
        KeyframeRecord("f4", video_id, 7.0, np.array([0.0, 1.0], dtype=np.float32),
                       siglip_embedding=np.array([0.0, 1.0], dtype=np.float32)),
        KeyframeRecord("f5", video_id, 8.0, np.array([0.1, 0.9], dtype=np.float32),
                       siglip_embedding=np.array([0.1, 0.9], dtype=np.float32)),
    ]
    index = KeyframeIndex.build(records)
    raws = {r.id: RawKeyframe(id=r.id, video_id=r.video_id,
                              timestamp=r.timestamp, image_path=f"{r.id}.jpg")
            for r in records}
    return VideoIndexEntry(video_id, index, raws, len(records), len(records))


def test_temporal_baseline_retriever_cosine_and_rrf():
    entry = _make_synthetic_entry("v_synth")
    specs = {
        "clip": EncoderSpec("model-a", "rev-a"),
        "siglip": EncoderSpec("model-b", "rev-b"),
    }

    # 1. Single encoder exact search
    frame_scores, score_type = score_video_frames(
        entry, {"clip": specs["clip"]}, {"clip": np.array([1.0, 0.0], dtype=np.float32)}
    )
    assert score_type == "temporal_cosine"
    assert len(frame_scores) == 5
    # Chronological ordering check
    assert [fs.timestamp for fs in frame_scores] == [1.0, 2.0, 3.0, 7.0, 8.0]
    # f1 should have highest cosine score with [1.0, 0.0]
    scores_by_id = {fs.frame_id: fs.score for fs in frame_scores}
    assert scores_by_id["f1"] > scores_by_id["f4"]

    # 2. Two encoder RRF fusion
    retriever = TemporalBaselineRetriever(entry, specs)
    moments = retriever.search_temporal_vectors(
        {"clip": np.array([1.0, 0.0], dtype=np.float32),
         "siglip": np.array([1.0, 0.0], dtype=np.float32)},
        video_duration=10.0,
    )
    assert len(moments) >= 1
    top1 = moments[0]
    assert top1.video_id == "v_synth"
    assert top1.rank == 1
    assert "f1" in top1.evidence_frame_ids
    assert top1.score_type == "temporal_rrf"


def test_compatibility_with_evaluation_temporal_metrics():
    # End-to-end check: TemporalMoment.to_dict() fed directly to evaluate_temporal_predictions
    entry = _make_synthetic_entry("v_test")
    specs = {"clip": EncoderSpec("model-a", "rev-a")}
    retriever = TemporalBaselineRetriever(entry, specs)

    moments = retriever.search_temporal_vectors(
        {"clip": np.array([1.0, 0.0], dtype=np.float32)},
        video_duration=10.0,
    )
    assert len(moments) >= 1

    # Format predictions for query 'vi-001'
    predictions = {
        "vi-001": [m.to_dict() for m in moments]
    }

    queries = [{
        "query_id": "vi-001",
        "video_id": "v_test",
        "query_vi": "cảnh đầu video",
        "time_windows": [[0.5, 3.5]],
        "duration_s": 10.0,
        "split": "dev",
    }]

    eval_result = evaluate_temporal_predictions(queries, predictions)
    assert eval_result["n_evaluable"] == 1
    summary = eval_result["summary"]["all"]
    assert summary["n_queries"] == 1
    assert summary["n_no_prediction"] == 0
    assert summary["miou"] > 0.0  # Successfully evaluated tIoU
    assert "r@1_tiou_0.3" in summary


# ---------------------------------------------------------------------------
# 4. Phase 1 LiteRetriever Regression Safety
# ---------------------------------------------------------------------------

def test_phase1_lite_retriever_remains_unaffected():
    # Ensure LiteRetriever on Phase 1 contract behaves identically
    entry = _make_synthetic_entry("v_synth")
    specs = {"clip": EncoderSpec("model-a", "rev-a")}
    lite_retriever = LiteRetriever(entry, specs)
    results = lite_retriever.search_vectors({"clip": np.array([1.0, 0.0])}, top_k=3)

    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].frame_id == "f1"
    assert results[0].score_kind == "cosine"
    assert results[0].timestamp == 1.0


def test_r06_dark_frame_luma_max_library_default_and_filtering():
    # 1. Default is None (disabled at library level) and validates [0.0, 255.0]
    default_cfg = TemporalConfig()
    assert default_cfg.dark_frame_luma_max is None

    cfg_enabled = TemporalConfig(dark_frame_luma_max=5.0)
    assert cfg_enabled.dark_frame_luma_max == pytest.approx(5.0)

    with pytest.raises(ValueError, match="dark_frame_luma_max"):
        TemporalConfig(dark_frame_luma_max=-1.0)
    with pytest.raises(ValueError, match="dark_frame_luma_max"):
        TemporalConfig(dark_frame_luma_max=256.0)
    with pytest.raises(ValueError, match="dark_frame_luma_max"):
        TemporalConfig(dark_frame_luma_max=True)

    # 2. Timeline with a false-positive black frame (f_black at t=26.0, score=0.95, luma=0.0)
    # and a valid night-scene peak (f_night at t=12.0, score=0.88, luma=15.0)
    frames = [
        FrameScore("f0", "v", 1.0, 0.10),
        FrameScore("f_night_1", "v", 11.0, 0.82),
        FrameScore("f_night_2", "v", 12.0, 0.88),
        FrameScore("f20", "v", 20.0, 0.12),
        FrameScore("f_black", "v", 26.0, 0.95),
        FrameScore("f30", "v", 30.0, 0.10),
    ]
    lumas = {
        "f0": 120.0,
        "f_night_1": 18.0,
        "f_night_2": 15.0,
        "f20": 110.0,
        "f_black": 0.0,
        "f30": 100.0,
    }

    # With dark_frame_luma_max=None (library default), results are unchanged even if lumas/dark_ids passed
    ms_default = propose_temporal_moments(
        frames,
        video_duration=35.0,
        config=TemporalConfig(smoothing_window=1, score_quantile=0.5, dark_frame_luma_max=None),
        dark_frame_ids={"f_black"},
        frame_lumas=lumas,
    )
    ms_legacy = propose_temporal_moments(
        frames,
        video_duration=35.0,
        config=TemporalConfig(smoothing_window=1, score_quantile=0.5),
    )
    assert ms_default == ms_legacy
    assert any(m.peak_frame_id == "f_black" for m in ms_default)

    # With dark_frame_luma_max=5.0, f_black (luma=0.0) is removed before proposal while night scene (luma=15.0) stays @1
    ms_filtered = propose_temporal_moments(
        frames,
        video_duration=35.0,
        config=TemporalConfig(smoothing_window=1, score_quantile=0.5, dark_frame_luma_max=5.0),
        frame_lumas=lumas,
    )
    assert len(ms_filtered) >= 1
    assert all("f_black" not in m.evidence_frame_ids for m in ms_filtered)
    assert all(m.peak_frame_id != "f_black" for m in ms_filtered)
    assert ms_filtered[0].peak_frame_id == "f_night_2"
