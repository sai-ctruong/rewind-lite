"""P06 guardrail unit tests for action/relation limitations (NO-GO Option A).

Per `REWIND_LITE_IMPROVEMENT_PLAN.md` §3 P06 and `docs/phase2-p06-action-verification.md`:
1. Multi-person ambiguity: multiple `person` detections in the same frame trigger
   `multi_person_ambiguous=True`, apply `multi_person_penalty`, and flag
   unsupported clothing/action cues (`attribute:áo xanh`, `action:lấy`).
2. Same objects, wrong action: two segments with identical static object boxes
   (`person + bottle`, e.g., holding vs drinking) receive identical object scores,
   never set `action_verified=True`, and rely on the semantic embedding score to
   disambiguate the action.
3. Short action (1-second window): handles sparse 1-second events honestly
   without inventing bounding boxes or synthetic frames when index coverage is
   sparse or empty.
"""
from __future__ import annotations

import pytest

from retrieval.object_reranker import ObjectRerankConfig, rerank_temporal_moments
from retrieval.temporal import FrameScore, TemporalConfig, TemporalMoment, propose_temporal_moments


def test_p06_multi_person_ambiguity_and_unsupported_attribute_flags():
    # Segment A: 1 person + 1 bottle (unambiguous single person)
    dets_single = (
        {
            "frame_id": "f1",
            "class_name": "person",
            "confidence": 0.90,
            "bbox_norm_xyxy": (0.2, 0.1, 0.5, 0.9),
            "image_size": (100, 100),
        },
        {
            "frame_id": "f1",
            "class_name": "bottle",
            "confidence": 0.85,
            "bbox_norm_xyxy": (0.4, 0.4, 0.5, 0.6),
            "image_size": (100, 100),
        },
    )
    # Segment B: 3 people + 1 bottle in the same frame (multi-person ambiguity)
    dets_crowded = (
        {
            "frame_id": "f2",
            "class_name": "person",
            "confidence": 0.90,
            "bbox_norm_xyxy": (0.05, 0.1, 0.25, 0.9),
            "image_size": (100, 100),
        },
        {
            "frame_id": "f2",
            "class_name": "person",
            "confidence": 0.90,
            "bbox_norm_xyxy": (0.2, 0.1, 0.5, 0.9),
            "image_size": (100, 100),
        },
        {
            "frame_id": "f2",
            "class_name": "person",
            "confidence": 0.88,
            "bbox_norm_xyxy": (0.65, 0.1, 0.9, 0.9),
            "image_size": (100, 100),
        },
        {
            "frame_id": "f2",
            "class_name": "bottle",
            "confidence": 0.85,
            "bbox_norm_xyxy": (0.4, 0.4, 0.5, 0.6),
            "image_size": (100, 100),
        },
    )

    m_crowded = TemporalMoment(
        video_id="v_walk",
        t_start=2.0,
        t_end=5.0,
        rank=1,
        score=0.75,
        evidence_frame_ids=("f2",),
        detections=dets_crowded,
    )
    m_single = TemporalMoment(
        video_id="v_walk",
        t_start=8.0,
        t_end=11.0,
        rank=2,
        score=0.75,
        evidence_frame_ids=("f1",),
        detections=dets_single,
    )

    cfg = ObjectRerankConfig(
        enabled=True,
        semantic_weight=0.5,
        object_weight=0.5,
        multi_person_penalty=0.5,
    )
    reranked, prov, spec = rerank_temporal_moments(
        [m_crowded, m_single],
        "người mặc áo xanh lấy chai nước",
        config=cfg,
    )

    # Unsupported cues must explicitly flag both clothing color and action verb
    assert "attribute:áo xanh" in spec.unsupported_cues
    assert "action:lấy" in spec.unsupported_cues

    prov_by_start = {p.t_start: p for p in prov}
    assert prov_by_start[2.0].multi_person_ambiguous is True
    assert prov_by_start[8.0].multi_person_ambiguous is False
    assert prov_by_start[2.0].action_verified is False
    assert prov_by_start[8.0].action_verified is False

    # Because of multi-person ambiguity penalty, single-person frame receives higher object score
    assert prov_by_start[8.0].object_score > prov_by_start[2.0].object_score
    assert reranked[0].t_start == 8.0


def test_p06_same_objects_wrong_action_never_claims_action_verification():
    # Both segments have identical static person + bottle detections (e.g. holding vs drinking).
    # Segment 1 has higher semantic similarity for "Người uống từ chai thủy tinh".
    shared_boxes = (
        {
            "frame_id": "f_drink",
            "class_name": "person",
            "confidence": 0.92,
            "bbox_norm_xyxy": (0.2, 0.1, 0.7, 0.95),
            "image_size": (200, 200),
        },
        {
            "frame_id": "f_drink",
            "class_name": "bottle",
            "confidence": 0.88,
            "bbox_norm_xyxy": (0.4, 0.25, 0.5, 0.45),
            "image_size": (200, 200),
        },
    )
    wrong_action_boxes = (
        {
            "frame_id": "f_hold",
            "class_name": "person",
            "confidence": 0.92,
            "bbox_norm_xyxy": (0.2, 0.1, 0.7, 0.95),
            "image_size": (200, 200),
        },
        {
            "frame_id": "f_hold",
            "class_name": "bottle",
            "confidence": 0.88,
            "bbox_norm_xyxy": (0.4, 0.25, 0.5, 0.45),
            "image_size": (200, 200),
        },
    )

    m_drinking = TemporalMoment(
        video_id="video_test3",
        t_start=6.0,
        t_end=9.0,
        rank=1,
        score=0.84,
        evidence_frame_ids=("f_drink",),
        detections=shared_boxes,
    )
    m_holding_only = TemporalMoment(
        video_id="video_test3",
        t_start=11.0,
        t_end=14.0,
        rank=2,
        score=0.71,
        evidence_frame_ids=("f_hold",),
        detections=wrong_action_boxes,
    )

    cfg = ObjectRerankConfig(enabled=True, semantic_weight=0.7, object_weight=0.3)
    reranked, prov, spec = rerank_temporal_moments(
        [m_drinking, m_holding_only],
        "Người uống từ chai thủy tinh",
        config=cfg,
    )

    assert "action:uống" in spec.unsupported_cues
    assert prov[0].object_score == pytest.approx(prov[1].object_score)
    assert prov[0].action_verified is False
    assert prov[1].action_verified is False
    # Semantic score preserves the true action ordering when object evidence is identical
    assert reranked[0].t_start == 6.0


def test_p06_short_action_sparse_sampling_honesty():
    # Simulate a 1-second short action (like vi-dev-013 [5.25, 6.25] or vi-dev-015 [8.75, 9.75]):
    # Case 1: One sampled frame at 5.5s with high score -> produces valid bounded interval >= min_duration_s
    frames_one_hit = [
        FrameScore("f3", "video_test3", 3.0, 0.20),
        FrameScore("f5", "video_test3", 5.5, 0.91),
        FrameScore("f10", "video_test3", 10.0, 0.18),
    ]
    cfg = TemporalConfig(
        smoothing_window=1,
        score_quantile=0.7,
        min_score_floor=0.5,
        min_duration_s=1.0,
        max_duration_s=4.0,
    )
    moments = propose_temporal_moments(
        frames_one_hit, video_duration=18.72, config=cfg
    )
    assert len(moments) == 1
    assert moments[0].evidence_frame_ids == ("f5",)
    assert 0.0 <= moments[0].t_start < moments[0].t_end <= 18.72
    assert moments[0].detections is None

    # Case 2: Sampling miss (frames at 7.0s and 10.0s both score below floor for [8.75, 9.75] query)
    frames_missed = [
        FrameScore("f7", "video_test3", 7.0, 0.22),
        FrameScore("f10", "video_test3", 10.0, 0.19),
    ]
    moments_missed = propose_temporal_moments(
        frames_missed, video_duration=18.72, config=cfg
    )
    assert moments_missed == []
