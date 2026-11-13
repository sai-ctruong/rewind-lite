"""Unit tests for P05 object-aware reranking (retrieval.object_reranker).

Covers:
- Vietnamese query parser (parse_vietnamese_query_objects):
  * Multi-word matching (người đàn ông -> person, chai nước -> bottle)
  * Unsupported action/attribute cue detection (action:lấy, attribute:áo xanh)
  * Unsupported queries (no mapped COCO classes -> mapping_supported=False)
- Bounding box 2D IoU calculation and deduplication.
- Fallback guarantees:
  * config.enabled=False preserves original P03 rank and scores exactly
  * unsupported query or no detections falls back safely
- Reranking behavior:
  * Moment with matching object evidence ranks higher than moment without
  * Same-frame co-occurrence boost for subject + object queries
  * Multi-person discount (multi_person_ambiguity)
  * Action verification contract: action_verified is ALWAYS False
  * Output compatibility with evaluation.temporal_metrics.
"""
from __future__ import annotations

import pytest

from retrieval.object_reranker import (
    ObjectRerankConfig,
    RerankProvenance,
    bbox_iou_xyxy,
    parse_vietnamese_query_objects,
    rerank_temporal_moments,
)
from retrieval.temporal import TemporalMoment
from evaluation.temporal_metrics import evaluate_temporal_predictions


def test_r07_person_mapping_and_same_frame_count():
    for query in ("hai học sinh", "cặp đôi", "đám đông", "nhóm học sinh"):
        spec = parse_vietnamese_query_objects(query)
        assert spec.mapping_supported and "person" in spec.target_classes
        assert spec.person_count == 2
    assert "person" not in parse_vietnamese_query_objects("nhóm xe").target_classes
    query = "hai học sinh cầm chai nước"
    spec = parse_vietnamese_query_objects(query)
    assert set(spec.target_classes) == {"person", "bottle"}
    assert spec.subject_class == "person"
    def moment(fid, count, rank):
        dets = [{"frame_id": fid, "class_name": "person", "confidence": .9,
                 "bbox_norm_xyxy": (.05 + i*.4, .1, .35 + i*.4, .9)} for i in range(count)]
        dets.append({"frame_id": fid, "class_name": "bottle", "confidence": .9,
                     "bbox_norm_xyxy": (.2, .2, .3, .4)})
        return TemporalMoment("v", rank*10., rank*10.+2, rank, .8, (fid,), detections=tuple(dets))
    out, prov, _ = rerank_temporal_moments(
        [moment("one", 1, 1), moment("two", 2, 2)], query,
        config=ObjectRerankConfig(enabled=True))
    assert out[0].evidence_frame_ids == ("two",)
    assert prov[0].cooccurring_frame_ids == ("two",)
    assert not prov[0].action_verified


def test_vietnamese_query_parser():
    # 1. Subject + Object with unsupported action
    spec1 = parse_vietnamese_query_objects("người đàn ông lấy chai nước trên bàn")
    assert spec1.mapping_supported is True
    assert "person" in spec1.target_classes
    assert "bottle" in spec1.target_classes
    assert "dining table" in spec1.target_classes
    assert spec1.subject_class == "person"
    assert "action:lấy" in spec1.unsupported_cues

    # 2. Unsupported action and color attribute
    spec2 = parse_vietnamese_query_objects("người áo xanh đang uống nước")
    assert "person" in spec2.target_classes
    assert "attribute:áo xanh" in spec2.unsupported_cues
    assert "action:uống" in spec2.unsupported_cues

    # 3. Query with no COCO classes mapped
    spec3 = parse_vietnamese_query_objects("bầu trời hoàng hôn tuyệt đẹp")
    assert spec3.mapping_supported is False
    assert spec3.target_classes == ()


def test_bbox_iou_2d():
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (0.0, 0.0, 10.0, 10.0)
    assert bbox_iou_xyxy(b1, b2) == pytest.approx(1.0)

    b3 = (5.0, 0.0, 15.0, 10.0)  # half overlap: inter=50, union=150 -> 1/3
    assert bbox_iou_xyxy(b1, b3) == pytest.approx(50.0 / 150.0)

    b4 = (20.0, 20.0, 30.0, 30.0)  # disjoint
    assert bbox_iou_xyxy(b1, b4) == pytest.approx(0.0)


def test_reranker_fallback_when_disabled_or_unsupported():
    m1 = TemporalMoment(
        video_id="v1", t_start=1.0, t_end=5.0, rank=1, score=0.9,
        evidence_frame_ids=("f1",), score_type="temporal_cosine"
    )
    m2 = TemporalMoment(
        video_id="v1", t_start=10.0, t_end=15.0, rank=2, score=0.8,
        evidence_frame_ids=("f2",), score_type="temporal_cosine"
    )

    # Fallback 1: disabled config
    cfg_off = ObjectRerankConfig(enabled=False)
    out1, prov1, _ = rerank_temporal_moments([m1, m2], "người lấy chai nước", config=cfg_off)
    assert len(out1) == 2
    assert out1[0].rank == 1 and out1[0].score == pytest.approx(0.9)
    assert prov1[0].fallback_reason == "reranker_disabled"
    assert prov1[0].action_verified is False

    # Fallback 2: unsupported query
    cfg_on = ObjectRerankConfig(enabled=True)
    out2, prov2, _ = rerank_temporal_moments([m1, m2], "bầu trời hoàng hôn", config=cfg_on)
    assert out2[0].rank == 1
    assert prov2[0].fallback_reason == "no_mapped_classes"


def test_reranking_boosts_matching_object_and_respects_multiperson():
    # Candidate 1: slightly higher semantic score (0.85) but NO bottle detection (only person)
    det_c1 = [
        {
            "frame_id": "f1", "class_name": "person", "confidence": 0.9,
            "bbox_norm_xyxy": (0.1, 0.1, 0.5, 0.9), "image_size": (100, 100)
        }
    ]
    m1 = TemporalMoment(
        video_id="v1", t_start=1.0, t_end=4.0, rank=1, score=0.85,
        evidence_frame_ids=("f1",), score_type="temporal_cosine", detections=tuple(det_c1)
    )

    # Candidate 2: slightly lower semantic score (0.80) but HAS BOTH person AND bottle in same frame!
    det_c2 = [
        {
            "frame_id": "f10", "class_name": "person", "confidence": 0.95,
            "bbox_norm_xyxy": (0.1, 0.1, 0.4, 0.9), "image_size": (100, 100)
        },
        {
            "frame_id": "f10", "class_name": "bottle", "confidence": 0.90,
            "bbox_norm_xyxy": (0.35, 0.4, 0.45, 0.6), "image_size": (100, 100)
        },
    ]
    m2 = TemporalMoment(
        video_id="v1", t_start=10.0, t_end=14.0, rank=2, score=0.80,
        evidence_frame_ids=("f10",), score_type="temporal_cosine", detections=tuple(det_c2)
    )

    cfg = ObjectRerankConfig(
        enabled=True,
        semantic_weight=0.5,
        object_weight=0.5,
        presence_weight=0.5,
        cooccurrence_weight=0.3,
        spatial_proximity_weight=0.2,
    )

    reranked, prov, spec = rerank_temporal_moments([m1, m2], "người lấy chai nước", config=cfg)
    assert len(reranked) == 2

    # m2 has both person and bottle -> object score is much higher, promoting it to Rank 1!
    assert reranked[0].video_id == "v1"
    assert reranked[0].t_start == 10.0
    assert reranked[0].rank == 1
    assert reranked[0].score_type == "temporal_object_reranked"
    assert prov[0].original_rank == 2
    assert prov[0].final_rank == 1
    assert prov[0].action_verified is False  # Never claim action verified
    assert "bottle" in prov[0].matched_classes
    assert "person" in prov[0].matched_classes


def test_compatibility_with_evaluation_temporal_metrics():
    det = [
        {
            "frame_id": "f1", "class_name": "bottle", "confidence": 0.9,
            "bbox_norm_xyxy": (0.1, 0.1, 0.3, 0.5), "image_size": (100, 100)
        }
    ]
    m = TemporalMoment(
        video_id="video-test", t_start=2.0, t_end=6.0, rank=1, score=0.85,
        evidence_frame_ids=("f1",), score_type="temporal_cosine", detections=tuple(det)
    )
    cfg = ObjectRerankConfig(enabled=True)
    reranked, _, _ = rerank_temporal_moments([m], "chai nước", config=cfg)

    predictions = {
        "vi-dev-001": [moment.to_dict() for moment in reranked]
    }
    queries = [{
        "query_id": "vi-dev-001",
        "video_id": "video-test",
        "query_vi": "tìm chai nước",
        "time_windows": [[1.0, 5.0]],
        "duration_s": 20.0,
        "split": "dev",
    }]

    eval_result = evaluate_temporal_predictions(queries, predictions)
    assert eval_result["n_evaluable"] == 1
    assert eval_result["summary"]["all"]["r@1_tiou_0.5"] == 1.0


def test_r03_continuous_semantic_normalization_and_generic_person_scale():
    from retrieval.object_reranker import _normalize_semantic_scores

    def make_m(score: float, t0: float, rank: int = 1, dets=None):
        return TemporalMoment(
            video_id="v",
            t_start=t0,
            t_end=t0 + 2.0,
            rank=rank,
            score=score,
            evidence_frame_ids=(f"f{int(t0)}",),
            score_type="temporal_cosine",
            detections=dets,
        )

    # 1. Mixed positive and negative scores -> clipped into [0, 1]
    norm_mixed = _normalize_semantic_scores([make_m(0.12, 0.0, 1), make_m(-0.05, 5.0, 2)])
    assert all(0.0 <= v <= 1.0 for v in norm_mixed)
    assert norm_mixed == pytest.approx([1.0, 0.0])

    # 2. Continuity across 0.5 (no jump between hi=0.49 and hi=0.51)
    norm_49 = _normalize_semantic_scores([make_m(0.49, 0.0, 1), make_m(0.40, 5.0, 2)])
    norm_51 = _normalize_semantic_scores([make_m(0.51, 0.0, 1), make_m(0.40, 5.0, 2)])
    assert norm_49 == pytest.approx([1.0, 0.40 / 0.49])
    assert norm_51 == pytest.approx([1.0, 0.40 / 0.51])
    assert abs(norm_49[1] - norm_51[1]) < 0.05

    # 3. All negative scores -> min-max into [0, 1]
    norm_neg = _normalize_semantic_scores([make_m(-0.02, 0.0, 1), make_m(-0.10, 5.0, 2)])
    assert norm_neg == pytest.approx([1.0, 0.0])

    # 4. All equal scores -> all 1.0
    norm_eq = _normalize_semantic_scores([make_m(0.25, 0.0, 1), make_m(0.25, 5.0, 2)])
    assert norm_eq == pytest.approx([1.0, 1.0])
    norm_eq_neg = _normalize_semantic_scores([make_m(-0.1, 0.0, 1), make_m(-0.1, 5.0, 2)])
    assert norm_eq_neg == pytest.approx([1.0, 1.0])

    # 5. ObjectRerankConfig.generic_person_object_scale validation and effect
    default_cfg = ObjectRerankConfig()
    assert default_cfg.generic_person_object_scale == pytest.approx(0.5)
    with pytest.raises(ValueError, match="generic_person_object_scale"):
        ObjectRerankConfig(generic_person_object_scale=1.5)
    with pytest.raises(ValueError, match="generic_person_object_scale"):
        ObjectRerankConfig(generic_person_object_scale=-0.1)

    det_person = (
        {
            "frame_id": "f0",
            "class_name": "person",
            "confidence": 0.9,
            "bbox_norm_xyxy": (0.1, 0.1, 0.5, 0.9),
            "image_size": (100, 100),
        },
    )
    m_p = make_m(0.2, 0.0, 1, dets=det_person)
    _, prov_half, _ = rerank_temporal_moments(
        [m_p], "người đàn ông", config=ObjectRerankConfig(enabled=True, generic_person_object_scale=0.5)
    )
    _, prov_full, _ = rerank_temporal_moments(
        [m_p], "người đàn ông", config=ObjectRerankConfig(enabled=True, generic_person_object_scale=1.0)
    )
    assert prov_half[0].object_score == pytest.approx(0.5 * prov_full[0].object_score)


def test_r04_person_count_parsing_and_multi_person_reward():
    # 1. Parser extracts person_count accurately across phrases
    s_two = parse_vietnamese_query_objects("2 người đi giữa cánh đồng hoa")
    assert s_two.person_count == 2
    assert s_two.to_dict()["person_count"] == 2

    assert parse_vietnamese_query_objects("hai học sinh ngồi viết bài trên bàn").person_count == 2
    assert parse_vietnamese_query_objects("ba cô gái đang nói chuyện").person_count == 3
    assert parse_vietnamese_query_objects("3 chàng trai chạy bộ").person_count == 3
    assert parse_vietnamese_query_objects("cặp đôi đi dạo").person_count == 2
    assert parse_vietnamese_query_objects("nhóm bạn trong phòng").person_count == 2
    assert parse_vietnamese_query_objects("đám đông trong hội trường").person_count == 2
    assert parse_vietnamese_query_objects("một người đàn ông").person_count == 1
    assert parse_vietnamese_query_objects("người lấy chai nước").person_count is None

    # 2. Probe E5: "2 người đi giữa cánh đồng hoa", equal semantic scores -> moment with 2 persons wins
    def make_person_dets(fid: str, n: int, conf: float = 0.9):
        return tuple(
            {
                "frame_id": fid,
                "class_name": "person",
                "confidence": conf,
                "bbox_norm_xyxy": (0.1 + 0.4 * i, 0.1, 0.3 + 0.4 * i, 0.9),
                "image_size": (100, 100),
            }
            for i in range(n)
        )

    m_one = TemporalMoment(
        video_id="v", t_start=0.0, t_end=2.0, rank=1, score=0.18,
        evidence_frame_ids=("a",), score_type="temporal_cosine",
        detections=make_person_dets("a", 1),
    )
    m_two = TemporalMoment(
        video_id="v", t_start=5.0, t_end=7.0, rank=2, score=0.18,
        evidence_frame_ids=("b",), score_type="temporal_cosine",
        detections=make_person_dets("b", 2),
    )
    reranked_e5, prov_e5, spec_e5 = rerank_temporal_moments(
        [m_one, m_two],
        "2 người đi giữa cánh đồng hoa",
        config=ObjectRerankConfig(enabled=True),
    )
    assert spec_e5.person_count == 2
    assert reranked_e5[0].t_start == 5.0
    prov_by_t = {p.t_start: p for p in prov_e5}
    assert prov_by_t[5.0].object_score > prov_by_t[0.0].object_score
    assert prov_by_t[5.0].multi_person_ambiguous is False

    # 3. Single/unspecified person query ("người lấy chai nước") still penalizes multi-person frame
    dets_single_bottle = (
        {"frame_id": "f1", "class_name": "person", "confidence": 0.9, "bbox_norm_xyxy": (0.1, 0.1, 0.4, 0.9)},
        {"frame_id": "f1", "class_name": "bottle", "confidence": 0.9, "bbox_norm_xyxy": (0.35, 0.4, 0.45, 0.6)},
    )
    dets_multi_bottle = (
        {"frame_id": "f2", "class_name": "person", "confidence": 0.9, "bbox_norm_xyxy": (0.1, 0.1, 0.4, 0.9)},
        {"frame_id": "f2", "class_name": "person", "confidence": 0.9, "bbox_norm_xyxy": (0.6, 0.1, 0.9, 0.9)},
        {"frame_id": "f2", "class_name": "bottle", "confidence": 0.9, "bbox_norm_xyxy": (0.35, 0.4, 0.45, 0.6)},
    )
    m_multi_b = TemporalMoment("v", 0.0, 2.0, 1, 0.5, ("f2",), detections=dets_multi_bottle)
    m_single_b = TemporalMoment("v", 5.0, 7.0, 2, 0.5, ("f1",), detections=dets_single_bottle)
    reranked_b, prov_b, spec_b = rerank_temporal_moments(
        [m_multi_b, m_single_b],
        "người lấy chai nước",
        config=ObjectRerankConfig(enabled=True),
    )
    assert spec_b.person_count is None
    assert reranked_b[0].t_start == 5.0
    prov_b_by_t = {p.t_start: p for p in prov_b}
    assert prov_b_by_t[0.0].multi_person_ambiguous is True
    assert prov_b_by_t[5.0].object_score > prov_b_by_t[0.0].object_score


def test_r05_additive_object_boost_does_not_penalize_missing_detection():
    # Validate max_object_boost bounds [0, 0.5]
    default_cfg = ObjectRerankConfig()
    assert 0.0 < default_cfg.max_object_boost <= 0.5
    with pytest.raises(ValueError, match="max_object_boost"):
        ObjectRerankConfig(max_object_boost=-0.05)
    with pytest.raises(ValueError, match="max_object_boost"):
        ObjectRerankConfig(max_object_boost=0.6)

    # TC03-like scenario:
    # Moment A (night car, true match): high semantic (0.22 -> sem_norm=1.0), detector missed car (obj=0.0)
    # Moment B (false match): lower semantic (0.184 -> sem_norm=0.836), detector found false car (obj~0.68)
    m_true_night = TemporalMoment(
        video_id="v", t_start=122.0, t_end=128.0, rank=1, score=0.220,
        evidence_frame_ids=("f_night",), score_type="temporal_cosine",
        detections=(),
    )
    det_false_car = (
        {
            "frame_id": "f_day",
            "class_name": "car",
            "confidence": 0.90,
            "bbox_norm_xyxy": (0.1, 0.2, 0.6, 0.8),
            "image_size": (100, 100),
        },
    )
    m_false_day = TemporalMoment(
        video_id="v", t_start=22.0, t_end=38.0, rank=2, score=0.184,
        evidence_frame_ids=("f_day",), score_type="temporal_cosine",
        detections=det_false_car,
    )

    reranked, prov, _ = rerank_temporal_moments(
        [m_true_night, m_false_day],
        "xe ô tô chạy trong đêm bật đèn pha",
        config=ObjectRerankConfig(enabled=True, max_object_boost=0.15),
    )
    assert reranked[0].t_start == 122.0
    prov_by_t = {p.t_start: p for p in prov}
    assert prov_by_t[122.0].object_score == pytest.approx(0.0)
    assert prov_by_t[122.0].object_boost == pytest.approx(0.0)
    assert prov_by_t[122.0].combined_score == pytest.approx(1.0)
    assert prov_by_t[22.0].object_boost == pytest.approx(0.15 * prov_by_t[22.0].object_score)
    assert prov_by_t[22.0].combined_score == pytest.approx(
        prov_by_t[22.0].semantic_score_norm + prov_by_t[22.0].object_boost
    )
    assert prov_by_t[122.0].combined_score > prov_by_t[22.0].combined_score
