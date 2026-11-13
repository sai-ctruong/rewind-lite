"""Unit tests for P07 temporal benchmark and error taxonomy (evaluation.temporal_benchmark & evaluation.error_analysis).

Covers:
- Taxonomy classification across all 7 categories:
  * hit (best_tiou >= threshold)
  * boundary_error (0 < best_tiou < threshold)
  * wrong_object (disjoint prediction missing target COCO classes)
  * wrong_relation (multi-person ambiguity or non-co-occurring objects)
  * wrong_action_or_stage (disjoint prediction despite matching objects)
  * sampling_or_coverage_miss (no frames indexed in GT window)
  * no_prediction (empty predictions with index coverage)
- compare_paired_ablation behavior:
  * paired comparison and delta computation
  * empirical conclusions (improved, neutral, degraded)
- run_temporal_ablation_on_entry:
  * synthetic timeline execution for E0 (context), E1 (temporal-only), E2 (YOLO rerank)
  * verified provenance and latency records
- CLI preflight check: exits with code 2 (BLOCKED) when sources are missing.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from evaluation.error_analysis import (
    TEMPORAL_ERROR_TAXONOMY,
    classify_temporal_query_error,
    compare_paired_ablation,
)
from evaluation.temporal_benchmark import main as benchmark_main, run_temporal_ablation_on_entry
from ingestion.build_index import KeyframeIndex
from ingestion.schemas import KeyframeRecord, RawKeyframe
from retrieval.detector import DetectorConfig, ObjectDetection
from retrieval.lite_core import EncoderSpec
from retrieval.object_reranker import ObjectRerankConfig, RerankProvenance
from retrieval.temporal import TemporalConfig
from retrieval.video_engine import VideoIndexEntry


def test_taxonomy_categories():
    query_row = {
        "query_id": "q1",
        "video_id": "v1",
        "split": "dev",
        "query_vi": "người lấy chai nước",
        "time_windows": [[5.0, 10.0]],
    }

    # 1. Hit
    eval_hit = {"predictions": [{"t_start": 5.0, "t_end": 10.0, "best_tiou": 1.0}]}
    c_hit = classify_temporal_query_error(query_row, eval_hit, tiou_threshold=0.3)
    assert c_hit["failure_category"] == "hit"

    # 2. Boundary error (partial overlap)
    eval_boundary = {"predictions": [{"t_start": 8.0, "t_end": 14.0, "best_tiou": 0.22}]}
    c_boundary = classify_temporal_query_error(query_row, eval_boundary, tiou_threshold=0.3)
    assert c_boundary["failure_category"] == "boundary_error"

    # 3. Wrong object (disjoint, missing bottle)
    eval_wrong_obj = {"predictions": [{"t_start": 15.0, "t_end": 20.0, "best_tiou": 0.0}]}
    prov_wrong_obj = RerankProvenance(
        video_id="v1", t_start=15.0, t_end=20.0, original_rank=1, final_rank=1,
        semantic_score_raw=0.7, semantic_score_norm=0.7, object_score=0.3,
        presence_score=0.5, cooccurrence_score=0.0, spatial_score=0.0, combined_score=0.6,
        matched_classes=("person",), missing_classes=("bottle",), cooccurring_frame_ids=(),
        multi_person_ambiguous=False, unsupported_query_cues=("action:lấy",),
    )
    c_wrong_obj = classify_temporal_query_error(query_row, eval_wrong_obj, top1_provenance=prov_wrong_obj)
    assert c_wrong_obj["failure_category"] == "wrong_object"

    # 4. Wrong relation (multi-person ambiguous)
    prov_multi_person = RerankProvenance(
        video_id="v1", t_start=15.0, t_end=20.0, original_rank=1, final_rank=1,
        semantic_score_raw=0.7, semantic_score_norm=0.7, object_score=0.7,
        presence_score=0.9, cooccurrence_score=0.5, spatial_score=0.5, combined_score=0.7,
        matched_classes=("person", "bottle"), missing_classes=(), cooccurring_frame_ids=(),
        multi_person_ambiguous=True, unsupported_query_cues=("action:lấy",),
    )
    c_multi = classify_temporal_query_error(query_row, eval_wrong_obj, top1_provenance=prov_multi_person)
    assert c_multi["failure_category"] == "wrong_relation"

    # 5. Wrong action or stage (objects match, but wrong temporal action)
    prov_wrong_action = RerankProvenance(
        video_id="v1", t_start=15.0, t_end=20.0, original_rank=1, final_rank=1,
        semantic_score_raw=0.7, semantic_score_norm=0.7, object_score=0.9,
        presence_score=0.9, cooccurrence_score=0.9, spatial_score=0.9, combined_score=0.75,
        matched_classes=("person", "bottle"), missing_classes=(), cooccurring_frame_ids=("f15",),
        multi_person_ambiguous=False, unsupported_query_cues=("action:lấy",),
    )
    c_act = classify_temporal_query_error(query_row, eval_wrong_obj, top1_provenance=prov_wrong_action)
    assert c_act["failure_category"] == "wrong_action_or_stage"

    # 6. Sampling or coverage miss
    eval_empty = {"predictions": []}
    c_miss = classify_temporal_query_error(query_row, eval_empty, covered_in_index=False)
    assert c_miss["failure_category"] == "sampling_or_coverage_miss"

    # 7. No prediction (index covered, but no proposal)
    c_nopred = classify_temporal_query_error(query_row, eval_empty, covered_in_index=True)
    assert c_nopred["failure_category"] == "no_prediction"


def test_compare_paired_ablation():
    queries = [
        {"query_id": "q1", "video_id": "v1", "query_vi": "tìm chai nước", "time_windows": [[1.0, 5.0]]},
        {"query_id": "q2", "video_id": "v1", "query_vi": "tìm cái cốc", "time_windows": [[10.0, 15.0]]},
    ]

    # q1 improved in E2 (0.2 -> 0.8), q2 unchanged (0.5 -> 0.5)
    e1_eval = {
        "summary": {"all": {"miou": 0.35, "r@1_tiou_0.3": 0.5, "r@1_tiou_0.5": 0.5}},
        "per_query": [
            {"query_id": "q1", "predictions": [{"t_start": 4.0, "t_end": 7.0, "best_tiou": 0.2}]},
            {"query_id": "q2", "predictions": [{"t_start": 10.0, "t_end": 15.0, "best_tiou": 1.0}]},
        ],
    }
    e2_eval = {
        "summary": {"all": {"miou": 0.90, "r@1_tiou_0.3": 1.0, "r@1_tiou_0.5": 1.0}},
        "per_query": [
            {"query_id": "q1", "predictions": [{"t_start": 1.0, "t_end": 5.0, "best_tiou": 1.0}]},
            {"query_id": "q2", "predictions": [{"t_start": 10.0, "t_end": 15.0, "best_tiou": 1.0}]},
        ],
    }

    paired = compare_paired_ablation(queries, e1_eval, e2_eval)
    assert paired["summary"]["improved_tiou_count"] == 1
    assert paired["summary"]["unchanged_tiou_count"] == 1
    assert paired["summary"]["improved_hit_count"] == 1
    assert paired["summary"]["empirical_conclusion"] == "object_rerank_improved"
    assert paired["summary"]["miou_delta"] > 0.0


class DummyEncoder:
    def __init__(self, vector: np.ndarray):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.vector


class FakeDetectorBackend:
    def detect_batch(self, frames, config):
        # Always detect bottle in frame f3
        dets = []
        for fid, vid, ts, img in frames:
            if fid == "f3":
                dets.append([
                    ObjectDetection(
                        frame_id="f3", video_id=vid, timestamp=ts, class_id=39,
                        class_name="bottle", confidence=0.92,
                        bbox_xyxy=(10, 10, 40, 40), bbox_norm_xyxy=(0.1, 0.1, 0.4, 0.4),
                        image_size=(100, 100), model_id="fake_yolo"
                    )
                ])
            else:
                dets.append([])
        return dets

    def unload(self):
        pass


def test_run_temporal_ablation_on_entry():
    records = [
        KeyframeRecord("f1", "v_synth", 1.0, np.array([1.0, 0.0], dtype=np.float32)),
        KeyframeRecord("f2", "v_synth", 2.0, np.array([0.9, 0.1], dtype=np.float32)),
        KeyframeRecord("f3", "v_synth", 5.0, np.array([0.7, 0.3], dtype=np.float32)),
    ]
    index = KeyframeIndex.build(records)
    raws = {r.id: RawKeyframe(id=r.id, video_id=r.video_id, timestamp=r.timestamp, image_path=None) for r in records}
    entry = VideoIndexEntry("v_synth", index, raws, len(records), len(records))

    encoders = {"clip": DummyEncoder(np.array([1.0, 0.0], dtype=np.float32))}
    specs = {"clip": EncoderSpec("dummy", "v1")}

    queries = [{
        "query_id": "q_synth",
        "video_id": "v_synth",
        "query_vi": "chai nước",
        "time_windows": [[4.0, 6.0]],
        "duration_s": 10.0,
        "split": "dev",
    }]
    frame_images = {r.id: np.zeros((100, 100, 3), dtype=np.uint8) for r in records}

    report = run_temporal_ablation_on_entry(
        entry,
        queries,
        encoders,
        specs,
        temporal_config=TemporalConfig(min_duration_s=1.0, max_duration_s=5.0),
        detector_config=DetectorConfig(enabled=True, model_id="fake_yolo"),
        rerank_config=ObjectRerankConfig(enabled=True),
        frame_images=frame_images,
        detector_backend=FakeDetectorBackend(),
    )

    assert report["status"] == "MEASURED"
    assert report["p06_action_verifier_gate"] == "NO-GO (Option A)"
    assert "E0_keyframe_context" in report
    assert "E1_temporal_only" in report
    assert "E2_temporal_object_rerank" in report
    assert "paired_ablation" in report
    assert report["latency_seconds"]["e2_query_total_s"] >= 0.0


def test_cli_blocked_when_missing_inputs(tmp_path):
    # Call benchmark_main with nonexistent paths -> should exit code 2 (BLOCKED)
    ret = benchmark_main(["--manifest", str(tmp_path / "m.json"), "--labels", str(tmp_path / "l.json"), "--root", str(tmp_path)])
    assert ret == 2
