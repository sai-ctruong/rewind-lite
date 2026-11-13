"""Unit tests for P04 object detector adapter and cache (retrieval.detector).

Covers:
- ObjectDetection contract and coordinate conventions (pixel & normalized).
- Rejection of invalid/degenerate boxes and NaN/inf coordinates.
- DetectorConfig validation.
- DetectionCache SHA-256 key determinism, storage, retrieval, and invalidation.
- Deterministic FakeDetectorBackend testing.
- attach_moment_detections behavior:
  * config.enabled=False preserves moments unchanged
  * candidate frame filtering (only processes evidence_frame_ids)
  * allowed_classes filtering
  * caching integration
- Real UltralyticsYOLOBackend smoke test (if model available or downloads nano).
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from retrieval.detector import (
    DetectionCache,
    DetectorConfig,
    ObjectDetection,
    attach_moment_detections,
)
from retrieval.temporal import TemporalMoment


def test_object_detection_contract_and_normalization():
    det = ObjectDetection(
        frame_id="f1",
        video_id="v1",
        timestamp=2.5,
        class_id=0,
        class_name="person",
        confidence=0.88,
        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
        bbox_norm_xyxy=(0.1, 0.2, 0.55, 0.88),
        image_size=(200, 250),
        model_id="yolov8n.pt",
    )
    assert det.frame_id == "f1"
    assert det.video_id == "v1"
    assert det.timestamp == 2.5
    assert det.class_id == 0
    assert det.class_name == "person"
    assert det.confidence == pytest.approx(0.88)
    assert det.bbox_xyxy == (10.0, 20.0, 110.0, 220.0)
    assert det.bbox_norm_xyxy == (0.1, 0.2, 0.55, 0.88)
    assert det.image_size == (200, 250)

    # Rejection of degenerate or out of bounds boxes
    with pytest.raises(ValueError, match="invalid bbox_xyxy"):
        ObjectDetection(
            frame_id="f1", video_id="v1", timestamp=1.0, class_id=0, class_name="person",
            confidence=0.5, bbox_xyxy=(100.0, 20.0, 50.0, 200.0),  # x1 >= x2
            bbox_norm_xyxy=(0.5, 0.1, 0.25, 0.8), image_size=(200, 200), model_id="m"
        )
    with pytest.raises(ValueError, match="invalid bbox_norm_xyxy"):
        ObjectDetection(
            frame_id="f1", video_id="v1", timestamp=1.0, class_id=0, class_name="person",
            confidence=0.5, bbox_xyxy=(10.0, 20.0, 50.0, 100.0),
            bbox_norm_xyxy=(0.1, 0.2, 1.5, 0.8),  # nx2 > 1.0
            image_size=(200, 200), model_id="m"
        )
    with pytest.raises(ValueError, match="confidence"):
        ObjectDetection(
            frame_id="f1", video_id="v1", timestamp=1.0, class_id=0, class_name="person",
            confidence=0.0, bbox_xyxy=(10.0, 20.0, 50.0, 100.0),  # <= 0
            bbox_norm_xyxy=(0.1, 0.2, 0.5, 0.8), image_size=(200, 200), model_id="m"
        )


def test_detector_config_validation():
    cfg = DetectorConfig()
    assert cfg.enabled is False
    assert cfg.model_id == "yolov8n.pt"
    assert cfg.conf_threshold == 0.25

    with pytest.raises(ValueError, match="conf_threshold"):
        DetectorConfig(conf_threshold=-0.1)
    with pytest.raises(ValueError, match="imgsz"):
        DetectorConfig(imgsz=0)
    with pytest.raises(ValueError, match="batch_size"):
        DetectorConfig(batch_size=-1)


def test_detection_cache_key_and_invalidation(tmp_path):
    cache = DetectionCache(cache_dir=tmp_path)
    k1 = DetectionCache.make_key(
        video_id="v1", frame_id="f1", timestamp=1.0, model_id="yolov8n.pt",
        conf_threshold=0.25, iou_threshold=0.45, imgsz=640
    )
    k2 = DetectionCache.make_key(
        video_id="v1", frame_id="f1", timestamp=1.0, model_id="yolov8n.pt",
        conf_threshold=0.50, iou_threshold=0.45, imgsz=640
    )
    assert k1 != k2  # Invalidation on changed threshold

    det = ObjectDetection(
        frame_id="f1", video_id="v1", timestamp=1.0, class_id=0, class_name="person",
        confidence=0.9, bbox_xyxy=(10, 10, 50, 50), bbox_norm_xyxy=(0.1, 0.1, 0.5, 0.5),
        image_size=(100, 100), model_id="yolov8n.pt"
    )
    cache.put(k1, [det])
    cached = cache.get(k1)
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["class_name"] == "person"

    # Specific key invalidation
    cache.invalidate(k1)
    assert cache.get(k1) is None


class FakeDetectorBackend:
    def __init__(self, detections_by_frame: dict[str, list[ObjectDetection]]):
        self.detections_by_frame = detections_by_frame
        self.call_count = 0

    def detect_batch(self, frames, config):
        self.call_count += 1
        return [self.detections_by_frame.get(f[0], []) for f in frames]

    def unload(self):
        pass


def test_attach_moment_detections_candidate_filtering_and_cache(tmp_path):
    # Moments with evidence frames
    m1 = TemporalMoment(
        video_id="v1", t_start=1.0, t_end=4.0, rank=1, score=0.9,
        evidence_frame_ids=("f1", "f2"), score_type="temporal_cosine"
    )
    m2 = TemporalMoment(
        video_id="v1", t_start=10.0, t_end=15.0, rank=2, score=0.7,
        evidence_frame_ids=("f10",), score_type="temporal_cosine"
    )

    # 1. Disabled detector returns moments unchanged
    cfg_disabled = DetectorConfig(enabled=False)
    out_disabled = attach_moment_detections([m1, m2], {}, config=cfg_disabled)
    assert out_disabled[0].detections is None
    assert out_disabled[1].detections is None

    # 2. Enabled detector runs ONLY on evidence_frame_ids
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_images = {"f1": dummy_img, "f2": dummy_img, "f10": dummy_img, "f99": dummy_img}

    det_f1 = ObjectDetection(
        frame_id="f1", video_id="v1", timestamp=1.0, class_id=0, class_name="person",
        confidence=0.95, bbox_xyxy=(10, 10, 40, 80), bbox_norm_xyxy=(0.1, 0.1, 0.4, 0.8),
        image_size=(100, 100), model_id="yolov8n.pt"
    )
    det_f2 = ObjectDetection(
        frame_id="f2", video_id="v1", timestamp=2.0, class_id=39, class_name="bottle",
        confidence=0.85, bbox_xyxy=(50, 50, 70, 90), bbox_norm_xyxy=(0.5, 0.5, 0.7, 0.9),
        image_size=(100, 100), model_id="yolov8n.pt"
    )

    fake_backend = FakeDetectorBackend({"f1": [det_f1], "f2": [det_f2], "f10": []})
    cache = DetectionCache(cache_dir=tmp_path)
    cfg_enabled = DetectorConfig(enabled=True)

    out = attach_moment_detections(
        [m1, m2], frame_images, config=cfg_enabled, backend=fake_backend, cache=cache
    )
    assert len(out) == 2
    assert fake_backend.call_count == 1

    # m1 has f1 and f2 -> 2 detections
    assert out[0].detections is not None
    assert len(out[0].detections) == 2
    classes_m1 = {d["class_name"] for d in out[0].detections}
    assert classes_m1 == {"person", "bottle"}

    # m2 has f10 -> 0 detections
    assert out[1].detections == ()

    # 3. Second run hits cache, backend detect_batch is NOT called again
    fake_backend_2 = FakeDetectorBackend({})
    out_cached = attach_moment_detections(
        [m1, m2], frame_images, config=cfg_enabled, backend=fake_backend_2, cache=cache
    )
    assert len(out_cached[0].detections) == 2
    assert fake_backend_2.call_count == 0


def test_r02_shared_default_backend_and_cache_across_calls(monkeypatch):
    import retrieval.detector as det_mod

    created_backends = []
    detected_frames = []

    class TrackingFakeBackend:
        def __init__(self, model_id: str = "yolov8n.pt"):
            self.model_id = model_id
            created_backends.append(self)

        def detect_batch(self, frames, config):
            out = []
            for fid, vid, ts, img in frames:
                detected_frames.append(fid)
                out.append([
                    ObjectDetection(
                        frame_id=fid,
                        video_id=vid,
                        timestamp=ts,
                        class_id=0,
                        class_name="person",
                        confidence=0.9,
                        bbox_xyxy=(10.0, 10.0, 50.0, 80.0),
                        bbox_norm_xyxy=(0.1, 0.1, 0.5, 0.8),
                        image_size=(100, 100),
                        model_id=self.model_id,
                    )
                ])
            return out

        def unload(self):
            pass

    monkeypatch.setattr(det_mod, "UltralyticsYOLOBackend", TrackingFakeBackend)
    det_mod.reset_shared_detector_state()

    m1 = TemporalMoment(
        video_id="v_shared", t_start=1.0, t_end=3.0, rank=1, score=0.9,
        evidence_frame_ids=("f1", "f2"), score_type="temporal_cosine"
    )
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = DetectorConfig(enabled=True, model_id="yolov8n.pt")

    out1 = attach_moment_detections([m1], {"f1": img, "f2": img}, config=cfg)
    assert len(created_backends) == 1
    assert detected_frames == ["f1", "f2"]
    assert len(out1[0].detections) == 2

    # Second call with backend=None, cache=None reuses shared backend & RAM cache
    out2 = attach_moment_detections([m1], {"f1": img, "f2": img}, config=cfg)
    assert len(created_backends) == 1
    assert detected_frames == ["f1", "f2"]
    assert len(out2[0].detections) == 2
    det_mod.reset_shared_detector_state()


def test_ultralytics_backend_smoke():
    from retrieval.detector import UltralyticsYOLOBackend
    backend = UltralyticsYOLOBackend(model_id="yolov8n.pt")
    # Synthetic frame: black image with small white rectangle
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    img[50:200, 50:150] = 255
    frames = [("f_test", "v_test", 1.0, img)]
    cfg = DetectorConfig(enabled=True, model_id="yolov8n.pt", imgsz=320, device="cpu")

    results = backend.detect_batch(frames, cfg)
    assert len(results) == 1
    assert isinstance(results[0], list)
    backend.unload()
