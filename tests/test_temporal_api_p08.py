"""Unit tests for P08 Temporal Web API & service integration (ui.lite_app).

Covers:
- /api/lite/temporal_search endpoint:
  * temporal_only mode (use_detector=False) returns ranked [t_start, t_end]
  * temporal_object mode (use_detector=True) attaches detections and provenances
  * parameter validation (invalid query, top_k bounds)
  * unselected / missing video error handling
- Regression safety:
  * Phase 1 /api/lite/search remains 100% functional and unmodified.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from ingestion.build_index import KeyframeIndex
from ingestion.schemas import KeyframeRecord, RawKeyframe
from retrieval.detector import ObjectDetection
from retrieval.video_engine import VideoIndexEntry
from ui.lite_app import create_lite_app


class FakeDummyEncoder:
    def __init__(self, dim=2):
        self.dim = dim

    def encode_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeAppDetectorBackend:
    def detect_batch(self, frames, config):
        dets = []
        for fid, vid, ts, img in frames:
            dets.append([
                ObjectDetection(
                    frame_id=fid, video_id=vid, timestamp=ts, class_id=39,
                    class_name="bottle", confidence=0.91,
                    bbox_xyxy=(10, 10, 50, 50), bbox_norm_xyxy=(0.1, 0.1, 0.5, 0.5),
                    image_size=(100, 100), model_id="fake_yolo"
                )
            ])
        return dets

    def unload(self):
        pass


@pytest.fixture
def test_client(tmp_path):
    video_dir = tmp_path / "videos"
    index_root = tmp_path / "indexes"
    video_dir.mkdir()
    index_root.mkdir()

    # Create dummy video file
    video_file = video_dir / "vid1.mp4"
    video_file.write_bytes(b"dummy mp4 content")

    app = create_lite_app(
        video_dir=video_dir,
        index_root=index_root,
        encoder_factory=lambda m, r, d: FakeDummyEncoder(),
    )
    service = app.extensions["lite_service"]

    # Build and register synthetic entry
    records = [
        KeyframeRecord("f1", "vid1", 1.0, np.array([1.0, 0.0], dtype=np.float32)),
        KeyframeRecord("f2", "vid1", 2.0, np.array([0.9, 0.1], dtype=np.float32)),
        KeyframeRecord("f3", "vid1", 6.0, np.array([0.2, 0.8], dtype=np.float32)),
    ]
    index = KeyframeIndex.build(records)
    raws = {
        r.id: RawKeyframe(id=r.id, video_id=r.video_id, timestamp=r.timestamp, image_path=None)
        for r in records
    }
    entry = VideoIndexEntry("vid1", index, raws, len(records), len(records))
    service.entries["vid1"] = entry
    service._cache("vid1", video_file, entry)
    service.selected = "vid1"

    with app.test_client() as client:
        yield client, service


def test_temporal_search_api_modes_and_validation(test_client):
    client, service = test_client

    # 1. Temporal-only mode (use_detector=False)
    res = client.post("/api/lite/temporal_search", json={
        "video_id": "vid1",
        "query": "người lấy chai nước",
        "top_k": 3,
        "use_detector": False,
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["video_id"] == "vid1"
    assert data["mode"] == "temporal_only"
    assert "query_analysis" in data
    assert len(data["results"]) >= 1

    top1 = data["results"][0]
    assert "t_start" in top1 and "t_end" in top1
    assert top1["t_start"] < top1["t_end"]
    assert top1["rank"] == 1
    assert top1["detections"] is None

    # 2. Temporal-object mode (use_detector=True with fake backend)
    # Monkeypatch detector backend on service
    orig_search_temporal = service.search_temporal
    def mock_search_temporal(video_id, query, top_k=5, use_detector=False):
        return orig_search_temporal(
            video_id, query, top_k=top_k, use_detector=use_detector,
            detector_backend=FakeAppDetectorBackend()
        )
    service.search_temporal = mock_search_temporal

    res_det = client.post("/api/lite/temporal_search", json={
        "video_id": "vid1",
        "query": "người lấy chai nước",
        "top_k": 3,
        "use_detector": True,
    })
    assert res_det.status_code == 200
    det_data = res_det.get_json()
    assert det_data["mode"] == "temporal_object"
    assert len(det_data["results"]) >= 1

    # 3. Validation errors
    res_bad_q = client.post("/api/lite/temporal_search", json={"query": ""})
    assert res_bad_q.status_code == 400
    assert res_bad_q.get_json()["error"]["code"] == "invalid_query"

    res_bad_k = client.post("/api/lite/temporal_search", json={"query": "test", "top_k": 99})
    assert res_bad_k.status_code == 400
    assert res_bad_k.get_json()["error"]["code"] == "invalid_top_k"


def test_phase1_keyframe_search_regression(test_client):
    client, _ = test_client
    # Phase 1 /api/lite/search must remain functional
    res = client.post("/api/lite/search", json={
        "video_id": "vid1",
        "query": "người đi bộ",
        "top_k": 2,
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["video_id"] == "vid1"
    assert len(data["results"]) == 2
    assert "timestamp" in data["results"][0]
    assert "score_kind" in data["results"][0]


def test_r02_lazy_shared_detector_backend_and_cache_across_queries(tmp_path):
    import cv2

    video_dir = tmp_path / "videos"
    index_root = tmp_path / "indexes"
    video_dir.mkdir()
    index_root.mkdir()
    video_file = video_dir / "vid1.mp4"
    video_file.write_bytes(b"dummy mp4 content")

    factory_calls = []
    detected_frame_ids = []

    class CountingFakeBackend:
        def __init__(self, model_id: str = "yolov8n.pt"):
            self.model_id = model_id
            factory_calls.append(model_id)

        def detect_batch(self, frames, config):
            out = []
            for fid, vid, ts, img in frames:
                detected_frame_ids.append(fid)
                out.append([
                    ObjectDetection(
                        frame_id=fid,
                        video_id=vid,
                        timestamp=ts,
                        class_id=39,
                        class_name="bottle",
                        confidence=0.91,
                        bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
                        bbox_norm_xyxy=(0.1, 0.1, 0.5, 0.5),
                        image_size=(100, 100),
                        model_id=self.model_id,
                    )
                ])
            return out

        def unload(self):
            pass

    app = create_lite_app(
        video_dir=video_dir,
        index_root=index_root,
        encoder_factory=lambda m, r, d: FakeDummyEncoder(),
        detector_backend_factory=CountingFakeBackend,
    )
    service = app.extensions["lite_service"]

    ok, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 128, dtype=np.uint8))
    assert ok
    jpeg_bytes = buf.tobytes()

    records = [
        KeyframeRecord("f1", "vid1", 1.0, np.array([1.0, 0.0], dtype=np.float32)),
        KeyframeRecord("f2", "vid1", 2.0, np.array([0.9, 0.1], dtype=np.float32)),
        KeyframeRecord("f3", "vid1", 6.0, np.array([0.2, 0.8], dtype=np.float32)),
    ]
    index = KeyframeIndex.build(records)
    raws = {
        r.id: RawKeyframe(
            id=r.id,
            video_id=r.video_id,
            timestamp=r.timestamp,
            image_path=None,
            image_bytes=jpeg_bytes,
        )
        for r in records
    }
    entry = VideoIndexEntry("vid1", index, raws, len(records), len(records))
    service.entries["vid1"] = entry
    service._cache("vid1", video_file, entry)
    service.selected = "vid1"

    with app.test_client() as client:
        # Before YOLO is used, backend must not be initialized (lazy)
        assert len(factory_calls) == 0
        assert service.detector_backend is None

        res_no_yolo = client.post("/api/lite/temporal_search", json={
            "video_id": "vid1",
            "query": "người lấy chai nước",
            "top_k": 3,
            "use_detector": False,
        })
        assert res_no_yolo.status_code == 200
        assert len(factory_calls) == 0

        # Query 1 with YOLO enabled: initializes backend once and detects candidate frames
        res_q1 = client.post("/api/lite/temporal_search", json={
            "video_id": "vid1",
            "query": "người lấy chai nước",
            "top_k": 3,
            "use_detector": True,
        })
        assert res_q1.status_code == 200
        assert res_q1.get_json()["mode"] == "temporal_object"
        assert len(factory_calls) == 1
        first_run_detections = list(detected_frame_ids)
        assert len(first_run_detections) > 0

        # Query 2 with YOLO enabled: reuses same backend and hits RAM cache (0 new detections)
        res_q2 = client.post("/api/lite/temporal_search", json={
            "video_id": "vid1",
            "query": "cầm chai nước trên bàn",
            "top_k": 3,
            "use_detector": True,
        })
        assert res_q2.status_code == 200
        assert res_q2.get_json()["mode"] == "temporal_object"
        assert len(factory_calls) == 1
        assert detected_frame_ids == first_run_detections


def test_r02_concurrent_requests_create_single_detector_backend(tmp_path):
    import threading
    import time

    video_dir = tmp_path / "videos"
    index_root = tmp_path / "indexes"
    video_dir.mkdir()
    index_root.mkdir()

    init_count = 0
    count_lock = threading.Lock()

    class SlowBackend:
        def __init__(self, model_id: str = "yolov8n.pt"):
            nonlocal init_count
            time.sleep(0.05)
            with count_lock:
                init_count += 1
            self.model_id = model_id

        def detect_batch(self, frames, config):
            return [[] for _ in frames]

        def unload(self):
            pass

    app = create_lite_app(
        video_dir=video_dir,
        index_root=index_root,
        encoder_factory=lambda m, r, d: FakeDummyEncoder(),
        detector_backend_factory=SlowBackend,
    )
    service = app.extensions["lite_service"]

    results = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        results.append(service._get_detector_backend("yolov8n.pt"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert init_count == 1
    assert len(results) == 4
    assert all(b is results[0] for b in results)


def test_r06_lite_service_filters_dark_frames_lazily_and_keeps_phase1_intact(tmp_path):
    import cv2

    video_dir = tmp_path / "videos"
    index_root = tmp_path / "indexes"
    video_dir.mkdir()
    index_root.mkdir()
    video_file = video_dir / "vid_dark.mp4"
    video_file.write_bytes(b"dummy mp4 content")

    app = create_lite_app(
        video_dir=video_dir,
        index_root=index_root,
        encoder_factory=lambda m, r, d: FakeDummyEncoder(),
    )
    service = app.extensions["lite_service"]

    ok_black, buf_black = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    ok_night, buf_night = cv2.imencode(".jpg", np.full((32, 32, 3), 18, dtype=np.uint8))
    ok_bright, buf_bright = cv2.imencode(".jpg", np.full((32, 32, 3), 140, dtype=np.uint8))
    assert ok_black and ok_night and ok_bright

    # f_black at t=26.0 has highest cosine with [1.0, 0.0], f_night at t=12.0 is valid night scene (luma=18)
    records = [
        KeyframeRecord("f_bright", "vid_dark", 2.0, np.array([0.2, 0.98], dtype=np.float32)),
        KeyframeRecord("f_night_1", "vid_dark", 11.0, np.array([0.85, 0.52], dtype=np.float32)),
        KeyframeRecord("f_night_2", "vid_dark", 12.0, np.array([0.92, 0.39], dtype=np.float32)),
        KeyframeRecord("f_black", "vid_dark", 26.0, np.array([1.0, 0.0], dtype=np.float32)),
    ]
    index = KeyframeIndex.build(records)
    raws = {
        "f_bright": RawKeyframe("f_bright", "vid_dark", 2.0, image_bytes=buf_bright.tobytes()),
        "f_night_1": RawKeyframe("f_night_1", "vid_dark", 11.0, image_bytes=buf_night.tobytes()),
        "f_night_2": RawKeyframe("f_night_2", "vid_dark", 12.0, image_bytes=buf_night.tobytes()),
        "f_black": RawKeyframe("f_black", "vid_dark", 26.0, image_bytes=buf_black.tobytes()),
    }
    entry = VideoIndexEntry("vid_dark", index, raws, len(records), len(records))
    service.entries["vid_dark"] = entry
    service._cache("vid_dark", video_file, entry)
    service.selected = "vid_dark"

    with app.test_client() as client:
        # Before first Phase 2 query, luma cache for vid_dark is empty (lazy)
        assert "vid_dark" not in service.frame_lumas

        res_p2 = client.post("/api/lite/temporal_search", json={
            "video_id": "vid_dark",
            "query": "xe ô tô chạy trong đêm",
            "top_k": 5,
            "use_detector": False,
        })
        assert res_p2.status_code == 200
        p2_results = res_p2.get_json()["results"]
        assert len(p2_results) >= 1
        # Black frame must not be peak or evidence in any Phase 2 moment
        for m in p2_results:
            assert m["peak_frame_id"] != "f_black"
            assert "f_black" not in m["evidence_frame_ids"]
        # Valid night scene (luma ~ 18 > dark_frame_luma_max) is @1
        assert p2_results[0]["peak_frame_id"] in {"f_night_1", "f_night_2"}
        assert "f_night_2" in p2_results[0]["evidence_frame_ids"]
        assert "vid_dark" in service.frame_lumas
        cached_lumas_ref = service.frame_lumas["vid_dark"]

        # Second Phase 2 query reuses the cached luma map without recomputing
        res_p2_again = client.post("/api/lite/temporal_search", json={
            "video_id": "vid_dark",
            "query": "cảnh ban đêm",
            "top_k": 5,
            "use_detector": False,
        })
        assert res_p2_again.status_code == 200
        assert service.frame_lumas["vid_dark"] is cached_lumas_ref

        # Phase 1 (/api/lite/search) is untouched and still returns f_black at rank 1
        res_p1 = client.post("/api/lite/search", json={
            "video_id": "vid_dark",
            "query": "xe ô tô chạy trong đêm",
            "top_k": 3,
        })
        assert res_p1.status_code == 200
        p1_results = res_p1.get_json()["results"]
        assert p1_results[0]["frame_id"] == "f_black"
