"""Corpus search uses one query embedding and reports incomplete coverage."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ui.lite_app import create_lite_app


class CorpusEncoder:
    def __init__(self):
        self.queries = []

    def embed(self, raw):
        return np.array([1, 0] if raw.video_id == "alpha" else [0, 1], dtype=np.float32)

    def encode_text(self, query):
        self.queries.append(query)
        return np.array([1, 0], dtype=np.float32)


def _video(path, num_frames=4, vary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (32, 32))
    assert writer.isOpened()
    for i in range(num_frames):
        val = 50 + i * 30 if vary else 120
        writer.write(np.full((32, 32, 3), val, dtype=np.uint8))
    writer.release()


def test_global_search_ranks_across_saved_indexes_and_embeds_once(tmp_path):
    videos, indexes = tmp_path / "videos", tmp_path / "indexes"
    for name in ("alpha", "beta", "missing"):
        _video(videos / f"{name}.mp4")
    encoder = CorpusEncoder()
    app = create_lite_app(video_dir=videos, index_root=indexes,
                          encoder_factory=lambda *_: encoder)
    client = app.test_client()
    for name in ("beta", "alpha"):
        assert client.post("/api/lite/index", json={"video_id": name}).status_code == 200

    listing = client.get("/api/lite/videos").get_json()
    assert listing["videos"] == ["alpha", "beta", "missing"]
    assert listing["saved_index_videos"] == ["alpha", "beta"]

    restarted = create_lite_app(video_dir=videos, index_root=indexes,
                                encoder_factory=lambda *_: encoder).test_client()
    response = restarted.post("/api/lite/search_all", json={"query": "người đi bộ", "top_k": 2})
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_videos"] == 3
    assert data["searched_videos"] == 2
    assert data["skipped"] == [{"video_id": "missing", "reason": "index_missing"}]
    assert encoder.queries == ["người đi bộ"]
    assert [hit["video_id"] for hit in data["results"]] == ["alpha", "beta"]
    assert [hit["rank"] for hit in data["results"]] == [1, 2]
    for hit in data["results"]:
        assert restarted.get(hit["image_url"]).status_code == 200


def test_global_search_skips_expired_index_and_rejects_empty_coverage(tmp_path):
    videos, indexes = tmp_path / "videos", tmp_path / "indexes"
    _video(videos / "alpha.mp4")
    _video(videos / "beta.mp4")
    app = create_lite_app(video_dir=videos, index_root=indexes,
                          encoder_factory=lambda *_: CorpusEncoder())
    client = app.test_client()
    assert client.post("/api/lite/search_all", json={"query": "x"}).status_code == 409
    assert client.post("/api/lite/search_all", json={"query": " "}).status_code == 400
    assert client.post("/api/lite/search_all", json={"query": "x", "top_k": 0}).status_code == 400
    assert client.post("/api/lite/index", json={"video_id": "alpha"}).status_code == 200
    assert client.post("/api/lite/index", json={"video_id": "beta"}).status_code == 200
    (videos / "alpha.mp4").write_bytes(b"changed")
    restarted = create_lite_app(video_dir=videos, index_root=indexes,
                                encoder_factory=lambda *_: CorpusEncoder()).test_client()
    response = restarted.post("/api/lite/search_all", json={"query": "x"})
    assert response.status_code == 200
    assert response.get_json()["searched_videos"] == 1
    assert response.get_json()["skipped"] == [{"video_id": "alpha", "reason": "index_expired"}]
    assert all(hit["video_id"] == "beta" for hit in response.get_json()["results"])


def test_global_search_busy_is_reported(tmp_path):
    videos = tmp_path / "videos"
    _video(videos / "alpha.mp4")
    app = create_lite_app(video_dir=videos, index_root=tmp_path / "indexes",
                          encoder_factory=lambda *_: CorpusEncoder())
    service = app.extensions["lite_service"]
    service.index_lock.acquire()
    try:
        response = app.test_client().post("/api/lite/search_all", json={"query": "x"})
        assert response.status_code == 409
        assert response.get_json()["error"]["code"] == "index_busy"
    finally:
        service.index_lock.release()


def test_global_search_diversifies_and_backfills_across_videos(tmp_path):
    videos, indexes = tmp_path / "videos", tmp_path / "indexes"
    _video(videos / "alpha.mp4", num_frames=6, vary=True)
    _video(videos / "beta.mp4", num_frames=6, vary=True)

    scores = {
        ("alpha", 0.0): 0.99,
        ("alpha", 0.5): 0.98,
        ("alpha", 1.0): 0.97,
        ("alpha", 2.5): 0.80,
        ("beta", 0.0): 0.95,
        ("beta", 0.5): 0.94,
        ("beta", 1.0): 0.10,
        ("beta", 2.5): 0.70,
    }

    class ScoredEncoder:
        def embed(self, raw):
            s = scores.get((raw.video_id, round(raw.timestamp, 1)), 0.05)
            return np.array([s, float(np.sqrt(max(0.0, 1.0 - s * s)))], dtype=np.float32)

        def encode_text(self, query):
            return np.array([1.0, 0.0], dtype=np.float32)

    app = create_lite_app(video_dir=videos, index_root=indexes,
                          encoder_factory=lambda *_: ScoredEncoder(),
                          sample_every_s=0.5)
    service = app.extensions["lite_service"]
    service.keyframe_min_gap_s = 1.4
    client = app.test_client()
    for name in ("alpha", "beta"):
        assert client.post("/api/lite/index", json={"video_id": name}).status_code == 200

    res4 = client.post("/api/lite/search_all", json={"query": "test", "top_k": 4}).get_json()
    assert [(h["video_id"], round(h["timestamp"], 1)) for h in res4["results"]] == [
        ("alpha", 0.0), ("beta", 0.0), ("alpha", 2.5), ("beta", 2.5)
    ]
    assert [h["rank"] for h in res4["results"]] == [1, 2, 3, 4]

    res5 = client.post("/api/lite/search_all", json={"query": "test", "top_k": 5}).get_json()
    assert [(h["video_id"], round(h["timestamp"], 1)) for h in res5["results"]] == [
        ("alpha", 0.0), ("beta", 0.0), ("alpha", 2.5), ("beta", 2.5), ("alpha", 0.5)
    ]
    assert [h["rank"] for h in res5["results"]] == [1, 2, 3, 4, 5]


def test_index_all_builds_missing_reuses_cache_rebuilds_expired_and_enables_search(tmp_path):
    videos, indexes = tmp_path / "videos", tmp_path / "indexes"
    app = create_lite_app(video_dir=videos, index_root=indexes,
                          encoder_factory=lambda *_: CorpusEncoder())
    client = app.test_client()

    empty_resp = client.post("/api/lite/index_all", json={})
    assert empty_resp.status_code == 200
    assert empty_resp.get_json() == {
        "total_videos": 0, "indexed": [], "cached": [], "failed": []
    }

    for name in ("alpha", "beta", "gamma"):
        _video(videos / f"{name}.mp4")

    # Pre-index alpha (will be cached) and gamma (will be expired by rewriting video).
    assert client.post("/api/lite/index", json={"video_id": "alpha"}).status_code == 200
    assert client.post("/api/lite/index", json={"video_id": "gamma"}).status_code == 200
    _video(videos / "gamma.mp4", num_frames=5, vary=True)

    bad_rebuild = client.post("/api/lite/index_all", json={"rebuild": "yes"})
    assert bad_rebuild.status_code == 400
    assert bad_rebuild.get_json()["error"]["code"] == "invalid_rebuild"

    service = app.extensions["lite_service"]
    service.index_lock.acquire()
    try:
        busy = client.post("/api/lite/index_all", json={})
        assert busy.status_code == 409
        assert busy.get_json()["error"]["code"] == "index_busy"
    finally:
        service.index_lock.release()

    resp = client.post("/api/lite/index_all", json={"rebuild": False})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_videos"] == 3
    assert data["cached"] == ["alpha"]
    assert data["indexed"] == ["beta", "gamma"]
    assert data["failed"] == []

    progress = client.get("/api/lite/progress").get_json()
    assert progress["status"] == "ready"
    assert progress["batch_current"] is None
    assert progress["batch_total"] is None

    search = client.post("/api/lite/search_all", json={"query": "người đi bộ", "top_k": 3})
    assert search.status_code == 200
    search_data = search.get_json()
    assert search_data["searched_videos"] == 3
    assert search_data["skipped"] == []


def test_index_all_records_per_video_failure_and_continues(tmp_path):
    videos, indexes = tmp_path / "videos", tmp_path / "indexes"
    _video(videos / "alpha.mp4")
    (videos / "broken.mp4").parent.mkdir(parents=True, exist_ok=True)
    (videos / "broken.mp4").write_bytes(b"not a valid mp4")
    _video(videos / "beta.mp4")

    app = create_lite_app(video_dir=videos, index_root=indexes,
                          encoder_factory=lambda *_: CorpusEncoder())
    resp = app.test_client().post("/api/lite/index_all", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_videos"] == 3
    assert data["indexed"] == ["alpha", "beta"]
    assert data["cached"] == []
    assert len(data["failed"]) == 1
    assert data["failed"][0]["video_id"] == "broken"
    assert data["failed"][0]["code"] == "index_failed"
