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


def _video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (32, 32))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((32, 32, 3), 120, dtype=np.uint8))
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
