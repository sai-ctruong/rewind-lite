"""A brief visual cue must remain searchable in the Lite web index."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ingestion.schemas import load_cv2_image
from ui.lite_app import create_lite_app


class CueEncoder:
    device = "test"

    def embed(self, raw):
        frame = load_cv2_image(raw)
        has_cue = int(frame[12:20, 12:20, 2].max()) > 150
        return np.array([0.0, 1.0] if has_cue else [1.0, 0.0],
                        dtype=np.float32)

    def encode_text(self, query):
        return np.array([0.0, 1.0], dtype=np.float32)


def test_web_indexes_half_second_event_for_top_one_search(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    video = videos / "cue.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (96, 96))
    assert writer.isOpened()
    for i in range(30):
        frame = np.full((96, 96, 3), 60, dtype=np.uint8)
        if 12 <= i < 18:  # 1.2–1.8 s, between one-second samples
            frame[12:20, 12:20] = (20, 20, 240)
        writer.write(frame)
    writer.release()

    app = create_lite_app(video_dir=videos, index_root=tmp_path / "indexes",
                          encoder_factory=lambda *_: CueEncoder())
    client = app.test_client()
    assert client.post("/api/lite/select", json={"video_id": "cue"}).status_code == 200
    assert client.post("/api/lite/index", json={}).status_code == 200
    response = client.post("/api/lite/search", json={"query": "đốm đỏ", "top_k": 1})
    assert response.status_code == 200
    hit = response.get_json()["results"][0]
    assert hit["timestamp"] == pytest.approx(1.5)
    assert hit["score"] == pytest.approx(1.0)


def test_web_keeps_small_cue_at_sampled_timestamp(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    writer = cv2.VideoWriter(str(videos / "tiny.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96))
    assert writer.isOpened()
    for i in range(30):
        frame = np.full((96, 96, 3), 60, dtype=np.uint8)
        if 10 <= i < 15:
            frame[12:16, 12:16] = (20, 20, 240)
        writer.write(frame)
    writer.release()

    app = create_lite_app(video_dir=videos, index_root=tmp_path / "indexes",
                          encoder_factory=lambda *_: CueEncoder())
    client = app.test_client()
    assert client.post("/api/lite/select", json={"video_id": "tiny"}).status_code == 200
    assert client.post("/api/lite/index", json={}).status_code == 200
    response = client.post("/api/lite/search", json={"query": "đốm đỏ", "top_k": 1})
    assert response.status_code == 200
    hit = response.get_json()["results"][0]
    assert hit["timestamp"] == pytest.approx(1.0)
    assert hit["score"] == pytest.approx(1.0)
