"""T07 Lite HTTP contract with a real tiny video and a fake text/image encoder."""
from __future__ import annotations

import json
from io import BytesIO
from threading import Event, Thread

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                             (32, 32))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


def _subtle_motion_video(path):
    """Moving small object: nearby frames look alike in a global histogram."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1,
                             (96, 96))
    assert writer.isOpened()
    background = np.full((96, 96, 3), (90, 120, 150), dtype=np.uint8)
    for i in range(5):
        frame = background.copy()
        frame[8:20, 8 + i * 14:20 + i * 14] = (10, 40, 230)
        writer.write(frame)
    writer.release()
    return path


def test_lite_index_keeps_subtle_motion_for_search(tmp_path):
    from ui.lite_app import create_lite_app

    videos = tmp_path / "videos"
    _subtle_motion_video(videos / "motion.mp4")
    app = create_lite_app(video_dir=videos, index_root=tmp_path / "indexes",
                          encoder_factory=lambda *_: FakeEncoder())
    client = app.test_client()
    assert client.post("/api/lite/select", json={"video_id": "motion"}).status_code == 200
    response = client.post("/api/lite/index", json={})
    assert response.status_code == 200
    assert response.get_json()["sampled"] >= 2
    manifest = json.loads((tmp_path / "indexes/motion/manifest.json").read_text())
    assert manifest["sampling"]["max_frames"] == 512
    assert manifest["sampling"]["hist_duplicate_threshold"] == 0.999
    assert manifest["dedup"]["similarity_threshold"] == 0.99


def test_lite_requires_rebuild_for_old_dedup_config(tmp_path):
    from ui.lite_app import create_lite_app

    videos = tmp_path / "videos"
    _subtle_motion_video(videos / "motion.mp4")
    indexes = tmp_path / "indexes"
    old = create_lite_app(video_dir=videos, index_root=indexes,
                          encoder_factory=lambda *_: FakeEncoder(),
                          hist_duplicate_threshold=0.985,
                          dedup_threshold=0.97).test_client()
    old.post("/api/lite/select", json={"video_id": "motion"})
    assert old.post("/api/lite/index", json={}).get_json()["sampled"] == 1

    current = create_lite_app(video_dir=videos, index_root=indexes,
                              encoder_factory=lambda *_: FakeEncoder()).test_client()
    current.post("/api/lite/select", json={"video_id": "motion"})
    stale = current.post("/api/lite/index", json={})
    assert stale.status_code == 409
    assert stale.get_json()["error"]["code"] == "index_expired"
    rebuilt = current.post("/api/lite/index", json={"rebuild": True})
    assert rebuilt.status_code == 200
    assert rebuilt.get_json()["sampled"] >= 2


def test_lite_frame_budget_spans_full_video(tmp_path):
    from ui.lite_app import create_lite_app

    videos = tmp_path / "videos"
    videos.mkdir()
    writer = cv2.VideoWriter(str(videos / "long.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 1, (64, 64))
    assert writer.isOpened()
    for i in range(12):
        frame = np.full((64, 64, 3), (i * 19, 30, 200 - i * 13),
                        dtype=np.uint8)
        writer.write(frame)
    writer.release()

    app = create_lite_app(video_dir=videos, index_root=tmp_path / "indexes",
                          encoder_factory=lambda *_: FakeEncoder(),
                          max_frames=3)
    client = app.test_client()
    client.post("/api/lite/select", json={"video_id": "long"})
    response = client.post("/api/lite/index", json={})
    assert response.status_code == 200
    entry = app.extensions["lite_service"].entries["long"]
    assert max(raw.timestamp for raw in entry.raws.values()) >= 8
    assert response.get_json()["sampled"] <= 3
    manifest = json.loads((tmp_path / "indexes/long/manifest.json").read_text())
    assert manifest["sampling"]["effective_every_s"] >= 4


class FakeEncoder:
    device = "test"

    def embed(self, raw):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_text(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)


@pytest.fixture()
def setup(tmp_path):
    from ui.lite_app import create_lite_app

    calls = []

    def factory(*args):
        calls.append(args)
        return FakeEncoder()

    videos = tmp_path / "videos"
    _video(videos / "walk.mp4")
    indexes = tmp_path / "indexes"
    app = create_lite_app(video_dir=videos, index_root=indexes,
                          encoder_factory=factory)
    app.testing = True
    return app, videos, indexes, calls, factory


def test_lite_routes_and_guards_without_advanced_features(setup):
    app, _, _, calls, _ = setup
    client = app.test_client()
    assert client.get("/api/lite/health").get_json()["ok"] is True
    assert client.get("/api/lite/videos").get_json()["videos"] == ["walk"]
    assert client.get("/api/agent/ask").status_code == 404
    assert client.post("/api/video/vqa", json={}).status_code == 404
    assert client.post("/api/lite/select", json={"video_id": "../secret"}).status_code == 400
    assert client.post("/api/lite/select", json={"video_id": "missing"}).status_code == 404
    assert client.post("/api/lite/search", json={"query": " "}).status_code == 400
    assert client.post("/api/lite/search", json={"query": "đi bộ"}).status_code == 409
    assert calls == []


def test_upload_index_search_frame_and_range_stream(setup):
    app, videos, _, calls, _ = setup
    client = app.test_client()
    assert client.post("/api/lite/upload", data={"video": (BytesIO(b"x"), "bad.avi")},
                       content_type="multipart/form-data").status_code == 415
    assert client.post("/api/lite/upload", data={"video": (BytesIO(b"x"), "bad.mp4")},
                       content_type="multipart/form-data").status_code == 415
    source = (videos / "walk.mp4").read_bytes()
    uploaded = client.post("/api/lite/upload", data={
        "video": (BytesIO(source), "../../unsafe.mp4")},
        content_type="multipart/form-data")
    assert uploaded.status_code == 201
    video_id = uploaded.get_json()["video_id"]
    assert video_id != "unsafe" and (videos / f"{video_id}.mp4").is_file()
    assert "path" not in uploaded.get_json()
    assert client.post("/api/lite/select", json={"video_id": video_id}).status_code == 200
    indexed = client.post("/api/lite/index", json={})
    assert indexed.status_code == 200 and indexed.get_json()["cached"] is False
    poster_url = indexed.get_json()["poster_url"]
    assert client.get(poster_url).status_code == 200
    assert client.get("/api/lite/progress").get_json()["status"] == "ready"
    cached = client.post("/api/lite/index", json={}).get_json()
    assert cached["cached"] is True and cached["poster_url"] == poster_url
    assert len(calls) == 1
    assert client.post("/api/lite/search", json={"query": " "}).status_code == 400
    assert client.post("/api/lite/search", json={"query": "đi bộ", "top_k": 0}).status_code == 400
    response = client.post("/api/lite/search", json={"query": "đi bộ"})
    assert response.status_code == 200
    hit = response.get_json()["results"][0]
    assert {"rank", "score", "score_kind", "video_id", "frame_id",
            "timestamp", "image_url"} <= hit.keys()
    assert hit["video_id"] == video_id and hit["score_kind"] == "cosine"
    image = client.get(hit["image_url"])
    assert image.status_code == 200 and image.mimetype == "image/jpeg"
    assert client.get(f"/api/lite/frames/{video_id}/wrong").status_code == 404
    stream = client.get(f"/api/lite/videos/{video_id}/stream",
                        headers={"Range": "bytes=0-99"})
    assert stream.status_code == 206 and len(stream.data) == 100
    assert "D:\\" not in response.get_data(as_text=True)


def test_saved_index_reused_after_app_restart_and_expired_video_rejected(setup):
    from ui.lite_app import create_lite_app

    app, videos, indexes, calls, factory = setup
    client = app.test_client()
    client.post("/api/lite/select", json={"video_id": "walk"})
    assert client.post("/api/lite/index", json={}).status_code == 200
    assert len(calls) == 1
    restarted = create_lite_app(video_dir=videos, index_root=indexes,
                                encoder_factory=factory)
    second = restarted.test_client()
    assert second.post("/api/lite/select", json={"video_id": "walk"}).status_code == 200
    result = second.post("/api/lite/index", json={})
    assert result.status_code == 200 and result.get_json()["cached"] is True
    assert len(calls) == 1  # saved index loaded without loading the encoder
    assert second.post("/api/lite/search", json={"query": "đi bộ"}).status_code == 200
    assert len(calls) == 2  # one encoder instance for this app, then cached
    assert second.post("/api/lite/search", json={"query": "đi bộ"}).status_code == 200
    assert len(calls) == 2
    (videos / "walk.mp4").write_bytes(b"changed")
    assert second.post("/api/lite/search", json={"query": "đi bộ"}).status_code == 409


def test_busy_and_index_failure_have_clear_errors(setup):
    app, _, _, _, _ = setup
    client = app.test_client()
    client.post("/api/lite/select", json={"video_id": "walk"})
    service = app.extensions["lite_service"]
    service.index_lock.acquire()
    try:
        busy = client.post("/api/lite/index", json={})
        assert busy.status_code == 409
        assert busy.get_json()["error"]["code"] == "index_busy"
        frame_busy = client.get("/api/lite/frames/walk/walk/0")
        assert frame_busy.status_code == 409
        assert frame_busy.get_json()["error"]["code"] == "index_busy"
    finally:
        service.index_lock.release()
    service.encoder_factory = lambda *args: (_ for _ in ()).throw(MemoryError("OOM at C:\\secret"))
    failed = client.post("/api/lite/index", json={})
    assert failed.status_code == 503
    assert failed.get_json()["error"]["code"] == "model_oom"
    assert "secret" not in failed.get_data(as_text=True)
    assert client.get("/api/lite/progress").get_json()["status"] == "error"


def test_frame_decode_failure_is_a_clean_http_error(setup, monkeypatch):
    app, _, _, _, _ = setup
    client = app.test_client()
    client.post("/api/lite/select", json={"video_id": "walk"})
    client.post("/api/lite/index", json={})
    hit = client.post("/api/lite/search", json={"query": "đi bộ"}).get_json()["results"][0]

    def broken_frame(raw):
        raise ValueError("internal path C:\\secret")

    monkeypatch.setattr("ingestion.schemas.frame_jpeg_bytes", broken_frame)
    response = client.get(hit["image_url"])
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "frame_unavailable"
    assert "secret" not in response.get_data(as_text=True)


def test_two_frame_requests_can_decode_concurrently(setup, monkeypatch):
    from ingestion.schemas import frame_jpeg_bytes

    app, _, _, _, _ = setup
    client = app.test_client()
    client.post("/api/lite/select", json={"video_id": "walk"})
    client.post("/api/lite/index", json={})
    hit = client.post("/api/lite/search", json={"query": "đi bộ"}).get_json()["results"][0]
    entered, release = Event(), Event()
    first_response = {}

    def slow_frame(raw):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return frame_jpeg_bytes(raw)

    monkeypatch.setattr("ingestion.schemas.frame_jpeg_bytes", slow_frame)

    def request_first():
        first_response["value"] = app.test_client().get(hit["image_url"])

    thread = Thread(target=request_first)
    thread.start()
    assert entered.wait(5)
    second = client.get(hit["image_url"])
    release.set()
    thread.join(5)
    assert second.status_code == 200
    assert first_response["value"].status_code == 200


def test_upload_size_limit_returns_json(tmp_path):
    from ui.lite_app import create_lite_app

    app = create_lite_app(video_dir=tmp_path / "videos",
                          index_root=tmp_path / "indexes", max_upload_bytes=128,
                          encoder_factory=lambda *args: FakeEncoder())
    result = app.test_client().post("/api/lite/upload", data={
        "video": (BytesIO(b"a" * 1024), "huge.mp4")},
        content_type="multipart/form-data")
    assert result.status_code == 413
    assert result.get_json()["error"]["code"] == "upload_too_large"


def test_upload_write_failure_returns_clean_error(setup, monkeypatch):
    from werkzeug.datastructures import FileStorage

    app, videos, _, _, _ = setup

    def fail_save(self, dst, *args, **kwargs):
        raise OSError("internal path C:\\secret")

    monkeypatch.setattr(FileStorage, "save", fail_save)
    result = app.test_client().post("/api/lite/upload", data={
        "video": (BytesIO(b"some bytes"), "clip.mp4")},
        content_type="multipart/form-data")
    assert result.status_code == 500
    assert result.get_json()["error"]["code"] == "upload_failed"
    assert "secret" not in result.get_data(as_text=True)
    assert sorted(path.name for path in videos.glob("*.mp4")) == ["walk.mp4"]


def test_progress_is_visible_during_index_and_second_index_is_rejected(setup):
    app, _, _, _, _ = setup
    service = app.extensions["lite_service"]
    entered, release = Event(), Event()
    original = service.engine.index_video

    def slow_index(*args, **kwargs):
        kwargs["progress_cb"](1, 0.25)
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    service.engine.index_video = slow_index
    result = {}

    def run_index():
        client = app.test_client()
        client.post("/api/lite/select", json={"video_id": "walk"})
        result["response"] = client.post("/api/lite/index", json={})

    thread = Thread(target=run_index)
    thread.start()
    assert entered.wait(5)
    client = app.test_client()
    progress = client.get("/api/lite/progress").get_json()
    assert progress["status"] == "indexing" and progress["count"] == 1
    assert client.post("/api/lite/index", json={"video_id": "walk"}).status_code == 409
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert result["response"].status_code == 200
    assert client.get("/api/lite/progress").get_json()["status"] == "ready"
