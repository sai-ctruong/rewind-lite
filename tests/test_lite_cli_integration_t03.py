"""T03 persistence contract with a local synthetic video; no model download."""
from __future__ import annotations

import json
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

from scripts.lite_search import run_baseline, sha256_file
from tests.test_video_engine import ColorMockEncoder, _make_video


def test_single_encoder_index_then_reuse_without_embedding_video(tmp_path):
    video = tmp_path / "sample.mp4"
    _make_video(video, [(0, 0, 255), (255, 0, 0)], frames_per_color=10)
    output = tmp_path / "index"
    kwargs = dict(index_dir=output, sample_every_s=0.2, max_frames=20,
                  top_k=2, model_id="test-model", revision="test-rev")

    first = run_baseline(video, "màu đỏ", encoder=ColorMockEncoder(), **kwargs)
    assert first["index_reused"] is False
    assert first["n_sampled"] >= first["n_indexed"] >= 2
    assert first["results"][0]["video_id"] == "sample"
    assert first["results"][0]["timestamp"] < 1.0
    assert (output / "manifest.json").is_file()
    assert (output / "entry.pkl").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video"]["sha256"] == sha256_file(video)
    assert manifest["video"]["duration_s"] > 0
    assert manifest["model"] == {"id": "test-model", "revision": "test-rev"}
    assert manifest["index"]["hnsw_m"] == 32

    class TextOnlyEncoder(ColorMockEncoder):
        def embed(self, raw):
            raise AssertionError("cached search must not embed video again")

    second = run_baseline(video, "màu đỏ", encoder=TextOnlyEncoder(), **kwargs)
    assert second["index_reused"] is True
    assert second["results"][0]["frame_id"] == first["results"][0]["frame_id"]


def test_corrupt_video_is_rejected_before_indexing(tmp_path):
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not an mp4")
    with pytest.raises(ValueError, match="video"):
        run_baseline(video, "người đi bộ", index_dir=tmp_path / "index",
                     encoder=ColorMockEncoder(), model_id="test", revision="test")
