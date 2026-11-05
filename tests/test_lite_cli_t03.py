"""T03 baseline CLI contracts; no model download in unit tests."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.lite_search import (
    dense_results, sha256_file, validate_manifest, validate_run_args,
)


def test_empty_query_is_rejected_before_model_load(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="query"):
        validate_run_args(video, "   ", 3)


def test_missing_video_is_rejected_before_model_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_run_args(tmp_path / "missing.mp4", "người đi bộ", 3)


def test_invalid_top_k_is_rejected(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="top_k"):
        validate_run_args(video, "người đi bộ", 0)


def test_dense_results_uses_searchable_index_metadata_and_caps_top_k():
    index = SimpleNamespace(
        ids=["v/0", "v/2"], video_ids=["v", "v"], timestamps=[0.0, 2.0],
        dense_search=lambda vec, encoder, k: [(1, 0.9), (0, 0.5)][:k],
    )
    entry = SimpleNamespace(index=index, raws={"v/0": SimpleNamespace(image_path=None),
                                                "v/2": SimpleNamespace(image_path="frame.jpg")})
    results = dense_results(entry, [1.0, 0.0], 5)
    assert [(r["rank"], r["frame_id"], r["video_id"], r["timestamp"])
            for r in results] == [(1, "v/2", "v", 2.0), (2, "v/0", "v", 0.0)]
    assert results[0]["score"] == pytest.approx(0.9)
    assert results[0]["image_path"] == "frame.jpg"


def test_sha256_file_records_video_content(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"abc")
    assert sha256_file(video) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_cached_index_rejects_changed_video(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"changed")
    manifest = {"video": {"sha256": "old"},
                "model": {"id": "m", "revision": "r"}}
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(manifest, video, "m", "r")


def test_cached_index_rejects_different_encoder(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"abc")
    manifest = {"video": {"sha256": sha256_file(video)},
                "model": {"id": "old", "revision": "r"}}
    with pytest.raises(ValueError, match="model"):
        validate_manifest(manifest, video, "new", "r")
