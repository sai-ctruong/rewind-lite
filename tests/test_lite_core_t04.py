"""T04 Lite contracts and legacy-index adapter behavior."""
from __future__ import annotations

import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ingestion.build_index import KeyframeIndex
from ingestion.schemas import KeyframeRecord, RawKeyframe
from retrieval.video_engine import VideoIndexEntry


def _entry(records):
    index = KeyframeIndex.build(records) if records else KeyframeIndex([], [], [], [])
    raws = {r.id: RawKeyframe(id=r.id, video_id=r.video_id,
                              timestamp=r.timestamp, image_path=f"{r.id}.jpg")
            for r in records}
    return VideoIndexEntry("v", index, raws, len(records), len(records))


def _records():
    return [
        KeyframeRecord("v/b", "v", 2.0, np.array([1.0, 0.0], dtype=np.float32),
                       siglip_embedding=np.array([0.0, 1.0], dtype=np.float32)),
        KeyframeRecord("v/a", "v", 1.0, np.array([2.0, 0.0], dtype=np.float32),
                       siglip_embedding=np.array([1.0, 0.0], dtype=np.float32)),
        KeyframeRecord("v/c", "v", 3.0, np.array([0.0, 1.0], dtype=np.float32),
                       siglip_embedding=np.array([0.0, 2.0], dtype=np.float32)),
    ]


def test_single_encoder_cosine_contract_and_stable_tie():
    from retrieval.lite_core import EncoderSpec, LiteRetriever, VectorIndex

    entry = _entry(_records())
    assert isinstance(entry.index, VectorIndex)
    assert entry.index.dimension("clip") == 2
    with pytest.raises(ValueError, match="encoder"):
        entry.index.dimension("other")
    with pytest.raises(ValueError, match="encoder"):
        entry.index.has_encoder("other")
    retriever = LiteRetriever(entry, {"clip": EncoderSpec("model-a", "rev-a")})
    hits = retriever.search_vectors({"clip": np.array([1.0, 0.0])}, top_k=99)
    assert [h.frame_id for h in hits] == ["v/a", "v/b", "v/c"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.timestamp for h in hits] == [1.0, 2.0, 3.0]
    assert hits[0].video_id == "v"
    assert hits[0].image_path == "v/a.jpg"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score_kind == "cosine"
    assert hits[0].encoder_versions == {"clip": "model-a@rev-a"}
    assert len(retriever.search_vectors({"clip": [1.0, 0.0]}, top_k=1)) == 1
    assert np.linalg.norm(entry.index._vectors("clip", [0])[0]) == pytest.approx(1.0)


def test_two_encoders_use_rrf_and_keep_score_type():
    from retrieval.lite_core import EncoderSpec, LiteRetriever

    entry = _entry(_records())
    retriever = LiteRetriever(entry, {"clip": EncoderSpec("a", "1"),
                                      "siglip": EncoderSpec("b", "2")})
    hits = retriever.search_vectors({"clip": [1.0, 0.0], "siglip": [0.0, 1.0]},
                                    top_k=2)
    assert len(hits) == 2
    assert hits[0].score_kind == "rrf"
    assert hits[0].source_ranks.keys() == {"clip", "siglip"}
    assert hits[0].encoder_versions == {"clip": "a@1", "siglip": "b@2"}


def test_lite_retriever_rejects_mismatched_persisted_model():
    from retrieval.lite_core import EncoderSpec, LiteRetriever

    entry = _entry(_records())
    entry.index.encoder_metadata = {"clip": {"id": "saved", "revision": "1"}}
    with pytest.raises(ValueError, match="model"):
        LiteRetriever(entry, {"clip": EncoderSpec("other", "2")})


@pytest.mark.parametrize("query", [[1.0], [0.0, 0.0], [float("nan"), 0.0],
                                         [float("inf"), 0.0]])
def test_lite_rejects_invalid_query_vector(query):
    from retrieval.lite_core import EncoderSpec, LiteRetriever

    retriever = LiteRetriever(_entry(_records()), {"clip": EncoderSpec("a", "1")})
    with pytest.raises(ValueError, match="dimension|finite|nonzero"):
        retriever.search_vectors({"clip": query})


def test_empty_query_and_empty_index_are_handled():
    from retrieval.lite_core import EncoderSpec, LiteRetriever

    retriever = LiteRetriever(_entry([]), {"clip": EncoderSpec("a", "1")})
    with pytest.raises(ValueError, match="query"):
        retriever.search_text("  ", {"clip": object()})
    assert retriever.search_vectors({"clip": [1.0, 0.0]}) == []


def test_versioned_index_load_and_legacy_payload(tmp_path):
    index = _entry(_records()).index
    index.encoder_metadata = {"clip": {"id": "a", "revision": "1"}}
    index.save(tmp_path)
    meta = tmp_path / "meta.pkl"
    with meta.open("rb") as fh:
        payload = pickle.load(fh)
    assert payload["schema_version"] == 1
    loaded = KeyframeIndex.load(tmp_path, expected_encoder_metadata=index.encoder_metadata,
                                expected_dimensions={"clip": 2})
    assert loaded.ids == index.ids
    with pytest.raises(ValueError, match="model"):
        KeyframeIndex.load(tmp_path, expected_encoder_metadata={"clip": {"id": "b", "revision": "2"}})
    with pytest.raises(ValueError, match="dimension"):
        KeyframeIndex.load(tmp_path, expected_dimensions={"clip": 3})
    payload.pop("schema_version")
    payload.pop("encoder_metadata")
    with meta.open("wb") as fh:
        pickle.dump(payload, fh)
    assert KeyframeIndex.load(tmp_path).ids == index.ids
    payload["schema_version"] = 999
    with meta.open("wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(ValueError, match="version"):
        KeyframeIndex.load(tmp_path)


def test_lite_import_does_not_load_heavy_retrieval_modules():
    script = (
        "import sys; import retrieval.lite_core; import retrieval.video_engine; "
        "blocked=('retrieval.fine_rerank','retrieval.temporal_check',"
        "'retrieval.vlm_rerank','retrieval.query_understanding',"
        "'retrieval.dialogue'); "
        "assert not [name for name in blocked if name in sys.modules]"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_t03_cli_uses_lite_contract_for_saved_index(tmp_path):
    pytest.importorskip("cv2")
    from scripts.lite_search import run_baseline
    from tests.test_video_engine import ColorMockEncoder, _make_video

    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (255, 0, 0)], frames_per_color=10)
    kwargs = dict(index_dir=tmp_path / "index", sample_every_s=0.2,
                  model_id="test-model", revision="test-rev", top_k=2)
    first = run_baseline(video, "màu đỏ", encoder=ColorMockEncoder(), **kwargs)
    assert first["results"][0]["score_kind"] == "cosine"
    assert first["results"][0]["encoder_versions"] == {
        "clip": "test-model@test-rev"}
    loaded = KeyframeIndex.load(tmp_path / "index" / "index",
                                expected_encoder_metadata={
                                    "clip": {"id": "test-model", "revision": "test-rev"}})
    assert loaded.ids
    for result in first["results"]:
        row = loaded.ids.index(result["frame_id"])
        assert (result["video_id"], result["timestamp"]) == (
            loaded.video_ids[row], loaded.timestamps[row])
    second = run_baseline(video, "màu đỏ", encoder=ColorMockEncoder(), **kwargs)
    assert second["index_reused"] is True
    assert second["results"] == first["results"]


def test_multi_encoder_rank_is_independent_of_requested_top_k():
    from retrieval.lite_core import EncoderSpec, LiteRetriever

    lists = {"clip": [0, 1, 2], "siglip": [1, 2, 0]}
    index = SimpleNamespace(
        ids=["a", "b", "c"], video_ids=["v"] * 3,
        timestamps=[0.0, 1.0, 2.0],
        has_encoder=lambda slot: slot in lists,
        dimension=lambda slot: 2,
        dense_search=lambda q, slot, k: [(r, 1.0 - i * 0.1)
                                         for i, r in enumerate(lists[slot][:k])],
    )
    entry = SimpleNamespace(index=index, raws={i: SimpleNamespace(image_path=None)
                                              for i in index.ids})
    retriever = LiteRetriever(entry, {s: EncoderSpec(s, "1") for s in lists},
                              exact_threshold=0)
    one = retriever.search_vectors({s: [1.0, 0.0] for s in lists}, top_k=1)
    three = retriever.search_vectors({s: [1.0, 0.0] for s in lists}, top_k=3)
    assert one[0].frame_id == three[0].frame_id


@pytest.mark.parametrize("bad", [[float("nan"), 1.0], [float("inf"), 1.0],
                                       [0.0, 0.0]])
def test_index_rejects_nonfinite_or_zero_embedding(bad):
    record = KeyframeRecord("v/0", "v", 0.0, np.asarray(bad, dtype=np.float32))
    with pytest.raises(ValueError, match="embedding"):
        KeyframeIndex.build([record])


def test_index_rejects_mixed_dimensions():
    records = [KeyframeRecord("v/0", "v", 0.0, np.ones(2)),
               KeyframeRecord("v/1", "v", 1.0, np.ones(3))]
    with pytest.raises(ValueError, match="dimension"):
        KeyframeIndex.build(records)


@pytest.mark.parametrize("records, error", [
    ([KeyframeRecord("same", "v", 0.0, np.ones(2)),
      KeyframeRecord("same", "v", 1.0, np.ones(2))], "duplicate"),
    ([KeyframeRecord("v/0", "v", float("nan"), np.ones(2))], "timestamp"),
    ([KeyframeRecord("v/0", "v", -1.0, np.ones(2))], "timestamp"),
])
def test_index_rejects_ambiguous_ids_and_timestamps(records, error):
    with pytest.raises(ValueError, match=error):
        KeyframeIndex.build(records)


def test_entry_format_version_and_legacy_roundtrip(tmp_path):
    entry = _entry(_records())
    entry.save(tmp_path)
    path = tmp_path / "entry.pkl"
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    assert payload["schema_version"] == 1
    payload.pop("schema_version")
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    assert VideoIndexEntry.load(tmp_path).index.ids == entry.index.ids
    payload["schema_version"] = 999
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(ValueError, match="version"):
        VideoIndexEntry.load(tmp_path)
