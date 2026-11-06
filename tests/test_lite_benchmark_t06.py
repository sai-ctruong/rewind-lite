"""T06 evaluation protocol checks with synthetic vectors only."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ingestion.build_index import KeyframeIndex
from ingestion.schemas import KeyframeRecord, RawKeyframe
from retrieval.lite_core import EncoderSpec
from retrieval.video_engine import VideoIndexEntry


def _entry():
    records = [
        KeyframeRecord("a/0", "a", 0.5, np.array([1., 0.], dtype=np.float32),
                       siglip_embedding=np.array([0., 1.], dtype=np.float32)),
        KeyframeRecord("a/1", "a", 1.5, np.array([0., 1.], dtype=np.float32),
                       siglip_embedding=np.array([1., 0.], dtype=np.float32)),
        KeyframeRecord("b/0", "b", 0.5, np.array([-1., 0.], dtype=np.float32),
                       siglip_embedding=np.array([-1., 0.], dtype=np.float32)),
    ]
    raws = {r.id: RawKeyframe(r.id, r.video_id, r.timestamp) for r in records}
    raws["a/dropped"] = RawKeyframe("a/dropped", "a", 2.5)
    return VideoIndexEntry("all", KeyframeIndex.build(records), raws, 4, 3)


class Encoder:
    def __init__(self, vector):
        self.vector = np.array(vector, dtype=np.float32)

    def encode_text(self, query):
        return self.vector


def _label(qid, video, lo, hi, split="dev"):
    return {"query_id": qid, "video_id": video, "query": "người đi bộ",
            "language": "vi", "time_windows": [[lo, hi]], "category": "action",
            "split": split, "review_status": "verified"}


def test_three_baselines_share_index_and_report_t02_outcomes():
    from evaluation.lite_benchmark import evaluate_entry

    entry = _entry()
    labels = [_label("hit", "a", 0, 1), _label("dedup", "a", 2, 3),
              _label("missed-sampling", "a", 4, 5),
              _label("test-hit", "b", 0, 1, "test")]
    specs = {"clip": EncoderSpec("e1", "r1"),
             "siglip": EncoderSpec("e2", "r2")}
    report = evaluate_entry(entry, labels,
                            {"clip": Encoder([1, 0]), "siglip": Encoder([0, 1])},
                            specs, top_k=5)
    assert set(report["baselines"]) == {"E1", "E2", "E3"}
    e1 = report["baselines"]["E1"]
    assert e1["dev"]["n_queries"] == 3
    assert e1["dev"]["n_covered"] == 1
    assert e1["dev"]["hit@1"] == pytest.approx(1 / 3)
    assert e1["dev"]["coverage"] == pytest.approx(1 / 3)
    rows = {r["query_id"]: r for r in report["per_query"]["E1"]}
    assert rows["hit"]["predictions"][0]["frame_id"] == "a/0"
    assert rows["hit"]["predictions"][0]["score_kind"] == "cosine"
    assert rows["dedup"]["outcome"] == "dedup_removed_positive"
    assert rows["missed-sampling"]["outcome"] == "no_sampled_positive"
    assert report["per_query"]["E3"][0]["predictions"][0]["score_kind"] == "rrf"
    assert e1["dev"]["mrr_covered"] == 1
    assert report["timing"]["E1"]["embedding_seconds_total"] >= 0
    assert report["timing"]["E1"]["retrieval_seconds_total"] >= 0


def test_unready_dataset_blocks_before_model_or_accuracy_report(tmp_path):
    from evaluation.lite_benchmark import DatasetBlocked, preflight

    manifest = tmp_path / "manifest.json"
    labels = tmp_path / "labels.json"
    manifest.write_text('{"schema_version":1,"videos":[]}', encoding="utf-8")
    labels.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetBlocked) as exc:
        preflight(manifest, labels, root=tmp_path)
    assert exc.value.report.n_verified == 0

    cmd = [sys.executable, "-m", "evaluation.lite_benchmark", "--manifest",
           str(manifest), "--labels", str(labels), "--root", str(tmp_path),
           "--output", str(tmp_path / "result.json")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 3
    assert json.loads(result.stdout)["status"] == "DATA_BLOCKED"
    assert not (tmp_path / "result.json").exists()


def test_video_index_dataset_accepts_explicit_manifest_ids(monkeypatch, tmp_path):
    from retrieval.video_engine import VideoSearchEngine

    engine = VideoSearchEngine(enable_ocr=False)
    seen = {}

    def collect(specs, *args, **kwargs):
        seen["specs"] = specs
        raise RuntimeError("stop after specs")

    monkeypatch.setattr(engine, "_collect_records", collect)
    with pytest.raises(RuntimeError, match="stop after specs"):
        engine.index_dataset([tmp_path / "source.mp4"], tmp_path,
                             video_ids=["stable-id"])
    assert seen["specs"] == [(tmp_path / "source.mp4", "stable-id")]
    with pytest.raises(ValueError, match="match video_paths"):
        engine.index_dataset([tmp_path / "source.mp4"], tmp_path, video_ids=[])
    with pytest.raises(ValueError, match="unique"):
        engine.index_dataset([tmp_path / "a.mp4", tmp_path / "b.mp4"],
                             tmp_path, video_ids=["same", "same"])
