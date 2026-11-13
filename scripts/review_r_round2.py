"""Independent review probes; run with python -m scripts.review_r_round2.

Reports observations, not acceptance assertions. No models/videos required.
"""
import json
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from ingestion.build_index import KeyframeIndex
from ingestion.schemas import KeyframeRecord, RawKeyframe
from retrieval.lite_core import EncoderSpec
from retrieval.object_reranker import parse_vietnamese_query_objects
from retrieval.temporal import TemporalBaselineRetriever, TemporalConfig
from retrieval.video_engine import VideoIndexEntry


def main():
    records = [KeyframeRecord("black", "v", 1.0, np.array([1., 0.], dtype=np.float32))]
    entry = VideoIndexEntry("v", KeyframeIndex.build(records),
                            {"black": RawKeyframe("black", "v", 1.0)}, 1, 1)
    cfg = TemporalConfig(dark_frame_luma_max=None, smoothing_window=1)
    retriever = TemporalBaselineRetriever(entry, {"clip": EncoderSpec("m", "r")}, config=cfg)
    kwargs = dict(video_duration=3.0, video_id="v")
    vectors = {"clip": np.array([1., 0.], dtype=np.float32)}
    plain = retriever.search_temporal_vectors(vectors, **kwargs)
    # Simulate the back-reference installed by LiteService._cache.
    entry._lite_service = SimpleNamespace(
        dark_frame_luma_max=5.0,
        _video_path=lambda vid: "unused",
        _get_dark_frame_ids=lambda *args: frozenset({"black"}),
    )
    attached = retriever.search_temporal_vectors(vectors, **kwargs)
    explicit = retriever.search_temporal_vectors(vectors, config=cfg, **kwargs)
    report = {
        "library_none_contract": {
            "config": asdict(cfg), "plain_count": len(plain),
            "service_attached_count": len(attached), "explicit_search_config_count": len(explicit),
            "contract_holds": plain == attached == explicit,
        },
        "person_mapping": [parse_vietnamese_query_objects(q).to_dict() for q in (
            "2 người đi giữa cánh đồng hoa", "hai học sinh", "cặp đôi", "đám đông",
            "nhóm học sinh", "hai học sinh cầm chai nước",
        )],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    Path("docs/benchmark-results/r-review2-probes.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
