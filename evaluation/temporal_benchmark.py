"""P07 temporal moment localization benchmark & ablation runner (E0, E1, E2).

Per `REWIND_LITE_IMPROVEMENT_PLAN.md` §3 P07:
- Evaluates on identical video/query/GT/sampling:
  * ``E0``: Phase 1 keyframe baseline (context only; timestamp Hit@1/Hit@5/MRR,
    never conflated with interval tIoU).
  * ``E1``: P03 temporal-only segment proposal baseline (detector OFF).
  * ``E2``: P03 candidates + P04 YOLO object detections + P05 object-aware
    reranking (detector ON, reranker ON).
  * (``E3`` omitted per P06 owner-approved NO-GO decision recorded in
    ``docs/phase2-p06-action-verification.md``).
- Never fabricates or estimates real-video accuracy metrics: if raw source
  videos are missing from the workstation, the CLI exits with code 2 and
  explicit ``BLOCKED`` status.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.error_analysis import compare_paired_ablation
from evaluation.metrics import hit_at_k, reciprocal_rank
from evaluation.temporal_metrics import (
    evaluate_temporal_predictions,
    prepare_temporal_dataset,
)
from retrieval.detector import (
    DetectionCache,
    DetectorBackend,
    DetectorConfig,
    attach_moment_detections,
)
from retrieval.lite_core import Encoder, EncoderSpec, LiteRetriever
from retrieval.object_reranker import (
    ObjectRerankConfig,
    RerankProvenance,
    rerank_temporal_moments,
)
from retrieval.temporal import (
    TemporalBaselineRetriever,
    TemporalConfig,
    TemporalMoment,
)


def _relevant_indexed_ids(entry, query_row: Mapping[str, Any]) -> set[str]:
    windows = query_row["time_windows"]
    vid = query_row["video_id"]
    sampled = {
        rid
        for rid, raw in entry.raws.items()
        if raw.video_id == vid and any(lo <= raw.timestamp < hi for lo, hi in windows)
    }
    return sampled.intersection(entry.index.ids)


def _peak_gpu_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return 0


def run_temporal_ablation_on_entry(
    entry,
    queries: Sequence[Mapping[str, Any]],
    encoders: Mapping[str, Encoder],
    encoder_specs: Mapping[str, EncoderSpec],
    *,
    temporal_config: TemporalConfig = TemporalConfig(),
    detector_config: DetectorConfig = DetectorConfig(enabled=True),
    rerank_config: ObjectRerankConfig = ObjectRerankConfig(enabled=True),
    frame_images: Mapping[str, np.ndarray] | None = None,
    detector_backend: DetectorBackend | None = None,
    detection_cache: DetectionCache | None = None,
    tiou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run E0 (context), E1 (temporal-only), and E2 (temporal + object rerank) on one shared index."""
    if not queries:
        raise ValueError("at least one query required for ablation")
    if not encoders:
        raise ValueError("at least one encoder required")

    lite_retriever = LiteRetriever(entry, encoder_specs)
    temporal_retriever = TemporalBaselineRetriever(
        entry, encoder_specs, config=temporal_config
    )
    images_map = frame_images if frame_images is not None else {}
    frame_ts_map = {
        fid: float(raw.timestamp) for fid, raw in entry.raws.items()
    }

    e0_rows: list[dict[str, Any]] = []
    e1_predictions: dict[str, list[dict[str, Any]]] = {}
    e2_predictions: dict[str, list[dict[str, Any]]] = {}
    e2_prov_by_qid: dict[str, list[dict[str, Any]]] = {}
    e2_detected_by_qid: dict[str, list[str]] = {}
    coverage_by_qid: dict[str, bool] = {}

    e0_time_s = 0.0
    e1_time_s = 0.0
    e2_det_time_s = 0.0
    e2_rerank_time_s = 0.0

    for q in queries:
        qid = str(q["query_id"])
        vid = str(q["video_id"])
        q_text = str(q.get("query_vi") or q.get("query") or "").strip()
        duration_s = float(q["duration_s"])

        rel_ids = _relevant_indexed_ids(entry, q)
        coverage_by_qid[qid] = len(rel_ids) > 0

        # Embed query once per encoder slot
        t0 = time.perf_counter()
        q_vecs = {
            slot: np.asarray(enc.encode_text(q_text), dtype=np.float32)
            for slot, enc in encoders.items()
        }
        embed_elapsed = time.perf_counter() - t0

        # --- E0: Phase 1 Keyframe Top-K Baseline (Context Only) ---
        t_e0 = time.perf_counter()
        kf_hits = lite_retriever.search_vectors(q_vecs, top_k=5)
        e0_time_s += embed_elapsed + (time.perf_counter() - t_e0)
        kf_ids = [h.frame_id for h in kf_hits]
        e0_rows.append(
            {
                "query_id": qid,
                "video_id": vid,
                "split": q.get("split", "dev"),
                "covered": coverage_by_qid[qid],
                "hit@1": float(hit_at_k(kf_ids, rel_ids, 1)),
                "hit@5": float(hit_at_k(kf_ids, rel_ids, 5)),
                "mrr": float(reciprocal_rank(kf_ids, rel_ids)),
            }
        )

        # --- E1: P03 Temporal-Only Baseline (Detector OFF, Reranker OFF) ---
        t_e1 = time.perf_counter()
        e1_moments: list[TemporalMoment] = temporal_retriever.search_temporal_vectors(
            q_vecs,
            video_duration=duration_s,
            config=temporal_config,
            video_id=vid,
        )
        e1_elapsed = embed_elapsed + (time.perf_counter() - t_e1)
        e1_time_s += e1_elapsed
        e1_predictions[qid] = [m.to_dict() for m in e1_moments]

        # --- E2: P03 Candidates + P04 YOLO Detections + P05 Object Reranking ---
        t_det = time.perf_counter()
        moments_with_dets = attach_moment_detections(
            e1_moments,
            images_map,
            config=detector_config,
            backend=detector_backend,
            cache=detection_cache,
            frame_timestamps=frame_ts_map,
        )
        e2_det_time_s += time.perf_counter() - t_det

        t_rerank = time.perf_counter()
        e2_moments, e2_prov, _ = rerank_temporal_moments(
            moments_with_dets,
            q_text,
            config=rerank_config,
        )
        e2_rerank_time_s += time.perf_counter() - t_rerank

        e2_predictions[qid] = [m.to_dict() for m in e2_moments]
        e2_prov_by_qid[qid] = [p.to_dict() for p in e2_prov]

        if e2_moments and e2_moments[0].detections:
            top1_classes = sorted(
                {
                    str(d.get("class_name", "")).strip().lower()
                    for d in e2_moments[0].detections
                    if isinstance(d, Mapping) and d.get("class_name")
                }
            )
        else:
            top1_classes = []
        e2_detected_by_qid[qid] = top1_classes

    e1_eval = evaluate_temporal_predictions(queries, e1_predictions)
    e2_eval = evaluate_temporal_predictions(queries, e2_predictions)
    paired = compare_paired_ablation(
        queries,
        e1_eval,
        e2_eval,
        e2_provenance_by_qid=e2_prov_by_qid,
        e2_detected_by_qid=e2_detected_by_qid,
        index_coverage_by_qid=coverage_by_qid,
        tiou_threshold=tiou_threshold,
    )

    n_q = len(queries)
    e0_summary = {
        "note": "Phase 1 timestamp keyframe metric (context only; not comparable to interval tIoU)",
        "n_queries": n_q,
        "n_covered": sum(1 for r in e0_rows if r["covered"]),
        "hit@1": sum(r["hit@1"] for r in e0_rows) / n_q,
        "hit@5": sum(r["hit@5"] for r in e0_rows) / n_q,
        "mrr": sum(r["mrr"] for r in e0_rows) / n_q,
    }

    return {
        "status": "MEASURED",
        "p06_action_verifier_gate": "NO-GO (Option A)",
        "configs": {
            "temporal": asdict(temporal_config),
            "detector": asdict(detector_config),
            "reranker": asdict(rerank_config),
        },
        "latency_seconds": {
            "e0_query_total_s": round(e0_time_s, 6),
            "e1_query_total_s": round(e1_time_s, 6),
            "e2_detector_total_s": round(e2_det_time_s, 6),
            "e2_rerank_total_s": round(e2_rerank_time_s, 6),
            "e2_query_total_s": round(e1_time_s + e2_det_time_s + e2_rerank_time_s, 6),
        },
        "peak_gpu_memory_bytes": _peak_gpu_bytes(),
        "E0_keyframe_context": {
            "summary": e0_summary,
            "per_query": e0_rows,
        },
        "E1_temporal_only": e1_eval,
        "E2_temporal_object_rerank": e2_eval,
        "E2_provenance": e2_prov_by_qid,
        "paired_ablation": paired,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P07 temporal ablation benchmark & paired error analysis"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--e1-predictions", type=Path, help="Ranked E1 predictions JSON")
    parser.add_argument("--e2-predictions", type=Path, help="Ranked E2 predictions JSON")
    parser.add_argument("--output", type=Path, help="Output JSON report path")
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        prepared = prepare_temporal_dataset(manifest, labels, root=args.root)
        queries = prepared["queries"]

        if args.e1_predictions is not None and args.e2_predictions is not None:
            e1_preds = json.loads(args.e1_predictions.read_text(encoding="utf-8"))
            e2_preds = json.loads(args.e2_predictions.read_text(encoding="utf-8"))
            e1_eval = evaluate_temporal_predictions(queries, e1_preds)
            e2_eval = evaluate_temporal_predictions(queries, e2_preds)
            paired = compare_paired_ablation(queries, e1_eval, e2_eval)
            report = {
                "status": "MEASURED_FROM_PREDICTIONS",
                "dataset": {k: v for k, v in prepared.items() if k != "queries"},
                "E1_temporal_only": e1_eval,
                "E2_temporal_object_rerank": e2_eval,
                "paired_ablation": paired,
            }
            out_str = json.dumps(report, ensure_ascii=False, indent=2)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(out_str + "\n", encoding="utf-8")
            print(out_str)
            return 0

        print(
            "temporal benchmark ready: dataset verified "
            f"({prepared['n_evaluable']} queries); supply --e1-predictions and "
            "--e2-predictions or invoke run_temporal_ablation_on_entry.",
            file=sys.stdout,
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"temporal benchmark BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
