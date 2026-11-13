"""Golden evaluation runner for Rewind Lite Phase 1 vs Phase 2.

Evaluates fixed golden queries across:
- Phase 1: Keyframe retrieval (exact cosine)
- Phase 2: Temporal moment proposal (P03 baseline)
- Phase 2 + YOLO: Object-aware reranking (P05)

Usage:
    python -m scripts.eval_golden_lite
    python -m scripts.eval_golden_lite --no-yolo
    python -m scripts.eval_golden_lite --temporal relative_peak_threshold=0.0 --no-yolo
    python -m scripts.eval_golden_lite --rerank max_object_boost=0.15
    python -m scripts.eval_golden_lite --out docs/benchmark-results/golden_lite_baseline.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Ensure root is in sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.temporal_metrics import temporal_iou
from retrieval.lite_core import EncoderSpec
from retrieval.object_reranker import ObjectRerankConfig, parse_vietnamese_query_objects, rerank_temporal_moments
from retrieval.temporal import TemporalBaselineRetriever, TemporalConfig, TemporalMoment
from ui.lite_app import LiteService


def parse_override_dict(overrides: Sequence[str]) -> dict[str, Any]:
    """Parse key=value override strings into typed Python values."""
    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format (expected key=value): {item!r}")
        key, val_str = item.split("=", 1)
        key = key.strip()
        val_str = val_str.strip()
        val_lower = val_str.lower()
        if val_lower in ("none", "null"):
            val: Any = None
        elif val_lower in ("true", "yes"):
            val = True
        elif val_lower in ("false", "no"):
            val = False
        else:
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
        result[key] = val
    return result


def is_timestamp_hit(timestamp: float, gt_windows: Sequence[Sequence[float]], tol: float = 1.0) -> bool:
    """Check if a timestamp falls within any ground truth interval (+/- tol seconds)."""
    return any(lo - tol <= timestamp <= hi + tol for lo, hi in gt_windows)


def best_tiou(interval: tuple[float, float], gt_windows: Sequence[Sequence[float]]) -> float:
    """Compute the maximum temporal IoU of an interval against all ground truth windows."""
    if not gt_windows:
        return 0.0
    return max(temporal_iou(interval, (float(w[0]), float(w[1]))) for w in gt_windows)


def compute_mode_metrics(eval_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate aggregate summary metrics (@1, @3, mIoU@1, R@1(tIoU>=0.5), median latency).

    Queries with empty ground truth windows (negative test cases) are excluded
    from @k, mIoU, and R@1 calculations, but their latency is included in the median.
    """
    evaluable = [r for r in eval_rows if r.get("gt_windows")]
    total_evaluable = len(evaluable)

    latencies = [float(r["latency_ms"]) for r in eval_rows if "latency_ms" in r]
    median_latency = round(statistics.median(latencies), 1) if latencies else 0.0

    if total_evaluable == 0:
        return {
            "n_evaluable": 0,
            "hit_at_1": 0.0,
            "hit_at_1_count": 0,
            "hit_at_3": 0.0,
            "hit_at_3_count": 0,
            "miou_at_1": 0.0,
            "r1_tiou_05": 0.0,
            "r1_tiou_05_count": 0,
            "median_latency_ms": median_latency,
        }

    hit1_count = sum(1 for r in evaluable if r.get("hit_at_1", False))
    hit3_count = sum(1 for r in evaluable if r.get("hit_at_3", False))

    # mIoU@1 and R@1(tIoU>=0.5) apply to temporal intervals (if available)
    has_tiou = any("tiou_at_1" in r and r["tiou_at_1"] is not None for r in evaluable)
    if has_tiou:
        tious = [float(r.get("tiou_at_1", 0.0) or 0.0) for r in evaluable]
        miou = round(sum(tious) / total_evaluable, 3)
        r1_tiou_05_count = sum(1 for t in tious if t >= 0.5 - 1e-9)
        r1_tiou_05 = round(r1_tiou_05_count / total_evaluable, 3)
    else:
        miou = None
        r1_tiou_05 = None
        r1_tiou_05_count = None

    return {
        "n_evaluable": total_evaluable,
        "hit_at_1": round(hit1_count / total_evaluable, 3),
        "hit_at_1_count": hit1_count,
        "hit_at_3": round(hit3_count / total_evaluable, 3),
        "hit_at_3_count": hit3_count,
        "miou_at_1": miou,
        "r1_tiou_05": r1_tiou_05,
        "r1_tiou_05_count": r1_tiou_05_count,
        "median_latency_ms": median_latency,
    }


def format_time_s(t: float) -> str:
    """Format seconds into MM:SS string."""
    val = max(0, int(round(t)))
    return f"{val // 60:02d}:{val % 60:02d}"


def load_golden_queries(queries_path: str | Path) -> list[dict[str, Any]]:
    path = Path(queries_path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden queries file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected nonempty list in {path}")
    return data


def build_temporal_retriever(service, entry, specs, video_id, path, config):
    """Resolve luma evidence explicitly; None never consults service defaults."""
    dark_ids = (service._get_dark_frame_ids(video_id, path, entry, config.dark_frame_luma_max)
                if config.dark_frame_luma_max is not None else frozenset())
    return TemporalBaselineRetriever(entry, specs, config=config, dark_frame_ids=dark_ids)


def run_evaluation(
    service: LiteService,
    queries: Sequence[Mapping[str, Any]],
    *,
    temporal_config: TemporalConfig,
    rerank_config: ObjectRerankConfig,
    no_yolo: bool = False,
    detector_backend: Any = None,
) -> dict[str, Any]:
    """Execute the full golden query evaluation across Phase 1, Phase 2, and Phase 2+YOLO."""
    import cv2
    from ingestion.schemas import frame_jpeg_bytes
    from retrieval.detector import DetectorConfig, attach_moment_detections

    # Ensure encoder and entries are cached
    for q in queries:
        vid = q["video_id"]
        vpath = service._video_path(vid)
        service._checked_entry(vid, vpath)
    encoder = service._get_encoder()
    specs = {"clip": EncoderSpec(service.model_id, service.revision)}

    p1_rows: list[dict[str, Any]] = []
    p2_rows: list[dict[str, Any]] = []
    p3_rows: list[dict[str, Any]] = []

    for item in queries:
        qid = item["id"]
        vid = item["video_id"]
        query_vi = item["query_vi"]
        gt = item.get("gt_windows", [])
        vpath = service._video_path(vid)
        entry = service.entries[vid]

        # Determine video duration
        duration = float(max(entry.index.timestamps, default=0.0)) + 1.0
        manifest_file = service.index_root / vid / "manifest.json"
        if manifest_file.is_file():
            try:
                m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                d = float(m_data.get("video", {}).get("duration_s", 0.0))
                if d > 0:
                    duration = d
            except Exception:
                pass

        # -------------------------------------------------------------------
        # 1. Phase 1 Search
        # -------------------------------------------------------------------
        t_start = time.perf_counter()
        p1_res = service.search(vid, query_vi, top_k=temporal_config.top_k)
        p1_ms = round((time.perf_counter() - t_start) * 1000, 2)

        p1_hits = [is_timestamp_hit(r.timestamp, gt) for r in p1_res]
        p1_rows.append({
            "id": qid,
            "video_id": vid,
            "query_vi": query_vi,
            "gt_windows": gt,
            "latency_ms": p1_ms,
            "hit_at_1": p1_hits[0] if p1_hits else False,
            "hit_at_3": any(p1_hits[:3]),
            "top1_timestamp": p1_res[0].timestamp if p1_res else None,
            "top1_score": float(p1_res[0].score) if p1_res else None,
            "results": [
                {
                    "rank": idx + 1,
                    "frame_id": r.frame_id,
                    "timestamp": r.timestamp,
                    "score": float(r.score),
                    "hit": is_timestamp_hit(r.timestamp, gt),
                }
                for idx, r in enumerate(p1_res)
            ],
        })

        # -------------------------------------------------------------------
        # 2. Phase 2 Temporal Baseline (no YOLO)
        # -------------------------------------------------------------------
        t_start = time.perf_counter()
        retriever = build_temporal_retriever(service, entry, specs, vid, vpath, temporal_config)
        moments = retriever.search_temporal_text(
            query_vi, {"clip": encoder}, video_duration=duration, video_id=vid
        )
        p2_ms = round((time.perf_counter() - t_start) * 1000, 2)

        p2_peaks = [getattr(m, "peak_timestamp", m.t_start) for m in moments]
        p2_hits = [is_timestamp_hit(ts, gt) for ts in p2_peaks]
        top1_tiou = best_tiou((moments[0].t_start, moments[0].t_end), gt) if moments and gt else 0.0

        p2_rows.append({
            "id": qid,
            "video_id": vid,
            "query_vi": query_vi,
            "gt_windows": gt,
            "latency_ms": p2_ms,
            "hit_at_1": p2_hits[0] if p2_hits else False,
            "hit_at_3": any(p2_hits[:3]),
            "tiou_at_1": round(top1_tiou, 3) if gt else None,
            "top1_interval": [moments[0].t_start, moments[0].t_end] if moments else None,
            "top1_peak_timestamp": p2_peaks[0] if p2_peaks else None,
            "top1_score": float(moments[0].score) if moments else None,
            "results": [
                {
                    "rank": m.rank,
                    "t_start": m.t_start,
                    "t_end": m.t_end,
                    "peak_timestamp": getattr(m, "peak_timestamp", m.t_start),
                    "score": float(m.score),
                    "hit": is_timestamp_hit(getattr(m, "peak_timestamp", m.t_start), gt),
                    "tiou": round(best_tiou((m.t_start, m.t_end), gt), 3) if gt else 0.0,
                }
                for m in moments
            ],
        })

        # -------------------------------------------------------------------
        # 3. Phase 2 + YOLO (Object Reranked)
        # -------------------------------------------------------------------
        if not no_yolo:
            t_start = time.perf_counter()
            query_spec = parse_vietnamese_query_objects(query_vi)
            if moments and query_spec.mapping_supported:
                needed_ids = list(dict.fromkeys(
                    fid for m in moments for fid in m.evidence_frame_ids if fid in entry.raws
                ))
                frame_images: dict[str, np.ndarray] = {}
                frame_ts: dict[str, float] = {}
                for fid in needed_ids:
                    raw = entry.raws[fid]
                    frame_ts[fid] = float(raw.timestamp)
                    safe_raw = replace(raw, image_path=None, source_video=str(vpath))
                    img_bytes = frame_jpeg_bytes(safe_raw)
                    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        frame_images[fid] = arr

                det_cfg = DetectorConfig(enabled=True, allowed_classes=query_spec.target_classes)
                moments_with_dets = attach_moment_detections(
                    moments, frame_images, config=det_cfg, backend=detector_backend, frame_timestamps=frame_ts
                )
                reranked_moments, provenances, _ = rerank_temporal_moments(
                    moments_with_dets, query_vi, config=rerank_config
                )
                final_moments = reranked_moments
                p3_mode = "temporal_object"
            else:
                final_moments = moments
                provenances = []
                p3_mode = "temporal_only_fallback"

            p3_ms = round((time.perf_counter() - t_start) * 1000, 2)
            p3_peaks = [getattr(m, "peak_timestamp", m.t_start) for m in final_moments]
            p3_hits = [is_timestamp_hit(ts, gt) for ts in p3_peaks]
            p3_top1_tiou = (
                best_tiou((final_moments[0].t_start, final_moments[0].t_end), gt)
                if final_moments and gt
                else 0.0
            )

            p3_rows.append({
                "id": qid,
                "video_id": vid,
                "query_vi": query_vi,
                "gt_windows": gt,
                "latency_ms": p3_ms,
                "mode": p3_mode,
                "hit_at_1": p3_hits[0] if p3_hits else False,
                "hit_at_3": any(p3_hits[:3]),
                "tiou_at_1": round(p3_top1_tiou, 3) if gt else None,
                "top1_interval": [final_moments[0].t_start, final_moments[0].t_end] if final_moments else None,
                "top1_peak_timestamp": p3_peaks[0] if p3_peaks else None,
                "top1_score": float(final_moments[0].score) if final_moments else None,
                "results": [
                    {
                        "rank": m.rank,
                        "t_start": m.t_start,
                        "t_end": m.t_end,
                        "peak_timestamp": getattr(m, "peak_timestamp", m.t_start),
                        "score": float(m.score),
                        "hit": is_timestamp_hit(getattr(m, "peak_timestamp", m.t_start), gt),
                        "tiou": round(best_tiou((m.t_start, m.t_end), gt), 3) if gt else 0.0,
                    }
                    for m in final_moments
                ],
            })

    summary: dict[str, Any] = {
        "phase1": compute_mode_metrics(p1_rows),
        "phase2_temporal": compute_mode_metrics(p2_rows),
    }
    if not no_yolo:
        summary["phase2_yolo"] = compute_mode_metrics(p3_rows)

    return {
        "summary": summary,
        "details": {
            "phase1": p1_rows,
            "phase2_temporal": p2_rows,
            **({"phase2_yolo": p3_rows} if not no_yolo else {}),
        },
    }


def print_summary_table(eval_result: Mapping[str, Any], *, no_yolo: bool = False) -> None:
    """Print an aligned ASCII terminal table of evaluation results."""
    summary = eval_result["summary"]
    p1 = summary["phase1"]
    p2 = summary["phase2_temporal"]
    p3 = summary.get("phase2_yolo")

    n_eval = p1["n_evaluable"]

    print("\n" + "=" * 78)
    print(" REWIND LITE GOLDEN EVALUATION SUMMARY")
    print("=" * 78)
    header = f"{'Mode':<26} | {'Hit@1':<10} | {'Hit@3':<10} | {'mIoU@1':<8} | {'R@1(>=0.5)':<11} | {'Latency':<8}"
    print(header)
    print("-" * 78)

    row1 = (
        f"{'Phase 1 (Khung hình đơn)':<26} | "
        f"{p1['hit_at_1_count']}/{n_eval:<7} | "
        f"{p1['hit_at_3_count']}/{n_eval:<7} | "
        f"{'N/A':<8} | "
        f"{'N/A':<11} | "
        f"{p1['median_latency_ms']:>6.1f} ms"
    )
    print(row1)

    r1_str_p2 = f"{p2['r1_tiou_05_count']}/{n_eval}" if p2['r1_tiou_05_count'] is not None else "N/A"
    row2 = (
        f"{'Phase 2 (Khoảng tg - P03)':<26} | "
        f"{p2['hit_at_1_count']}/{n_eval:<7} | "
        f"{p2['hit_at_3_count']}/{n_eval:<7} | "
        f"{p2['miou_at_1']:<8.3f} | "
        f"{r1_str_p2:<11} | "
        f"{p2['median_latency_ms']:>6.1f} ms"
    )
    print(row2)

    if not no_yolo and p3:
        r1_str_p3 = f"{p3['r1_tiou_05_count']}/{n_eval}" if p3['r1_tiou_05_count'] is not None else "N/A"
        row3 = (
            f"{'Phase 2 + YOLO (P05)':<26} | "
            f"{p3['hit_at_1_count']}/{n_eval:<7} | "
            f"{p3['hit_at_3_count']}/{n_eval:<7} | "
            f"{p3['miou_at_1']:<8.3f} | "
            f"{r1_str_p3:<11} | "
            f"{p3['median_latency_ms']:>6.1f} ms"
        )
        print(row3)

    print("=" * 78 + "\n")


def print_detailed_table(eval_result: Mapping[str, Any], *, no_yolo: bool = False) -> None:
    """Print per-query Rank-1 breakdown table."""
    p1_rows = eval_result["details"]["phase1"]
    p2_rows = eval_result["details"]["phase2_temporal"]
    p3_rows = eval_result["details"].get("phase2_yolo", [])

    print("Per-query Rank-1 Breakdown:")
    print("-" * 105)
    header = f"{'ID':<4} | {'Query':<30} | {'P1 Top-1':<14} | {'P2 Top-1':<24} | {'P2+YOLO Top-1':<24}"
    print(header)
    print("-" * 105)

    for idx, r1 in enumerate(p1_rows):
        qid = r1["id"]
        qtext = r1["query_vi"][:28]
        p1_ts_str = format_time_s(r1["top1_timestamp"]) if r1["top1_timestamp"] is not None else "--:--"
        p1_mark = "OK" if r1["hit_at_1"] else ("--" if not r1["gt_windows"] else "MISS")
        p1_col = f"{p1_ts_str} ({p1_mark})"

        r2 = p2_rows[idx]
        p2_peak = format_time_s(r2["top1_peak_timestamp"]) if r2["top1_peak_timestamp"] is not None else "--:--"
        p2_inter = (
            f"[{format_time_s(r2['top1_interval'][0])}-{format_time_s(r2['top1_interval'][1])}]"
            if r2["top1_interval"]
            else "[--:----:--]"
        )
        p2_mark = "OK" if r2["hit_at_1"] else ("--" if not r2["gt_windows"] else "MISS")
        p2_col = f"{p2_peak} {p2_inter} ({p2_mark})"

        if not no_yolo and p3_rows:
            r3 = p3_rows[idx]
            p3_peak = format_time_s(r3["top1_peak_timestamp"]) if r3["top1_peak_timestamp"] is not None else "--:--"
            p3_inter = (
                f"[{format_time_s(r3['top1_interval'][0])}-{format_time_s(r3['top1_interval'][1])}]"
                if r3["top1_interval"]
                else "[--:----:--]"
            )
            p3_mark = "OK" if r3["hit_at_1"] else ("--" if not r3["gt_windows"] else "MISS")
            p3_col = f"{p3_peak} {p3_inter} ({p3_mark})"
        else:
            p3_col = "N/A"

        print(f"{qid:<4} | {qtext:<30} | {p1_col:<14} | {p2_col:<24} | {p3_col:<24}")
    print("-" * 105 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Rewind Lite golden queries.")
    parser.add_argument(
        "--queries",
        type=str,
        default="evaluation/golden_lite_queries.json",
        help="Path to golden queries JSON file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write output benchmark JSON file.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        default=False,
        help="Skip Phase 2 + YOLO mode (speeds up ablation sweeps).",
    )
    parser.add_argument(
        "--temporal",
        action="append",
        default=[],
        help="Override TemporalConfig hyperparameter: key=value (e.g. --temporal relative_peak_threshold=0.0).",
    )
    parser.add_argument(
        "--rerank",
        action="append",
        default=[],
        help="Override ObjectRerankConfig hyperparameter: key=value (e.g. --rerank max_object_boost=0.15).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Override video directory path.",
    )
    parser.add_argument(
        "--index-root",
        type=str,
        default=None,
        help="Override index root path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K moments/keyframes to retrieve.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load golden queries
    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = _ROOT / queries_path
    queries = load_golden_queries(queries_path)

    # Parse overrides
    temp_overrides = parse_override_dict(args.temporal)
    rerank_overrides = parse_override_dict(args.rerank)

    temp_kwargs: dict[str, Any] = {"top_k": args.top_k}
    temp_kwargs.update(temp_overrides)
    temporal_cfg = TemporalConfig(**temp_kwargs)

    rerank_kwargs: dict[str, Any] = {"enabled": True}
    rerank_kwargs.update(rerank_overrides)
    rerank_cfg = ObjectRerankConfig(**rerank_kwargs)

    # Initialize LiteService
    vdir = Path(args.video_dir) if args.video_dir else _ROOT / "data" / "videos"
    iroot = Path(args.index_root) if args.index_root else _ROOT / "artifacts" / "indexes"
    service = LiteService(video_dir=vdir, index_root=iroot)

    print(f"Loaded {len(queries)} golden queries from {queries_path.relative_to(_ROOT)}")
    print(f"TemporalConfig: {temporal_cfg}")
    if not args.no_yolo:
        print(f"ObjectRerankConfig: {rerank_cfg}")
    else:
        print("Object detector: disabled (--no-yolo)")

    eval_result = run_evaluation(
        service,
        queries,
        temporal_config=temporal_cfg,
        rerank_config=rerank_cfg,
        no_yolo=args.no_yolo,
    )

    print_summary_table(eval_result, no_yolo=args.no_yolo)
    print_detailed_table(eval_result, no_yolo=args.no_yolo)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "queries_file": str(queries_path.relative_to(_ROOT)),
            "temporal_config": {
                k: getattr(temporal_cfg, k)
                for k in temporal_cfg.__dataclass_fields__
            },
            "rerank_config": (
                {
                    k: getattr(rerank_cfg, k)
                    for k in rerank_cfg.__dataclass_fields__
                }
                if not args.no_yolo
                else None
            ),
            "no_yolo": args.no_yolo,
            "summary": eval_result["summary"],
            "details": eval_result["details"],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Benchmark output written to: {out_path.relative_to(_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
