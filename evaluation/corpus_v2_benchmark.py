"""Per-video corpus v2 benchmark using the same Lite index/search path as the web app."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from evaluation.absent_dataset import validate_absent_queries
from evaluation.dataset import validate_dataset
from evaluation.metrics import wilson_interval
from evaluation.temporal_metrics import evaluate_temporal_predictions


def score_keyframes(entry, label: dict, hits: list) -> dict:
    """Score ranked keyframes against one video's half-open positive windows."""
    windows = label["time_windows"]
    video_id = label["video_id"]
    sampled = {frame_id for frame_id, raw in entry.raws.items()
               if raw.video_id == video_id
               and any(start <= raw.timestamp < end for start, end in windows)}
    relevant = sampled.intersection(entry.index.ids)
    ranked = [hit.frame_id for hit in hits]
    ranks = [rank for rank, frame_id in enumerate(ranked, 1) if frame_id in relevant]
    return {
        "n_sampled_relevant": len(sampled),
        "n_relevant": len(relevant),
        "coverage": ("covered" if relevant else
                     "dedup_removed_positive" if sampled else "no_sampled_positive"),
        "hit@1": int(bool(ranks and ranks[0] == 1)),
        "hit@5": int(any(rank <= 5 for rank in ranks)),
        "mrr": 1.0 / ranks[0] if ranks else 0.0,
        "top_score": float(hits[0].score) if hits else -1.0,
    }


def _balanced_accuracy(rows: list[dict], threshold: float) -> tuple[float, float, float]:
    present = [r for r in rows if r["expected_presence"] == "present"]
    absent = [r for r in rows if r["expected_presence"] == "absent"]
    if not present or not absent:
        raise ValueError("dev needs both present and absent queries")
    sensitivity = sum(r["top_score"] >= threshold for r in present) / len(present)
    specificity = sum(r["top_score"] < threshold for r in absent) / len(absent)
    return (sensitivity + specificity) / 2, sensitivity, specificity


def calibrate_threshold(rows: list[dict]) -> dict:
    """Choose one cosine threshold on dev only, with a conservative tie break."""
    dev = [row for row in rows if row["split"] == "dev"]
    if not dev:
        raise ValueError("dev needs both present and absent queries")
    scores = [float(row["top_score"]) for row in dev]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("top scores must be finite")
    candidates = sorted(set(scores) | {max(scores) + 1e-6})
    evaluated = [(threshold, *_balanced_accuracy(dev, threshold))
                 for threshold in candidates]
    threshold, balanced, sensitivity, specificity = max(
        evaluated, key=lambda item: (item[1], item[0]))
    return {"threshold": threshold, "balanced_accuracy": balanced,
            "sensitivity": sensitivity, "specificity": specificity,
            "selection": "maximize dev balanced accuracy; on ties choose higher threshold"}


def summarize_presence(rows: list[dict], threshold: float) -> dict:
    present = [row for row in rows if row["expected_presence"] == "present"]
    absent = [row for row in rows if row["expected_presence"] == "absent"]
    tp = sum(row["top_score"] >= threshold for row in present)
    tn = sum(row["top_score"] < threshold for row in absent)
    tpr = tp / len(present) if present else None
    tnr = tn / len(absent) if absent else None
    return {"n_present": len(present), "n_absent": len(absent),
            "tp": tp, "fn": len(present) - tp,
            "tn": tn, "fp": len(absent) - tn,
            "sensitivity": tpr, "specificity": tnr,
            "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None}


def summarize_keyframes(rows: list[dict]) -> dict:
    n = len(rows)
    hit1 = sum(row["hit@1"] for row in rows)
    return {"n_queries": n,
            "n_covered": sum(row["coverage"] == "covered" for row in rows),
            "coverage": sum(row["coverage"] == "covered" for row in rows) / n if n else 0.0,
            "hit@1": hit1 / n if n else 0.0,
            "hit@1_ci95_query_level": list(wilson_interval(hit1, n)) if n else None,
            "hit@5": statistics.mean(row["hit@5"] for row in rows) if n else 0.0,
            "mrr": statistics.mean(row["mrr"] for row in rows) if n else 0.0}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_timestamp() -> str:
    """Use an unambiguous timestamp without requiring IANA tzdata on Windows."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def select_queries(labels: list[dict], absent: list[dict], *,
                   split: str, exclude_query_ids: set[str] | None = None
                   ) -> tuple[list[dict], list[dict]]:
    """Keep the held-out test entirely out of exploratory dev runs."""
    if split not in {"all", "dev"}:
        raise ValueError("split must be all or dev")
    excluded = exclude_query_ids or set()
    if excluded and split != "dev":
        raise ValueError("query exclusions are allowed only on dev")
    if excluded:
        dev_ids = {row["query_id"] for row in labels + absent
                   if row["split"] == "dev"}
        unknown = excluded - dev_ids
        if unknown:
            raise ValueError(f"unknown dev query IDs for exclusion: {sorted(unknown)}")
    if split == "all":
        return labels, absent
    return ([row for row in labels if row["split"] == "dev"
             and row["query_id"] not in excluded],
            [row for row in absent if row["split"] == "dev"
             and row["query_id"] not in excluded])


def _prediction(hit) -> dict:
    return {"rank": hit.rank, "frame_id": hit.frame_id,
            "video_id": hit.video_id, "timestamp": hit.timestamp,
            "score": hit.score}


def run_benchmark(manifest_path: Path, labels_path: Path, absent_path: Path,
                  *, root: Path, output: Path, device: str = "cuda",
                  model_id: str | None = None, revision: str | None = None,
                  split: str = "all", index_root: Path | None = None,
                  exclude_query_ids: set[str] | None = None,
                  keyframe_min_gap_s: float = 0.0) -> dict:
    """Index only annotated videos, then score each query within its source video."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    absent = json.loads(absent_path.read_text(encoding="utf-8"))
    gate = validate_dataset(manifest, labels, root=root, sample_every_s=1.0)
    absence_issues = validate_absent_queries(manifest, absent, labels)
    if not gate.ready_for_accuracy or absence_issues:
        raise ValueError(f"dataset blocked: {len(gate.issues)} positive and "
                         f"{len(absence_issues)} absent issues; ready={gate.ready_for_accuracy}")
    from scripts.lite_search import DEFAULT_MODEL, DEFAULT_REVISION
    from ui.lite_app import LiteService

    if (model_id is None) != (revision is None):
        raise ValueError("model_id and revision must be specified together")
    model_id = model_id or DEFAULT_MODEL
    revision = revision or DEFAULT_REVISION
    labels_run, absent_run = select_queries(
        labels, absent, split=split, exclude_query_ids=exclude_query_ids)
    sources = {video["video_id"]: video for video in manifest["videos"]}
    selected = sorted({row["video_id"] for row in labels_run + absent_run},
                      key=lambda video_id: int(video_id.split("_")[-1]))
    index_root = index_root or root / "artifacts/corpus-v2-benchmark/indexes"
    service = LiteService(video_dir=root / "data/videos/corpus_v2",
                          index_root=index_root, device=device,
                          model_id=model_id, revision=revision,
                          sample_every_s=0.5, max_frames=512,
                          hist_duplicate_threshold=1.0, dedup_threshold=1.0,
                          keyframe_min_gap_s=keyframe_min_gap_s)
    positive_rows: list[dict] = []
    absent_rows: list[dict] = []
    temporal_predictions: dict[str, list[dict]] = {}
    video_runs: list[dict] = []
    started = time.perf_counter()
    for position, video_id in enumerate(selected, 1):
        video_started = time.perf_counter()
        indexed = service.index(video_id)
        entry = service.entries[video_id]
        source = sources[video_id]
        for label in [row for row in labels_run if row["video_id"] == video_id]:
            query_started = time.perf_counter()
            hits = service.search(video_id, label["query"], top_k=5)
            elapsed = time.perf_counter() - query_started
            scored = score_keyframes(entry, label, hits)
            positive_rows.append({"query_id": label["query_id"], "query": label["query"],
                                  "video_id": video_id, "group_id": source["group_id"],
                                  "split": label["split"], "category": label["category"],
                                  "expected_presence": "present",
                                  "time_windows": label["time_windows"],
                                  "query_seconds": elapsed,
                                  "predictions": [_prediction(hit) for hit in hits], **scored})
            temporal = service.search_temporal(video_id, label["query"], top_k=5,
                                               use_detector=False)
            temporal_predictions[label["query_id"]] = [moment.to_dict()
                                                         for moment in temporal["moments"]]
        for label in [row for row in absent_run if row["video_id"] == video_id]:
            query_started = time.perf_counter()
            hits = service.search(video_id, label["query"], top_k=5)
            absent_rows.append({"query_id": label["query_id"], "query": label["query"],
                                "video_id": video_id, "group_id": source["group_id"],
                                "split": label["split"], "category": label["category"],
                                "expected_presence": "absent",
                                "query_seconds": time.perf_counter() - query_started,
                                "top_score": float(hits[0].score) if hits else -1.0,
                                "predictions": [_prediction(hit) for hit in hits]})
        video_runs.append({"video_id": video_id, "split": source["split"],
                           "group_id": source["group_id"], "cached": indexed["cached"],
                           "n_sampled": entry.num_sampled, "n_indexed": entry.num_indexed,
                           "elapsed_seconds": time.perf_counter() - video_started})
        print(f"[{position}/{len(selected)}] {video_id}: {entry.num_indexed} frames, "
              f"{video_runs[-1]['elapsed_seconds']:.1f}s", flush=True)
        service.entries.pop(video_id, None)
        service.retrievers.pop(video_id, None)
        service.source_signatures.pop(video_id, None)
        service.frame_lumas.pop(video_id, None)
        gc.collect()

    rows = positive_rows + absent_rows
    threshold = calibrate_threshold(rows)
    presence = {split: summarize_presence([row for row in rows if row["split"] == split],
                                          threshold["threshold"])
                for split in ("dev", "test")}
    kf = {split: summarize_keyframes([row for row in positive_rows if row["split"] == split])
          for split in ("dev", "test")}
    group_results = {group: summarize_keyframes([row for row in positive_rows
                                                  if row["group_id"] == group])
                     for group in sorted({row["group_id"] for row in positive_rows})}
    temporal_queries = [{"query_id": label["query_id"],
                         "video_id": label["video_id"], "split": label["split"],
                         "duration_s": sources[label["video_id"]]["duration_s"],
                         "time_windows": label["time_windows"]}
                        for label in labels_run]
    temporal_metrics = evaluate_temporal_predictions(temporal_queries, temporal_predictions)
    report = {
        "status": "EXPLORATORY_DEV" if split == "dev" else "MEASURED_INTERNAL_LABELS",
        "measured_at": report_timestamp(),
        "input_sha256": {"manifest": _sha256(manifest_path),
                         "labels": _sha256(labels_path), "absent": _sha256(absent_path)},
        "config": {"model_id": model_id, "model_revision": revision,
                   "device": device, "sample_every_s": 0.5, "max_frames_per_video": 512,
                   "split": split,
                   "excluded_query_ids": sorted(exclude_query_ids or []),
                   "keyframe_min_gap_s": keyframe_min_gap_s,
                   "hist_duplicate_threshold": 1.0, "semantic_dedup_threshold": 1.0,
                   "search_scope": "one source video per query",
                   "keyframe_retrieval": (
                       "LiteService.search, exact cosine for <=512 frames; "
                       f"minimum gap {keyframe_min_gap_s}s with backfill"),
                   "temporal_retrieval": "LiteService.search_temporal, detector off",
                   "presence_rule": "accept when top-1 keyframe cosine >= dev threshold",
                   "top_k": 5, "command": sys.argv},
        "n_source_videos": gate.n_videos, "n_annotated_videos": len(selected),
        "n_positive": len(positive_rows), "n_absent": len(absent_rows),
        "index": video_runs, "index_and_query_seconds": time.perf_counter() - started,
        "calibration_dev": threshold, "presence": presence,
        "keyframe": {"by_split": kf, "by_group": group_results,
                     "per_query": positive_rows},
        "absent_per_query": absent_rows,
        "temporal": temporal_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def cli_summary(report: dict, *, split: str) -> dict:
    measured = "dev" if split == "dev" else "test"
    return {f"keyframe_{measured}": report["keyframe"]["by_split"][measured],
            f"temporal_{measured}": report["temporal"]["summary"][measured],
            f"presence_{measured}": report["presence"][measured]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the corpus v2 web baseline")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/annotations/corpus_v2_manifest.json"))
    parser.add_argument("--labels", type=Path,
                        default=Path("data/annotations/corpus_v2_labels.json"))
    parser.add_argument("--absent", type=Path,
                        default=Path("data/annotations/corpus_v2_absent.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/corpus-v2-benchmark/report.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--split", choices=("all", "dev"), default="all")
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--exclude-query-id", action="append", default=[])
    parser.add_argument("--keyframe-min-gap-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.manifest, args.labels, args.absent,
                               root=args.root.resolve(), output=args.output,
                               device=args.device, model_id=args.model_id,
                               revision=args.revision, split=args.split,
                               index_root=args.index_root,
                               exclude_query_ids=set(args.exclude_query_id),
                               keyframe_min_gap_s=args.keyframe_min_gap_s)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      **cli_summary(report, split=args.split)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
