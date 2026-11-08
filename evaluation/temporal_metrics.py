"""P02 temporal interval scoring; independent from Phase 1 keyframe metrics."""
from __future__ import annotations

import math
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from evaluation.dataset import validate_dataset


def _interval(value) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(t, bool) or not isinstance(t, (int, float))
                   or not math.isfinite(t) for t in value)
            or value[0] < 0 or value[0] >= value[1]):
        raise ValueError(f"invalid interval: {value!r}")
    return float(value[0]), float(value[1])


def temporal_iou(predicted: Sequence[float],
                 ground_truth: Sequence[float]) -> float:
    """Intersection over union of two nonempty half-open intervals."""
    p0, p1 = _interval(predicted)
    g0, g1 = _interval(ground_truth)
    overlap = max(0.0, min(p1, g1) - max(p0, g0))
    return overlap / ((p1 - p0) + (g1 - g0) - overlap)


def prepare_temporal_dataset(manifest: Mapping, labels: Sequence[Mapping], *,
                             root) -> dict:
    """Validate real sources with T05 rules, then map verified rows to P02 names.

    The original label objects and T06 JSON are never modified. A small valid
    fixture may be prepared even when T05's 50-query benchmark gate is false.
    """
    report = validate_dataset(manifest, labels, root=root)
    if report.issues:
        codes = ", ".join(sorted({issue.code for issue in report.issues}))
        raise ValueError(f"invalid temporal dataset: {codes}")
    sources = {video["video_id"]: video for video in manifest["videos"]}
    queries = [{
        "query_id": row["query_id"], "video_id": row["video_id"],
        "query_vi": row["query"],
        "time_windows": [list(window) for window in row["time_windows"]],
        "duration_s": sources[row["video_id"]]["duration_s"],
        "split": row["split"],
    } for row in labels if row["review_status"] == "verified"]
    return {"queries": queries, "n_evaluable": len(queries),
            "n_unreviewed": len(labels) - len(queries), "n_invalid": 0}


def _checked_query(row: Mapping, seen_ids: set[str]) -> tuple[str, str, str,
                                                                float, list[tuple[float, float]]]:
    if not isinstance(row, Mapping):
        raise ValueError("query must be an object")
    query_id, video_id, split = (row.get("query_id"), row.get("video_id"),
                                 row.get("split"))
    if not isinstance(query_id, str) or not query_id.strip() or query_id in seen_ids:
        raise ValueError("query_id missing or duplicate")
    seen_ids.add(query_id)
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id missing")
    if split not in {"dev", "test"}:
        raise ValueError("invalid split")
    duration = row.get("duration_s")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("invalid video duration")
    windows = row.get("time_windows")
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("at least one ground-truth interval required")
    checked = sorted(_interval(window) for window in windows)
    if (any(end > duration for _, end in checked)
            or any(next_lo < prev_hi for (_, prev_hi), (next_lo, _)
                   in zip(checked, checked[1:]))):
        raise ValueError("ground-truth interval out of bounds or overlapping")
    return query_id, video_id, split, float(duration), checked


def _summary(rows: list[dict]) -> dict:
    count = len(rows)
    keys = ("r@1_tiou_0.3", "r@1_tiou_0.5", "r@5_tiou_0.3",
            "r@5_tiou_0.5", "miou")
    return {"n_queries": count,
            "n_no_prediction": sum(not row["predictions"] for row in rows),
            **{key: sum(row[key] for row in rows) / count if count else 0.0
               for key in keys}}


def evaluate_temporal_predictions(queries: Sequence[Mapping],
                                  predictions: Mapping[str, Sequence[Mapping]]) -> dict:
    """Score Top-K intervals for each valid query, including unanswered ones.

    Each prediction has video_id, rank starting at 1, and t_start/t_end in
    seconds. For multiple GT windows each prediction takes its best tIoU.
    mIoU is the mean rank-1 best tIoU, with zero for no prediction.
    """
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must map query IDs to ranked intervals")
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for query in queries:
        query_id, video_id, split, duration, windows = _checked_query(query, seen_ids)
        hits = predictions.get(query_id, [])
        if not isinstance(hits, (list, tuple)):
            raise ValueError("prediction list required")
        checked_hits = []
        seen_intervals: set[tuple[float, float]] = set()
        for rank, hit in enumerate(hits, 1):
            if not isinstance(hit, Mapping) or hit.get("video_id") != video_id:
                raise ValueError("prediction video_id mismatch")
            if isinstance(hit.get("rank"), bool) or hit.get("rank") != rank:
                raise ValueError("prediction ranks must start at 1 without gaps")
            interval = _interval((hit.get("t_start"), hit.get("t_end")))
            if interval[1] > duration or interval in seen_intervals:
                raise ValueError("prediction interval out of bounds or duplicate")
            seen_intervals.add(interval)
            checked_hits.append({"rank": rank, "t_start": interval[0],
                                 "t_end": interval[1], "best_tiou": max(
                                     temporal_iou(interval, truth)
                                     for truth in windows)})
        best_at_1 = checked_hits[0]["best_tiou"] if checked_hits else 0.0
        best_at_5 = max((hit["best_tiou"] for hit in checked_hits[:5]),
                        default=0.0)
        rows.append({"query_id": query_id, "video_id": video_id,
                     "split": split, "predictions": checked_hits,
                     "miou": best_at_1,
                     "r@1_tiou_0.3": int(best_at_1 >= 0.3),
                     "r@1_tiou_0.5": int(best_at_1 >= 0.5),
                     "r@5_tiou_0.3": int(best_at_5 >= 0.3),
                     "r@5_tiou_0.5": int(best_at_5 >= 0.5)})
    unknown = set(predictions) - seen_ids
    if unknown:
        raise ValueError(f"predictions contain unknown query IDs: {sorted(unknown)}")
    return {"n_evaluable": len(rows), "per_query": rows,
            "summary": {split: _summary([row for row in rows
                                          if split == "all" or row["split"] == split])
                        for split in ("all", "dev", "test")}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and score temporal moments")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--predictions", type=Path,
                        help="JSON mapping query_id to ranked intervals")
    parser.add_argument("--output", type=Path, help="write JSON report")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        prepared = prepare_temporal_dataset(manifest, labels, root=args.root)
        queries = prepared["queries"]
        result = {"dataset": {key: value for key, value in prepared.items()
                              if key != "queries"}}
        result["dataset"]["n_dev"] = sum(q["split"] == "dev" for q in queries)
        result["dataset"]["n_test"] = sum(q["split"] == "test" for q in queries)
        if args.predictions is not None:
            predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
            result["metrics"] = evaluate_temporal_predictions(queries, predictions)
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"temporal evaluation blocked: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
