"""Dev-only corpus retrieval pool with explicit unjudged relevance.

This benchmark does not assume that a query's source video is its only match.
It gives lower/upper Hit@k bounds until every candidate needed for a metric has
been judged. The held-out test split is deliberately excluded from collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_hit(value) -> dict:
    if isinstance(value, dict):
        return {key: value[key] for key in ("rank", "video_id", "frame_id", "timestamp", "score")}
    return {key: getattr(value, key)
            for key in ("rank", "video_id", "frame_id", "timestamp", "score")}


def build_dev_pool(labels: list[dict], search, *, expected_videos: int,
                   top_k: int = 5) -> tuple[dict, dict]:
    """Run each dev query against the whole corpus and prepare review rows."""
    if not 1 <= top_k <= 50 or expected_videos < 1:
        raise ValueError("top_k and expected_videos must be positive")
    dev = [label for label in labels if label["split"] == "dev"]
    if not dev or len({label["query_id"] for label in dev}) != len(dev):
        raise ValueError("dev queries must exist and have unique IDs")
    queries = []
    review = []
    for label in dev:
        result = search(label["query"], top_k)
        if (result["total_videos"] != expected_videos or
                result["searched_videos"] != expected_videos or result["skipped"]):
            raise ValueError(f"incomplete corpus coverage for {label['query_id']}")
        hits = [_as_hit(hit) for hit in result["hits"]]
        if len(hits) > top_k or [hit["rank"] for hit in hits] != list(range(1, len(hits) + 1)):
            raise ValueError(f"invalid ranks for {label['query_id']}")
        if len({(hit["video_id"], hit["frame_id"]) for hit in hits}) != len(hits):
            raise ValueError(f"duplicate results for {label['query_id']}")
        queries.append({"query_id": label["query_id"], "query": label["query"],
                        "category": label["category"], "source_video_id": label["video_id"],
                        "source_time_windows": label["time_windows"], "hits": hits})
        judged_hits = []
        for hit in hits:
            known = (hit["video_id"] == label["video_id"] and
                     any(start <= hit["timestamp"] < end
                         for start, end in label["time_windows"]))
            judged_hits.append({"rank": hit["rank"], "video_id": hit["video_id"],
                                "frame_id": hit["frame_id"], "timestamp": hit["timestamp"],
                                "judgment": "relevant" if known else "unjudged",
                                "basis": "source_label" if known else None,
                                "reviewer": None, "reviewed_at": None})
        review.append({"query_id": label["query_id"], "hits": judged_hits})
    pool = {"schema_version": 1, "split": "dev", "search_scope": "all_indexed_videos",
            "expected_videos": expected_videos, "top_k": top_k, "queries": queries}
    judgments = {"schema_version": 1, "split": "dev", "queries": review}
    return pool, judgments


def validate_judgments(pool: dict, judgments: dict) -> None:
    if pool.get("split") != "dev" or judgments.get("split") != "dev":
        raise ValueError("only dev judgments are supported")
    predictions = pool["queries"]
    reviewed = judgments["queries"]
    if len(predictions) != len(reviewed):
        raise ValueError("judgments do not match predictions")
    for query, review in zip(predictions, reviewed):
        if query["query_id"] != review.get("query_id") or len(query["hits"]) != len(review["hits"]):
            raise ValueError("judgments do not match predictions")
        for hit, item in zip(query["hits"], review["hits"]):
            if any(hit[key] != item.get(key)
                   for key in ("rank", "video_id", "frame_id", "timestamp")):
                raise ValueError("judgment does not match prediction")
            judgment = item.get("judgment")
            if judgment not in {"relevant", "irrelevant", "unjudged"}:
                raise ValueError("invalid judgment")
            known = (hit["video_id"] == query["source_video_id"] and
                     any(start <= hit["timestamp"] < end
                         for start, end in query["source_time_windows"]))
            if known:
                if judgment != "relevant" or item.get("basis") != "source_label":
                    raise ValueError("source label positive cannot be overwritten")
            elif judgment == "unjudged":
                if item.get("basis") is not None or item.get("reviewer") is not None or item.get("reviewed_at") is not None:
                    raise ValueError("unjudged hit cannot claim review or source label")
            else:
                if (item.get("basis") != "visual_review" or
                        not item.get("reviewer") or not item.get("reviewed_at") or
                        not item.get("review_method")):
                    raise ValueError("manual judgment needs review evidence")


def score_pool(pool: dict, judgments: dict) -> dict:
    """Return honest Hit@k bounds; a point value exists once bounds agree."""
    validate_judgments(pool, judgments)
    rows = judgments["queries"]
    n = len(rows)
    if not n:
        raise ValueError("pool has no queries")
    metrics = {"n_queries": n,
               "n_manual_judgments": sum(hit["basis"] == "visual_review"
                                         for row in rows for hit in row["hits"]),
               "n_unjudged_hits": sum(hit["judgment"] == "unjudged"
                                       for row in rows for hit in row["hits"])}
    for k in (1, pool["top_k"]):
        bounds = []
        for row in rows:
            statuses = [hit["judgment"] for hit in row["hits"][:k]]
            lower = int("relevant" in statuses)
            upper = int(lower or "unjudged" in statuses)
            bounds.append((lower, upper))
        metrics[f"hit@{k}"] = {"lower": sum(low for low, _ in bounds) / n,
                               "upper": sum(high for _, high in bounds) / n,
                               "resolved": sum(low == high for low, high in bounds)}
    return metrics


def collect(args) -> tuple[dict, dict]:
    from evaluation.dataset import validate_dataset
    from ui.lite_app import LiteService

    manifest_path = Path(args.manifest)
    labels_path = Path(args.labels)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    gate = validate_dataset(manifest, labels, root=Path(args.root), sample_every_s=1.0)
    if not gate.ready_for_accuracy:
        raise ValueError(f"dataset blocked: {len(gate.issues)} issues")
    service = LiteService(video_dir=args.video_dir, index_root=args.index_root,
                          device=args.device, keyframe_min_gap_s=5.0)
    pool, judgments = build_dev_pool(labels, service.search_all,
                                     expected_videos=len(manifest["videos"]), top_k=args.top_k)
    pool["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pool["input_sha256"] = {"manifest": _sha256(manifest_path),
                            "labels": _sha256(labels_path)}
    pool["config"] = {"model_id": service.model_id, "model_revision": service.revision,
                      "sample_every_s": service.sample_every_s,
                      "max_frames": service.max_frames,
                      "keyframe_min_gap_s": service.keyframe_min_gap_s,
                      "device": args.device}
    return pool, judgments


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--root", default=".")
    collect_parser.add_argument("--manifest", default="data/annotations/corpus_v2_manifest.json")
    collect_parser.add_argument("--labels", default="data/annotations/corpus_v3_labels.json")
    collect_parser.add_argument("--video-dir", default="data/videos/corpus_v2")
    collect_parser.add_argument("--index-root", default="artifacts/corpus-v2-benchmark/indexes")
    collect_parser.add_argument("--device", default="auto")
    collect_parser.add_argument("--top-k", type=int, default=5)
    collect_parser.add_argument("--pool", default="artifacts/corpus-global-dev/pool.json")
    collect_parser.add_argument("--judgments", default="artifacts/corpus-global-dev/judgments.json")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--pool", default="artifacts/corpus-global-dev/pool.json")
    score_parser.add_argument("--judgments", default="artifacts/corpus-global-dev/judgments.json")
    score_parser.add_argument("--output", default="artifacts/corpus-global-dev/score.json")
    args = parser.parse_args(argv)
    if args.command == "collect":
        pool, judgments = collect(args)
        _write_json(Path(args.pool), pool)
        _write_json(Path(args.judgments), judgments)
        print(json.dumps(score_pool(pool, judgments), ensure_ascii=False))
    else:
        pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
        judgments = json.loads(Path(args.judgments).read_text(encoding="utf-8"))
        report = score_pool(pool, judgments)
        report["status"] = ("FULLY_JUDGED_INTERNAL" if report["n_unjudged_hits"] == 0
                            else "PARTIAL_JUDGMENTS")
        report["input_sha256"] = pool.get("input_sha256")
        report["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json(Path(args.output), report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
