"""T06 verified-data benchmark for E1, E2, and their two-encoder RRF.

No model is loaded and no accuracy report is written until the T05 dataset
gate passes. All three baselines search the same sampled/deduplicated index.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from evaluation.dataset import DatasetReport, validate_dataset
from evaluation.metrics import (hit_at_k, mcnemar_paired, recall_at_k,
                                reciprocal_rank, wilson_interval)
from retrieval.lite_core import EncoderSpec, LiteRetriever


MODEL_SPECS = {
    "clip": EncoderSpec("google/siglip2-base-patch16-256",
                        "6328738b83b563ff8396fdb87f8ce130a7b60d74"),
    "siglip": EncoderSpec("google/siglip-base-patch16-256-multilingual",
                          "8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6"),
}
BASELINES = {"E1": ("clip",), "E2": ("siglip",),
             "E3": ("clip", "siglip")}
ERROR_TAXONOMY = ("language_gap", "attribute_action_confusion",
                  "sampling_miss", "visual_ambiguity", "label_issue")


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "transformers", "faiss-cpu",
                 "opencv-python-headless", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class DatasetBlocked(ValueError):
    def __init__(self, report: DatasetReport):
        super().__init__("T05 verified dataset gate is not ready")
        self.report = report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(manifest_path: str | Path, labels_path: str | Path, *,
              root: str | Path, sample_every_s: float = 1.0
              ) -> tuple[dict, list[dict], DatasetReport]:
    """Fail closed before model loading, indexing, or reporting accuracy."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    report = validate_dataset(manifest, labels, root=root,
                              sample_every_s=sample_every_s)
    if not report.ready_for_accuracy:
        raise DatasetBlocked(report)
    return manifest, [row for row in labels if row["review_status"] == "verified"], report


def _relevant_ids(entry, label: Mapping) -> tuple[set[str], set[str]]:
    windows = label["time_windows"]
    sampled = {rid for rid, raw in entry.raws.items()
               if raw.video_id == label["video_id"]
               and any(lo <= raw.timestamp < hi for lo, hi in windows)}
    return sampled, sampled.intersection(entry.index.ids)


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    covered = [row for row in rows if row["index_coverage"] == "covered"]
    successes = sum(row["hit@1"] for row in rows)
    ci = wilson_interval(int(successes), n)
    return {
        "n_queries": n, "n_valid": n, "n_invalid": 0,
        "n_covered": len(covered),
        "coverage": len(covered) / n if n else 0.0,
        "hit@1": successes / n if n else 0.0,
        "hit@1_ci95": list(ci) if n else None,
        "hit@5": statistics.mean(row["hit@5"] for row in rows) if n else 0.0,
        "recall@5": statistics.mean(row["recall@5"] for row in rows) if n else 0.0,
        "mrr": statistics.mean(row["mrr"] for row in rows) if n else 0.0,
        "hit@5_covered": (statistics.mean(row["hit@5"] for row in covered)
                          if covered else None),
        "mrr_covered": (statistics.mean(row["mrr"] for row in covered)
                        if covered else None),
    }


def evaluate_entry(entry, labels: list[dict], encoders: Mapping[str, object],
                   specs: Mapping[str, EncoderSpec], *, top_k: int = 5,
                   sync: Callable[[], None] | None = None,
                   exact_threshold: int = 512) -> dict:
    """Score verified rows on one shared index; test fixtures call this offline."""
    if top_k < 5:
        raise ValueError("top_k must be at least 5 for hit@5")
    if set(encoders) != {"clip", "siglip"} or set(specs) != set(encoders):
        raise ValueError("E1/E2 require both pinned encoder slots")
    retriever = LiteRetriever(entry, specs, exact_threshold=exact_threshold)
    per_query: dict[str, list[dict]] = {name: [] for name in BASELINES}
    timing = {name: {"embedding_seconds_total": 0.0,
                     "retrieval_seconds_total": 0.0} for name in BASELINES}
    ann_checks: dict[str, list[float]] = {slot: [] for slot in encoders}

    def clocked(fn):
        if sync:
            sync()
        started = time.perf_counter()
        result = fn()
        if sync:
            sync()
        return result, time.perf_counter() - started

    for label in labels:
        query = label["query"]
        vectors: dict[str, np.ndarray] = {}
        embedding_times: dict[str, float] = {}
        for slot, encoder in encoders.items():
            vectors[slot], embedding_times[slot] = clocked(
                lambda encoder=encoder: encoder.encode_text(query))
        sampled, relevant = _relevant_ids(entry, label)
        coverage = ("no_sampled_positive" if not sampled else
                    "dedup_removed_positive" if not relevant else "covered")
        for name, slots in BASELINES.items():
            hits, retrieval_s = clocked(lambda slots=slots: retriever.search_vectors(
                {slot: vectors[slot] for slot in slots}, top_k=top_k))
            timing[name]["embedding_seconds_total"] += sum(
                embedding_times[slot] for slot in slots)
            timing[name]["retrieval_seconds_total"] += retrieval_s
            retrieved = [hit.frame_id for hit in hits]
            has_hit = bool(set(retrieved) & relevant)
            outcome = (coverage if coverage != "covered" else
                       "retrieval_hit" if has_hit else "retrieval_rank_miss")
            per_query[name].append({
                "query_id": label["query_id"], "query": query,
                "video_id": label["video_id"], "split": label["split"],
                "category": label["category"],
                "time_windows": label["time_windows"],
                "n_sampled_relevant": len(sampled),
                "n_relevant": len(relevant),
                "annotation_validity": "valid", "index_coverage": coverage,
                "retrieval_hit": has_hit, "outcome": outcome,
                "hit@1": hit_at_k(retrieved, relevant, 1),
                "hit@5": hit_at_k(retrieved, relevant, 5),
                "recall@5": recall_at_k(retrieved, relevant, 5),
                "mrr": reciprocal_rank(retrieved, relevant),
                "embedding_seconds": sum(embedding_times[slot] for slot in slots),
                "retrieval_seconds": retrieval_s,
                "predictions": [{"rank": hit.rank, "frame_id": hit.frame_id,
                                 "video_id": hit.video_id,
                                 "timestamp": hit.timestamp, "score": hit.score,
                                 "score_kind": hit.score_kind,
                                 "source_ranks": hit.source_ranks}
                                for hit in hits],
            })

        if len(entry.index.ids) > exact_threshold:
            rows = list(range(len(entry.index.ids)))
            for slot, vector in vectors.items():
                ann = entry.index.dense_search(vector, slot, 5)
                exact = entry.index.exact_dense_search(vector, slot, rows, 5)
                exact_ids = {row for row, _ in exact}
                ann_checks[slot].append(len(exact_ids.intersection(
                    row for row, _ in ann)) / max(1, len(exact_ids)))

    baselines = {name: {split: _summary([row for row in rows
                                         if row["split"] == split])
                        for split in ("dev", "test")}
                 for name, rows in per_query.items()}
    for name in BASELINES:
        total = sum(timing[name].values())
        timing[name]["queries_per_second"] = len(labels) / total if total else None
        timing[name]["embedding_seconds_per_query"] = (
            timing[name]["embedding_seconds_total"] / len(labels) if labels else None)
        timing[name]["retrieval_seconds_per_query"] = (
            timing[name]["retrieval_seconds_total"] / len(labels) if labels else None)
    paired = {split: {f"{a}_vs_{b}": mcnemar_paired(
        [row["hit@5"] for row in per_query[a] if row["split"] == split],
        [row["hit@5"] for row in per_query[b] if row["split"] == split])
        for a, b in (("E1", "E2"), ("E1", "E3"), ("E2", "E3"))}
        for split in ("dev", "test")}
    failures = [{"query_id": row["query_id"], "split": row["split"],
                 "outcome": row["outcome"], "human_review": "pending"}
                for row in per_query["E3"] if row["outcome"] != "retrieval_hit"][:10]
    return {
        "baselines": baselines, "per_query": per_query, "timing": timing,
        "ann_recall@5": {slot: (statistics.mean(values) if values else None)
                         for slot, values in ann_checks.items()},
        "ann_recall_note": ("null when exact search is used for indexes with at most "
                            f"{exact_threshold} rows"),
        "paired_hit@5": paired,
        "failure_candidates": failures,
        "error_taxonomy_for_human_review": list(ERROR_TAXONOMY),
    }


def run_benchmark(manifest_path: str | Path, labels_path: str | Path, *,
                  root: str | Path, output: str | Path, device: str = "cuda",
                  sample_every_s: float = 1.0, top_k: int = 5) -> dict:
    """Run real pinned encoders only after the verified dataset gate passes."""
    manifest, labels, gate = preflight(
        manifest_path, labels_path, root=root, sample_every_s=sample_every_s)
    if top_k < 5:
        raise ValueError("top_k must be at least 5 for hit@5")
    import torch
    from huggingface_hub import snapshot_download

    from ingestion.embed_siglip import SiglipEncoder
    from retrieval.video_engine import VideoSearchEngine

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root, output = Path(root).resolve(), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoders = {}
    model_started = time.perf_counter()
    for slot, spec in MODEL_SPECS.items():
        snapshot = snapshot_download(repo_id=spec.model_id,
                                     revision=spec.revision, max_workers=1)
        encoders[slot] = SiglipEncoder(model_name=snapshot, device=device)
    model_load_seconds = time.perf_counter() - model_started
    engine = VideoSearchEngine(
        encoder_names=tuple(spec.model_id for spec in MODEL_SPECS.values()),
        sample_every_s=sample_every_s, max_frames=None,
        dedup_threshold=0.97, query_templates=("{q}",),
        enable_ocr=False, enable_asr=False, enable_caption=False,
        embed_batch_size=8, decode_backend="cv2", use_gpu_decode=False,
        parallel_index=False)
    engine.set_encoders(list(encoders.values()))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    videos = manifest["videos"]
    started = time.perf_counter()
    entry = engine.index_dataset(
        [root / video["path"] for video in videos], output.parent / "frames",
        video_ids=[video["video_id"] for video in videos])
    if device == "cuda":
        torch.cuda.synchronize()
    index_seconds = time.perf_counter() - started
    entry.index.encoder_metadata = {
        slot: {"id": spec.model_id, "revision": spec.revision}
        for slot, spec in MODEL_SPECS.items()}
    entry.save(output.parent / "index")
    scored = evaluate_entry(
        entry, labels, encoders, MODEL_SPECS, top_k=top_k,
        sync=torch.cuda.synchronize if device == "cuda" else None)
    report = {
        "status": "MEASURED_VERIFIED_DATA", "dataset_gate": {
            "n_verified": gate.n_verified, "n_videos": gate.n_videos},
        "input_sha256": {"manifest": _sha256(Path(manifest_path)),
                         "labels": _sha256(Path(labels_path))},
        "config": {"models": {slot: {"id": spec.model_id,
                                      "revision": spec.revision}
                              for slot, spec in MODEL_SPECS.items()},
                   "source_videos": [{"video_id": video["video_id"],
                                      "sha256": video["sha256"],
                                      "split": video["split"]}
                                     for video in videos],
                   "sample_every_s": sample_every_s, "max_frames": None,
                   "hist_duplicate_threshold": 0.985,
                   "semantic_dedup_threshold": 0.97,
                   "semantic_dedup_primary": "clip/E1",
                   "text_preprocessing": "raw Vietnamese; padding=max_length; max_length=64",
                   "top_k": top_k, "device": device,
                   "retrieval": {"exact_threshold": 512,
                                 "rrf_depth": 1000},
                   "environment": {
                       "python": sys.version.split()[0],
                       "packages": _package_versions(),
                       "gpu": (torch.cuda.get_device_name()
                               if device == "cuda" else None)},
                   "command": sys.argv},
        "index": {"seconds": index_seconds, "n_sampled": entry.num_sampled,
                  "n_indexed": entry.num_indexed,
                  "model_load_seconds": model_load_seconds,
                  "frames_per_second": (entry.num_sampled / index_seconds
                                        if index_seconds else None),
                  "gpu_peak_memory_bytes": (torch.cuda.max_memory_allocated()
                                            if device == "cuda" else None),
                  "hnsw": {"m": entry.index.config.hnsw_m,
                           "ef_construction": entry.index.config.ef_construction,
                           "ef_search": entry.index.config.ef_search}},
        **scored,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    markdown = output.with_suffix(".md")
    lines = ["# T06 measured benchmark", "",
             "All baselines used one sampled/deduplicated index.", "",
             "| Split | Baseline | N | Coverage | Hit@1 | Hit@5 | MRR |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for split in ("dev", "test"):
        for name in BASELINES:
            row = scored["baselines"][name][split]
            lines.append(f"| {split} | {name} | {row['n_queries']} | "
                         f"{row['coverage']:.3f} | {row['hit@1']:.3f} | "
                         f"{row['hit@5']:.3f} | {row['mrr']:.3f} |")
    lines += ["", f"Config and per-query predictions: `{output.name}`.",
              "Failure candidates require human review before assigning causes.",
              "No hyperparameter was tuned on the test split."]
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T06 verified-data E1/E2/E3 benchmark")
    parser.add_argument("--manifest", default="data/annotations/manifest.json")
    parser.add_argument("--labels", default="data/annotations/labels.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/benchmarks/t06.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--every", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--check", action="store_true", help="only check T05 gate")
    args = parser.parse_args(argv)
    try:
        if args.check:
            _, _, gate = preflight(args.manifest, args.labels, root=args.root,
                                   sample_every_s=args.every)
            print(json.dumps({"status": "READY", "n_verified": gate.n_verified}))
        else:
            result = run_benchmark(args.manifest, args.labels, root=args.root,
                                   output=args.output, device=args.device,
                                   sample_every_s=args.every, top_k=args.top_k)
            print(json.dumps({"status": result["status"],
                              "output": args.output}, ensure_ascii=True))
    except DatasetBlocked as exc:
        print(json.dumps({"status": "DATA_BLOCKED",
                          "n_verified": exc.report.n_verified,
                          "n_videos": exc.report.n_videos,
                          "issues": [issue.__dict__ for issue in exc.report.issues]},
                         ensure_ascii=True))
        return 3
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
