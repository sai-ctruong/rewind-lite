"""Reproduce both R08 configurations sequentially on the reviewed checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scripts.eval_golden_lite import load_golden_queries, run_evaluation
from retrieval.temporal import TemporalConfig
from retrieval.object_reranker import ObjectRerankConfig
from ui.lite_app import LiteService


def compare_with_implementation(root, report):
    comparisons = {}
    for name, result in report["runs"].items():
        previous = json.loads((root / f"docs/benchmark-results/r08-filter-{name}.json").read_text(encoding="utf-8"))
        mismatches = []
        for mode, metrics in result["summary"].items():
            for key, value in metrics.items():
                if key != "median_latency_ms" and value != previous["summary"][mode][key]:
                    mismatches.append(f"{mode}.{key}")
            for row, old in zip(result["details"][mode], previous["details"][mode]):
                for key in ("id", "hit_at_1", "hit_at_3", "top1_timestamp", "top1_interval", "top1_peak_timestamp"):
                    if row.get(key) != old.get(key):
                        mismatches.append(f"{mode}.{row['id']}.{key}")
        comparisons[name] = mismatches
    report["implementation_comparison_mismatches"] = comparisons
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    queries_path = root / "evaluation/golden_lite_queries.json"
    queries = load_golden_queries(queries_path)
    service = LiteService(video_dir=root / "data/videos", index_root=root / "artifacts/indexes")
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "reviewer": "GPT-6 Astra", "review_date": "2026-11-13",
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "runs": {},
    }
    for name, threshold in (("off", None), ("on", 5.0)):
        result = run_evaluation(service, queries,
            temporal_config=TemporalConfig(dark_frame_luma_max=threshold),
            rerank_config=ObjectRerankConfig(enabled=True))
        report["runs"][name] = {"dark_frame_luma_max": threshold, **result}
        print(name, json.dumps(result["summary"]), flush=True)
        (root / "docs/benchmark-results/r-review3-golden.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compare_with_implementation(root, report)
    (root / "docs/benchmark-results/r-review3-golden.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
