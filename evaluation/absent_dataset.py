"""Validate absent-scene queries without mixing them into temporal scoring."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from evaluation.dataset import ValidationIssue, _CATEGORIES, _normalized_query


def validate_absent_queries(
    manifest: Mapping, absent: Sequence[Mapping], positives: Sequence[Mapping],
) -> list[ValidationIssue]:
    """Check absent-query structure and link to the same video/split manifest.

    Absence itself requires visual review of the source; static checks cannot prove it.
    """
    issues: list[ValidationIssue] = []
    videos = {v["video_id"]: v for v in manifest.get("videos", [])
              if isinstance(v, Mapping) and isinstance(v.get("video_id"), str)}
    positive_ids = {p.get("query_id") for p in positives if isinstance(p, Mapping)}
    positive_queries = {(p.get("video_id"), _normalized_query(p["query"]))
                        for p in positives if isinstance(p, Mapping)
                        and isinstance(p.get("query"), str)}
    seen = set(positive_ids)
    seen_queries: set[tuple[str, str]] = set()
    for row_number, row in enumerate(absent, 1):
        if not isinstance(row, Mapping):
            issues.append(ValidationIssue(row_number, "invalid_entry", "expected object"))
            continue
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            issues.append(ValidationIssue(row_number, "missing_query_id", str(query_id)))
        elif query_id in seen:
            issues.append(ValidationIssue(row_number, "duplicate_query_id", query_id))
        else:
            seen.add(query_id)
        video_id = row.get("video_id")
        video = videos.get(video_id) if isinstance(video_id, str) else None
        if video is None:
            issues.append(ValidationIssue(row_number, "missing_video_id", str(video_id)))
        elif row.get("split") != video.get("split"):
            issues.append(ValidationIssue(row_number, "split_mismatch", str(video_id)))
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            issues.append(ValidationIssue(row_number, "empty_query", str(query)))
        elif isinstance(video_id, str):
            key = (video_id, _normalized_query(query))
            if key in positive_queries:
                issues.append(ValidationIssue(row_number, "positive_collision", query))
            if key in seen_queries:
                issues.append(ValidationIssue(row_number, "duplicate_query", query))
            seen_queries.add(key)
        if row.get("expected_presence") != "absent":
            issues.append(ValidationIssue(row_number, "invalid_expected_presence", str(row.get("expected_presence"))))
        if row.get("time_windows") != [] or "time_window" in row:
            issues.append(ValidationIssue(row_number, "invalid_windows", "absent query requires []"))
        if row.get("language") != "vi":
            issues.append(ValidationIssue(row_number, "invalid_language", "language must be vi"))
        if row.get("category") not in _CATEGORIES:
            issues.append(ValidationIssue(row_number, "invalid_category", str(row.get("category"))))
        if not isinstance(row.get("paraphrase_group"), str) or not row["paraphrase_group"].strip():
            issues.append(ValidationIssue(row_number, "missing_paraphrase_group", str(query_id)))
        status = row.get("review_status")
        if status not in {"draft", "verified", "rejected"}:
            issues.append(ValidationIssue(row_number, "invalid_review_status", str(status)))
        if status == "verified":
            for field in ("annotator", "reviewer"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    issues.append(ValidationIssue(row_number, f"missing_{field}", str(query_id)))
            try:
                date.fromisoformat(row.get("reviewed_at", ""))
            except (TypeError, ValueError):
                issues.append(ValidationIssue(row_number, "invalid_review_date", str(query_id)))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local absent-scene query labels")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--absent", required=True)
    parser.add_argument("--positives", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        absent = json.loads(Path(args.absent).read_text(encoding="utf-8"))
        positives = json.loads(Path(args.positives).read_text(encoding="utf-8"))
        issues = validate_absent_queries(manifest, absent, positives)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"n_absent": len(absent),
                      "n_verified": sum(x.get("review_status") == "verified" for x in absent),
                      "issues": [vars(x) for x in issues]}, ensure_ascii=False, indent=2))
    return 2 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
