"""Static validation for timestamp annotations; no video or model dependency."""
from __future__ import annotations

import argparse
import json
import math
import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path


@dataclass(frozen=True)
class ValidationIssue:
    row: int
    code: str
    detail: str


def validate_labels(
    labels: Sequence[Mapping], video_ids: set[str] | None = None,
    require_ids: bool = False,
) -> list[ValidationIssue]:
    """Validate legacy single-window or explicit multi-window annotations.

    ``video_ids`` must come from an independently verified video manifest.
    Without it, existence of a video cannot be established statically.
    """
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for row, label in enumerate(labels, 1):
        if not isinstance(label, Mapping):
            issues.append(ValidationIssue(row, "invalid_entry", "expected an object"))
            continue
        label_id = label.get("id")
        if label_id is None:
            if require_ids:
                issues.append(ValidationIssue(row, "missing_id", "id required"))
        elif not isinstance(label_id, str) or not label_id.strip():
            issues.append(ValidationIssue(row, "invalid_id", "id must be nonempty text"))
        elif label_id in seen_ids:
            issues.append(ValidationIssue(row, "duplicate_id", label_id))
        else:
            seen_ids.add(label_id)

        if not isinstance(label.get("query"), str) or not label["query"].strip():
            issues.append(ValidationIssue(row, "empty_query", "query must be nonempty text"))
        video_id = label.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            issues.append(ValidationIssue(row, "missing_video_id", "video_id required"))
        elif video_ids is not None and video_id not in video_ids:
            issues.append(ValidationIssue(row, "missing_video_id", video_id))

        has_one = "time_window" in label
        has_many = "time_windows" in label
        if has_one == has_many:
            issues.append(ValidationIssue(row, "invalid_windows", "provide exactly one window field"))
            continue
        windows = [label["time_window"]] if has_one else label["time_windows"]
        if not isinstance(windows, (list, tuple)) or not windows:
            issues.append(ValidationIssue(row, "invalid_windows", "at least one window required"))
            continue
        valid: list[tuple[float, float]] = []
        for window in windows:
            if (not isinstance(window, (list, tuple)) or len(window) != 2 or
                    any(isinstance(t, bool) or not isinstance(t, (int, float)) or
                        not math.isfinite(t) for t in window) or
                    window[0] < 0 or window[0] >= window[1]):
                issues.append(ValidationIssue(row, "invalid_interval", str(window)))
            else:
                valid.append((float(window[0]), float(window[1])))
        valid.sort()
        if any(next_lo < prev_hi for (_, prev_hi), (next_lo, _) in zip(valid, valid[1:])):
            issues.append(ValidationIssue(row, "overlapping_windows", "merge overlapping windows"))
    return issues


@dataclass(frozen=True)
class DatasetReport:
    issues: list[ValidationIssue]
    n_videos: int
    n_labels: int
    n_verified: int
    n_nominal_covered: int
    category_counts: dict[str, int]
    ready_for_accuracy: bool


_CATEGORIES = {"object", "color", "action", "count", "context", "short_event"}
_SPLITS = {"dev", "test"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.findall(r"\w+", text, re.UNICODE))


def _sampled_in_window(windows: Sequence[Sequence[float]],
                       duration: float, every_s: float) -> bool:
    count = math.ceil(duration / every_s)
    return any(lo <= i * every_s < hi
               for i in range(count) for lo, hi in windows)


def validate_dataset(
    manifest: Mapping, labels: Sequence[Mapping], *, root: str | Path,
    sample_every_s: float = 1.0,
) -> DatasetReport:
    """Validate local evidence; metadata cannot prove semantic label correctness.

    The reported sampling coverage is a *nominal uniform grid* before histogram
    filtering and semantic deduplication. It is not index coverage or accuracy.
    """
    if not math.isfinite(sample_every_s) or sample_every_s <= 0:
        raise ValueError("sample_every_s must be positive and finite")
    root = Path(root).resolve()
    issues: list[ValidationIssue] = []
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        issues.append(ValidationIssue(0, "manifest_version", "expected schema_version=1"))
        videos = []
    else:
        videos = manifest.get("videos", [])
        if not isinstance(videos, list):
            issues.append(ValidationIssue(0, "invalid_videos", "videos must be an array"))
            videos = []

    sources: dict[str, Mapping] = {}
    source_ok: dict[str, bool] = {}
    groups: dict[str, str] = {}
    for row, video in enumerate(videos, 1):
        before = len(issues)
        if not isinstance(video, Mapping):
            issues.append(ValidationIssue(row, "invalid_video", "expected an object"))
            continue
        video_id = video.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            issues.append(ValidationIssue(row, "missing_video_id", "video_id required"))
            continue
        if video_id in sources:
            issues.append(ValidationIssue(row, "duplicate_video_id", video_id))
            continue
        sources[video_id] = video
        split = video.get("split")
        group = video.get("group_id")
        if not isinstance(split, str) or split not in _SPLITS:
            issues.append(ValidationIssue(row, "invalid_split", str(split)))
        if not isinstance(group, str) or not group.strip():
            issues.append(ValidationIssue(row, "missing_group_id", video_id))
        elif group in groups and groups[group] != split:
            issues.append(ValidationIssue(row, "split_group_leakage", group))
        else:
            groups[group] = split
        if not isinstance(video.get("source"), str) or not video["source"].strip():
            issues.append(ValidationIssue(row, "missing_source", video_id))
        rights = video.get("rights")
        if (not isinstance(rights, Mapping) or rights.get("status") != "confirmed"
                or not isinstance(rights.get("basis"), str)
                or not rights["basis"].strip()):
            issues.append(ValidationIssue(row, "rights_unconfirmed", video_id))

        relative = video.get("path")
        path = None
        if not isinstance(relative, str) or not relative.strip():
            issues.append(ValidationIssue(row, "invalid_path", video_id))
        else:
            candidate = Path(relative)
            path = (root / candidate).resolve()
            if candidate.is_absolute() or not path.is_relative_to(root):
                issues.append(ValidationIssue(row, "invalid_path", relative))
                path = None
            elif not path.is_file():
                issues.append(ValidationIssue(row, "missing_video", relative))
                path = None
        sha = video.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            issues.append(ValidationIssue(row, "invalid_checksum", video_id))
        elif path is not None and _sha256_file(path) != sha:
            issues.append(ValidationIssue(row, "checksum_mismatch", video_id))

        duration = video.get("duration_s")
        fps = video.get("fps")
        frames = video.get("frame_count")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0
                or isinstance(fps, bool) or not isinstance(fps, (int, float))
                or not math.isfinite(fps) or fps <= 0
                or isinstance(frames, bool) or not isinstance(frames, int) or frames < 1):
            issues.append(ValidationIssue(row, "invalid_video_metadata", video_id))
        elif path is not None:
            import cv2

            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    issues.append(ValidationIssue(row, "unreadable_video", video_id))
                else:
                    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if (actual_fps <= 0 or actual_frames <= 0
                            or actual_frames != frames or abs(actual_fps - fps) > 0.05
                            or abs(actual_frames / actual_fps - duration) > 0.05):
                        issues.append(ValidationIssue(row, "duration_mismatch", video_id))
            finally:
                cap.release()
        source_ok[video_id] = len(issues) == before

    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        issues.append(ValidationIssue(0, "invalid_labels", "labels must be an array"))
        labels = []
    seen_ids: set[str] = set()
    categories: Counter[str] = Counter()
    n_verified = 0
    n_nominal_covered = 0
    grouped_queries: list[tuple[int, str, str, str]] = []
    splits_with_verified: set[str] = set()
    for row, label in enumerate(labels, 1):
        before = len(issues)
        if not isinstance(label, Mapping):
            issues.append(ValidationIssue(row, "invalid_entry", "expected an object"))
            continue
        query_id = label.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            issues.append(ValidationIssue(row, "missing_query_id", str(query_id)))
        elif query_id in seen_ids:
            issues.append(ValidationIssue(row, "duplicate_query_id", query_id))
        else:
            seen_ids.add(query_id)
        query = label.get("query")
        if not isinstance(query, str) or not query.strip():
            issues.append(ValidationIssue(row, "empty_query", str(query)))
        if label.get("language") != "vi":
            issues.append(ValidationIssue(row, "invalid_language", "language must be vi"))
        video_id = label.get("video_id")
        source = sources.get(video_id) if isinstance(video_id, str) else None
        if source is None:
            issues.append(ValidationIssue(row, "missing_video_id", str(video_id)))
        category = label.get("category")
        if not isinstance(category, str) or category not in _CATEGORIES:
            issues.append(ValidationIssue(row, "invalid_category", str(category)))
        else:
            categories[category] += 1
        split = label.get("split")
        if source is not None and split != source.get("split"):
            issues.append(ValidationIssue(row, "split_mismatch", str(split)))
        group = label.get("paraphrase_group")
        if not isinstance(group, str) or not group.strip():
            issues.append(ValidationIssue(row, "missing_paraphrase_group", str(group)))
        elif isinstance(query, str) and isinstance(split, str) and split in _SPLITS:
            grouped_queries.append((row, split, group, _normalized_query(query)))

        windows = label.get("time_windows")
        if (not isinstance(windows, list) or not windows
                or "time_window" in label):
            issues.append(ValidationIssue(row, "invalid_windows", str(windows)))
            valid_windows = []
        else:
            valid_windows = []
            for window in windows:
                if (not isinstance(window, (list, tuple)) or len(window) != 2
                        or any(isinstance(t, bool) or not isinstance(t, (int, float))
                               or not math.isfinite(t) for t in window)
                        or window[0] < 0 or window[0] >= window[1]):
                    issues.append(ValidationIssue(row, "invalid_interval", str(window)))
                else:
                    valid_windows.append((float(window[0]), float(window[1])))
            valid_windows.sort()
            if any(b[0] < a[1] for a, b in zip(valid_windows, valid_windows[1:])):
                issues.append(ValidationIssue(row, "overlapping_windows", str(windows)))
            if source is not None and isinstance(source.get("duration_s"), (int, float)):
                if any(hi > source["duration_s"] + 1e-3 for _, hi in valid_windows):
                    issues.append(ValidationIssue(row, "window_out_of_bounds", video_id))

        status = label.get("review_status")
        if not isinstance(status, str) or status not in {"draft", "verified", "rejected"}:
            issues.append(ValidationIssue(row, "invalid_review_status", str(status)))
        if status == "verified":
            if not isinstance(label.get("annotator"), str) or not label["annotator"].strip():
                issues.append(ValidationIssue(row, "missing_annotator", str(query_id)))
            if not isinstance(label.get("reviewer"), str) or not label["reviewer"].strip():
                issues.append(ValidationIssue(row, "missing_reviewer", str(query_id)))
            try:
                date.fromisoformat(label.get("reviewed_at", ""))
            except (TypeError, ValueError):
                issues.append(ValidationIssue(row, "invalid_review_date", str(query_id)))
        if (valid_windows and source is not None and source_ok.get(video_id)
                and len(issues) == before):
            if _sampled_in_window(valid_windows, source["duration_s"], sample_every_s):
                n_nominal_covered += 1
            if status == "verified":
                n_verified += 1
                splits_with_verified.add(split)

    for i, (row, split, group, query) in enumerate(grouped_queries):
        for _, other_split, other_group, other_query in grouped_queries[:i]:
            if split == other_split:
                continue
            if (group == other_group or query == other_query
                    or SequenceMatcher(None, query, other_query).ratio() >= 0.85):
                issues.append(ValidationIssue(row, "paraphrase_leakage", group))
                break

    return DatasetReport(
        issues=issues, n_videos=len(sources), n_labels=len(labels),
        n_verified=n_verified, n_nominal_covered=n_nominal_covered,
        category_counts=dict(categories),
        ready_for_accuracy=(n_verified >= 50 and not issues
                            and splits_with_verified == _SPLITS),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate T05 video and annotation evidence")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--every", type=float, default=1.0,
                        help="nominal uniform sampling interval in seconds")
    parser.add_argument("--require-ready", action="store_true",
                        help="exit 3 until at least 50 verified labels and split gate pass")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        report = validate_dataset(manifest, labels, root=args.root,
                                  sample_every_s=args.every)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(asdict(report), ensure_ascii=True, indent=2))
    if report.issues:
        return 2
    return 3 if args.require_ready and not report.ready_for_accuracy else 0


if __name__ == "__main__":
    raise SystemExit(main())
