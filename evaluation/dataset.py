"""Static validation for timestamp annotations; no video or model dependency."""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence


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
