"""P03 temporal-only moment localization baseline over Phase 1 frame embeddings.

This module converts per-frame similarity scores and timestamps from a
``VideoIndexEntry`` into ranked ``[t_start, t_end)`` intervals without loading
or depending on any object detector. The Phase 1 keyframe Top-K path in
``retrieval.lite_core`` remains untouched.

Sampling limitation note:
    Frame-level similarity over sparsely sampled keyframes can only approximate
    event boundaries up to the underlying frame sampling and deduplication
    granularity. Returned ``[t_start, t_end)`` intervals are ranked temporal
    proposals and must not be interpreted as exact frame-accurate action
    boundaries or calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from evaluation.temporal_metrics import temporal_iou
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.lite_core import Encoder, EncoderSpec

if TYPE_CHECKING:
    from retrieval.video_engine import VideoIndexEntry


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class FrameScore:
    """Single searchable keyframe score at a known video timestamp."""

    frame_id: str
    video_id: str
    timestamp: float
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a nonempty string")
        if not _is_finite_number(self.timestamp) or float(self.timestamp) < 0.0:
            raise ValueError(f"timestamp must be finite and >= 0: {self.timestamp!r}")
        if not _is_finite_number(self.score):
            raise ValueError(f"score must be finite: {self.score!r}")
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class TemporalMoment:
    """Ranked temporal interval contract frozen in P01 (`docs/phase2-p01-handoff.md`).

    Intervals use the half-open convention ``[t_start, t_end)`` in seconds, with
    ``0 <= t_start < t_end <= video_duration``. ``score`` is a relative ranking
    value, never a calibrated probability. ``detections=None`` indicates that no
    object detector was run.
    """

    video_id: str
    t_start: float
    t_end: float
    rank: int
    score: float
    evidence_frame_ids: tuple[str, ...]
    score_type: str = "temporal_relevance"
    detections: tuple[dict, ...] | None = None
    peak_frame_id: str | None = None
    peak_timestamp: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a nonempty string")
        if not _is_finite_number(self.t_start) or not _is_finite_number(self.t_end):
            raise ValueError("t_start and t_end must be finite numbers")
        start = float(self.t_start)
        end = float(self.t_end)
        if start < 0.0 or start >= end:
            raise ValueError(
                f"invalid interval [{start}, {end}): require 0 <= t_start < t_end"
            )
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {self.rank!r}")
        if not _is_finite_number(self.score):
            raise ValueError(f"score must be finite, got {self.score!r}")
        if not isinstance(self.evidence_frame_ids, (tuple, list)):
            raise ValueError("evidence_frame_ids must be a tuple or list of frame IDs")
        evidence = tuple(self.evidence_frame_ids)
        if not evidence or any(
            not isinstance(fid, str) or not fid.strip() for fid in evidence
        ):
            raise ValueError("evidence_frame_ids must contain at least one valid frame_id")
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence_frame_ids must not contain duplicates")
        if not isinstance(self.score_type, str) or not self.score_type.strip():
            raise ValueError("score_type must be a nonempty string")
        if self.detections is not None:
            if not isinstance(self.detections, (tuple, list)) or any(
                not isinstance(item, dict) for item in self.detections
            ):
                raise ValueError("detections must be None or a sequence of dicts")
            object.__setattr__(self, "detections", tuple(self.detections))
        if self.peak_frame_id is not None and (
            not isinstance(self.peak_frame_id, str) or not self.peak_frame_id.strip()
        ):
            raise ValueError("peak_frame_id must be None or a nonempty string")
        if self.peak_timestamp is not None and (
            not _is_finite_number(self.peak_timestamp) or float(self.peak_timestamp) < 0.0
        ):
            raise ValueError(f"peak_timestamp must be None or finite >= 0, got {self.peak_timestamp!r}")
        if self.peak_timestamp is not None:
            object.__setattr__(self, "peak_timestamp", float(self.peak_timestamp))
        object.__setattr__(self, "t_start", start)
        object.__setattr__(self, "t_end", end)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "evidence_frame_ids", evidence)

    def to_dict(self) -> dict:
        """Serialize to a JSON-ready dict compatible with ``evaluation.temporal_metrics``."""
        return {
            "video_id": self.video_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "rank": self.rank,
            "score": self.score,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "score_type": self.score_type,
            "peak_frame_id": self.peak_frame_id,
            "peak_timestamp": self.peak_timestamp,
            "detections": (
                [dict(det) for det in self.detections]
                if self.detections is not None
                else None
            ),
        }


@dataclass(frozen=True)
class TemporalConfig:
    """Configurable hyperparameters for P03 temporal segment proposal.

    All thresholds are explicit and must only be tuned on the ``dev`` split.
    """

    smoothing_window: int = 3
    score_quantile: float = 0.70
    min_score_floor: float | None = None
    include_local_peaks: bool = True
    max_merge_gap_s: float = 2.5
    default_frame_span_s: float = 1.0
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0
    nms_tiou_threshold: float | None = 0.5
    relative_peak_threshold: float = 0.50
    dark_frame_luma_max: float | None = None
    top_k: int = 5
    boundary_tolerance_s: float = 1e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.smoothing_window, bool)
            or not isinstance(self.smoothing_window, int)
            or self.smoothing_window < 1
        ):
            raise ValueError("smoothing_window must be an integer >= 1")
        if (
            not _is_finite_number(self.score_quantile)
            or not (0.0 <= float(self.score_quantile) <= 1.0)
        ):
            raise ValueError("score_quantile must be in [0.0, 1.0]")
        if self.min_score_floor is not None and not _is_finite_number(self.min_score_floor):
            raise ValueError("min_score_floor must be None or a finite number")
        if not isinstance(self.include_local_peaks, bool):
            raise ValueError("include_local_peaks must be a bool")
        if not _is_finite_number(self.max_merge_gap_s) or float(self.max_merge_gap_s) < 0.0:
            raise ValueError("max_merge_gap_s must be finite and >= 0")
        if (
            not _is_finite_number(self.default_frame_span_s)
            or float(self.default_frame_span_s) <= 0.0
        ):
            raise ValueError("default_frame_span_s must be finite and > 0")
        if not _is_finite_number(self.min_duration_s) or float(self.min_duration_s) <= 0.0:
            raise ValueError("min_duration_s must be finite and > 0")
        if (
            not _is_finite_number(self.max_duration_s)
            or float(self.max_duration_s) < float(self.min_duration_s)
        ):
            raise ValueError("max_duration_s must be finite and >= min_duration_s")
        if self.nms_tiou_threshold is not None:
            if (
                not _is_finite_number(self.nms_tiou_threshold)
                or not (0.0 < float(self.nms_tiou_threshold) <= 1.0)
            ):
                raise ValueError("nms_tiou_threshold must be None or in (0.0, 1.0]")
        if (
            not _is_finite_number(self.relative_peak_threshold)
            or not (0.0 <= float(self.relative_peak_threshold) <= 1.0)
        ):
            raise ValueError("relative_peak_threshold must be in [0.0, 1.0]")
        if self.dark_frame_luma_max is not None:
            if (
                not _is_finite_number(self.dark_frame_luma_max)
                or not (0.0 <= float(self.dark_frame_luma_max) <= 255.0)
            ):
                raise ValueError("dark_frame_luma_max must be None or in [0.0, 255.0]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be an integer >= 1")
        if (
            not _is_finite_number(self.boundary_tolerance_s)
            or float(self.boundary_tolerance_s) < 0.0
        ):
            raise ValueError("boundary_tolerance_s must be finite and >= 0")


def _resolve_excluded_dark_frame_ids(
    config: TemporalConfig,
    *,
    dark_frame_ids: Sequence[str] | set[str] | frozenset[str] | None = None,
    frame_lumas: Mapping[str, float] | None = None,
) -> set[str]:
    """Resolve frame IDs to exclude when ``config.dark_frame_luma_max`` is enabled."""
    if config.dark_frame_luma_max is None:
        return set()
    thresh = float(config.dark_frame_luma_max)
    excluded: set[str] = set()
    if dark_frame_ids is not None:
        excluded.update(str(fid) for fid in dark_frame_ids)
    if frame_lumas is not None:
        for fid, luma in frame_lumas.items():
            if _is_finite_number(luma) and float(luma) <= thresh:
                excluded.add(str(fid))
    return excluded


def _coerce_frame_scores(
    frame_scores: Sequence[FrameScore | Mapping[str, object]],
    *,
    video_duration: float,
    video_id: str | None,
    boundary_tolerance_s: float,
) -> tuple[str | None, list[FrameScore]]:
    if not _is_finite_number(video_duration) or float(video_duration) <= 0.0:
        raise ValueError(f"video_duration must be finite and > 0, got {video_duration!r}")
    duration = float(video_duration)
    if video_id is not None and (not isinstance(video_id, str) or not video_id.strip()):
        raise ValueError("video_id must be None or a nonempty string")

    coerced: list[FrameScore] = []
    seen_frame_ids: set[str] = set()
    resolved_video_id: str | None = video_id

    for item in frame_scores:
        if isinstance(item, FrameScore):
            fs = item
        elif isinstance(item, Mapping):
            fs = FrameScore(
                frame_id=str(item["frame_id"]) if "frame_id" in item else "",
                video_id=str(item["video_id"]) if "video_id" in item else "",
                timestamp=item.get("timestamp"),  # type: ignore[arg-type]
                score=item.get("score"),  # type: ignore[arg-type]
            )
        else:
            raise ValueError(f"unsupported frame score entry: {item!r}")

        if resolved_video_id is None:
            resolved_video_id = fs.video_id
        elif fs.video_id != resolved_video_id:
            raise ValueError(
                f"mixed video_ids in frame_scores: {resolved_video_id!r} vs {fs.video_id!r}"
            )
        if fs.frame_id in seen_frame_ids:
            raise ValueError(f"duplicate frame_id in frame_scores: {fs.frame_id!r}")
        seen_frame_ids.add(fs.frame_id)

        if fs.timestamp > duration + boundary_tolerance_s:
            raise ValueError(
                f"frame {fs.frame_id!r} timestamp {fs.timestamp} exceeds "
                f"video_duration {duration}"
            )
        clipped_ts = min(fs.timestamp, duration)
        coerced.append(
            FrameScore(
                frame_id=fs.frame_id,
                video_id=fs.video_id,
                timestamp=clipped_ts,
                score=fs.score,
            )
        )

    coerced.sort(key=lambda f: (f.timestamp, f.frame_id))
    return resolved_video_id, coerced


def smooth_frame_scores(
    scores: np.ndarray,
    window: int,
    timestamps: np.ndarray | None = None,
    max_gap_s: float | None = None,
) -> np.ndarray:
    """Moving-average smoothing with optional timestamp gap awareness."""
    arr = np.asarray(scores, dtype=np.float64)
    n = arr.size
    if n <= 1 or window <= 1:
        return arr.copy()
    half = window // 2
    ts = np.asarray(timestamps, dtype=np.float64) if timestamps is not None else None
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if ts is not None and max_gap_s is not None and max_gap_s > 0:
            valid_indices = [
                j for j in range(lo, hi) if abs(ts[j] - ts[i]) <= max_gap_s + 1e-9
            ]
            out[i] = float(np.mean(arr[valid_indices])) if valid_indices else float(arr[i])
        else:
            out[i] = float(np.mean(arr[lo:hi]))
    return out


def _select_candidate_indices(
    smoothed: np.ndarray, config: TemporalConfig
) -> list[int]:
    n = smoothed.size
    if n == 0:
        return []

    floor = float(config.min_score_floor) if config.min_score_floor is not None else None
    if floor is not None:
        eligible_mask = smoothed >= floor
        if not np.any(eligible_mask):
            return []
    else:
        eligible_mask = np.ones(n, dtype=bool)

    quantile_thresh = float(np.quantile(smoothed, float(config.score_quantile)))
    effective_thresh = max(quantile_thresh, floor) if floor is not None else quantile_thresh

    selected = set(np.flatnonzero(smoothed >= effective_thresh - 1e-12).tolist())

    if config.include_local_peaks and n >= 3:
        for i in range(n):
            if not eligible_mask[i]:
                continue
            left_ok = i == 0 or smoothed[i] > smoothed[i - 1]
            right_ok = i == n - 1 or smoothed[i] >= smoothed[i + 1]
            strict_neighbor = (i > 0 and smoothed[i] > smoothed[i - 1]) or (
                i < n - 1 and smoothed[i] > smoothed[i + 1]
            )
            if left_ok and right_ok and strict_neighbor:
                selected.add(i)

    return sorted(selected)


def _estimate_half_span(frames: Sequence[FrameScore], default_span_s: float) -> float:
    if len(frames) >= 2:
        diffs = [
            frames[i + 1].timestamp - frames[i].timestamp
            for i in range(len(frames) - 1)
            if frames[i + 1].timestamp > frames[i].timestamp
        ]
        if diffs:
            median_step = float(np.median(np.asarray(diffs, dtype=np.float64)))
            return 0.5 * max(min(median_step, default_span_s * 2.0), 1e-3)
    return 0.5 * default_span_s


def _split_long_cluster(
    cluster_indices: list[int],
    frames: Sequence[FrameScore],
    smoothed: np.ndarray,
    *,
    max_duration_s: float,
    half_span_s: float,
) -> list[list[int]]:
    """Split a merged cluster whose temporal span exceeds ``max_duration_s``."""
    if not cluster_indices:
        return []
    first_ts = frames[cluster_indices[0]].timestamp
    last_ts = frames[cluster_indices[-1]].timestamp
    if (last_ts - first_ts) + 2.0 * half_span_s <= max_duration_s + 1e-9:
        return [cluster_indices]

    remaining = list(cluster_indices)
    sub_clusters: list[list[int]] = []
    max_core_span = max(0.0, max_duration_s - 2.0 * half_span_s)
    half_core = 0.5 * max_core_span

    while remaining:
        peak_idx = max(
            remaining,
            key=lambda idx: (float(smoothed[idx]), -frames[idx].timestamp, frames[idx].frame_id),
        )
        center_ts = frames[peak_idx].timestamp
        group = [
            idx
            for idx in remaining
            if abs(frames[idx].timestamp - center_ts) <= half_core + 1e-9
        ]
        if not group:
            group = [peak_idx]
        sub_clusters.append(sorted(group))
        group_set = set(group)
        remaining = [idx for idx in remaining if idx not in group_set]

    return sub_clusters


def _clip_and_bound_interval(
    raw_start: float,
    raw_end: float,
    center_ts: float,
    *,
    video_duration: float,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[float, float]:
    target_min = min(float(min_duration_s), video_duration)
    target_max = min(float(max_duration_s), video_duration)

    start = float(raw_start)
    end = float(raw_end)
    length = end - start

    if length < target_min:
        mid = 0.5 * (start + end)
        start = mid - 0.5 * target_min
        end = mid + 0.5 * target_min
    elif length > target_max:
        start = center_ts - 0.5 * target_max
        end = center_ts + 0.5 * target_max

    if start < 0.0:
        end = min(video_duration, end - start)
        start = 0.0
    if end > video_duration:
        start = max(0.0, start - (end - video_duration))
        end = video_duration

    start = max(0.0, min(start, video_duration))
    end = max(0.0, min(end, video_duration))
    if end <= start:
        eps = min(1e-3, video_duration)
        if start + eps <= video_duration:
            end = start + eps
        else:
            start = max(0.0, video_duration - eps)
            end = video_duration

    return float(round(start, 6)), float(round(end, 6))


def propose_temporal_moments(
    frame_scores: Sequence[FrameScore | Mapping[str, object]],
    *,
    video_duration: float,
    config: TemporalConfig = TemporalConfig(),
    video_id: str | None = None,
    score_type: str = "temporal_relevance",
    dark_frame_ids: Sequence[str] | set[str] | frozenset[str] | None = None,
    frame_lumas: Mapping[str, float] | None = None,
) -> list[TemporalMoment]:
    """Convert frame-level scores into ranked ``TemporalMoment`` intervals (no YOLO)."""
    if not isinstance(config, TemporalConfig):
        raise ValueError("config must be a TemporalConfig instance")
    if not isinstance(score_type, str) or not score_type.strip():
        raise ValueError("score_type must be a nonempty string")

    resolved_video_id, frames = _coerce_frame_scores(
        frame_scores,
        video_duration=video_duration,
        video_id=video_id,
        boundary_tolerance_s=config.boundary_tolerance_s,
    )
    excluded_dark = _resolve_excluded_dark_frame_ids(
        config,
        dark_frame_ids=dark_frame_ids,
        frame_lumas=frame_lumas,
    )
    if excluded_dark:
        frames = [f for f in frames if f.frame_id not in excluded_dark]
    if not frames or resolved_video_id is None:
        return []

    duration = float(video_duration)
    raw_scores = np.asarray([f.score for f in frames], dtype=np.float64)
    timestamps = np.asarray([f.timestamp for f in frames], dtype=np.float64)
    smoothed = smooth_frame_scores(
        raw_scores,
        config.smoothing_window,
        timestamps=timestamps,
        max_gap_s=float(config.max_merge_gap_s),
    )

    selected_indices = _select_candidate_indices(smoothed, config)
    if not selected_indices:
        return []

    half_span = _estimate_half_span(frames, float(config.default_frame_span_s))

    # Merge neighboring selected frames within max_merge_gap_s
    merged_clusters: list[list[int]] = []
    current_cluster = [selected_indices[0]]
    for idx in selected_indices[1:]:
        prev_idx = current_cluster[-1]
        gap = frames[idx].timestamp - frames[prev_idx].timestamp
        if gap <= float(config.max_merge_gap_s) + 1e-9:
            current_cluster.append(idx)
        else:
            merged_clusters.append(current_cluster)
            current_cluster = [idx]
    merged_clusters.append(current_cluster)

    # Split any cluster exceeding max_duration_s
    candidate_clusters: list[list[int]] = []
    for cluster in merged_clusters:
        candidate_clusters.extend(
            _split_long_cluster(
                cluster,
                frames,
                smoothed,
                max_duration_s=float(config.max_duration_s),
                half_span_s=half_span,
            )
        )

    # Also add single-peak proposals for isolated local maxima not already part of multi-frame clusters
    if config.include_local_peaks and len(frames) >= 3:
        multi_frame_indices = {
            idx for cluster in merged_clusters if len(cluster) > 1 for idx in cluster
        }
        for idx in selected_indices:
            if idx in multi_frame_indices:
                continue
            left_ok = idx == 0 or smoothed[idx] > smoothed[idx - 1]
            right_ok = idx == len(frames) - 1 or smoothed[idx] >= smoothed[idx + 1]
            if left_ok and right_ok:
                candidate_clusters.append([idx])

    proposals: list[tuple[float, float, float, tuple[str, ...], str, float]] = []
    for cluster in candidate_clusters:
        peak_idx = max(
            cluster,
            key=lambda idx: (float(smoothed[idx]), -frames[idx].timestamp, frames[idx].frame_id),
        )
        center_ts = frames[peak_idx].timestamp
        peak_score = float(smoothed[peak_idx])

        # Prune peripheral frames far below peak score (relative peak trimming)
        if config.relative_peak_threshold > 0.0 and len(cluster) > 1:
            trim_thresh = float(config.relative_peak_threshold) * peak_score
            p_pos = cluster.index(peak_idx)
            l_pos = p_pos
            while l_pos > 0 and float(smoothed[cluster[l_pos - 1]]) >= trim_thresh - 1e-9:
                l_pos -= 1
            r_pos = p_pos
            while r_pos < len(cluster) - 1 and float(smoothed[cluster[r_pos + 1]]) >= trim_thresh - 1e-9:
                r_pos += 1
            trimmed_cluster = cluster[l_pos : r_pos + 1]
        else:
            trimmed_cluster = cluster

        first_ts = frames[trimmed_cluster[0]].timestamp
        last_ts = frames[trimmed_cluster[-1]].timestamp
        t_start, t_end = _clip_and_bound_interval(
            first_ts - half_span,
            last_ts + half_span,
            center_ts,
            video_duration=duration,
            min_duration_s=float(config.min_duration_s),
            max_duration_s=float(config.max_duration_s),
        )

        # Evidence frames: all searchable frames inside [t_start, t_end]
        in_window = [
            frames[i].frame_id
            for i in range(len(frames))
            if t_start - 1e-6 <= frames[i].timestamp <= t_end + 1e-6
        ]
        if not in_window:
            in_window = [frames[i].frame_id for i in trimmed_cluster]
        evidence_ids = tuple(dict.fromkeys(in_window))

        cluster_scores = np.sort(smoothed[trimmed_cluster])[::-1]
        top_support = float(np.mean(cluster_scores[: min(3, cluster_scores.size)]))
        segment_score = 0.6 * peak_score + 0.4 * top_support

        peak_fid = frames[peak_idx].frame_id
        peak_ts = frames[peak_idx].timestamp
        proposals.append((segment_score, t_start, t_end, evidence_ids, peak_fid, peak_ts))

    # Deterministic sort: descending score, then earlier start, shorter/earlier end, IDs
    proposals.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    # Deduplicate identical intervals and apply Temporal NMS
    kept: list[tuple[float, float, float, tuple[str, ...], str, float]] = []
    seen_intervals: set[tuple[float, float]] = set()
    nms_thresh = (
        float(config.nms_tiou_threshold)
        if config.nms_tiou_threshold is not None
        else None
    )

    for score, t_start, t_end, evidence_ids, peak_fid, peak_ts in proposals:
        interval = (t_start, t_end)
        if interval in seen_intervals:
            continue
        if nms_thresh is not None and any(
            temporal_iou(interval, (k_start, k_end)) >= nms_thresh - 1e-12
            for _, k_start, k_end, _, _, _ in kept
        ):
            continue
        seen_intervals.add(interval)
        kept.append((score, t_start, t_end, evidence_ids, peak_fid, peak_ts))
        if len(kept) >= config.top_k:
            break

    return [
        TemporalMoment(
            video_id=resolved_video_id,
            t_start=t_start,
            t_end=t_end,
            rank=rank,
            score=score,
            evidence_frame_ids=evidence_ids,
            score_type=score_type,
            detections=None,
            peak_frame_id=peak_fid,
            peak_timestamp=peak_ts,
        )
        for rank, (score, t_start, t_end, evidence_ids, peak_fid, peak_ts) in enumerate(kept, 1)
    ]


def score_video_frames(
    entry: VideoIndexEntry,
    encoder_specs: Mapping[str, EncoderSpec],
    query_vectors: Mapping[str, np.ndarray],
    *,
    video_id: str | None = None,
) -> tuple[list[FrameScore], str]:
    """Compute exact per-frame scores across all searchable frames of ``video_id``.

    Unlike ``LiteRetriever.search_vectors`` (which truncates to Top-K keyframes),
    this read-only path scores every searchable frame in ``entry.index`` belonging
    to ``video_id`` and orders them chronologically by ``(timestamp, frame_id)``.
    """
    if not query_vectors:
        raise ValueError("query requires at least one vector")
    saved_models = getattr(entry.index, "encoder_metadata", {})
    for slot, spec in encoder_specs.items():
        saved = saved_models.get(slot)
        if saved is not None and saved != {"id": spec.model_id, "revision": spec.revision}:
            raise ValueError(f"Index model mismatch for {slot}; rebuild index")

    target_video_id = video_id if video_id is not None else entry.video_id
    if not isinstance(target_video_id, str) or not target_video_id.strip():
        raise ValueError("video_id must be a nonempty string")

    index = entry.index
    candidate_rows = [
        row for row, vid in enumerate(index.video_ids) if vid == target_video_id
    ]
    if not candidate_rows:
        return [], "temporal_cosine" if len(query_vectors) == 1 else "temporal_rrf"

    per_slot_pairs: dict[str, list[tuple[int, float]]] = {}
    for slot, vector in query_vectors.items():
        if slot not in encoder_specs or slot not in {"clip", "siglip"}:
            raise ValueError(f"unknown encoder slot: {slot}")
        if not index.has_encoder(slot):
            raise ValueError(f"index has no encoder slot: {slot}")
        q = np.asarray(vector, dtype=np.float32)
        dim = index.dimension(slot)
        if q.ndim != 1 or q.size != dim:
            raise ValueError(f"query dimension for {slot} must be {dim}")
        if not np.isfinite(q).all():
            raise ValueError("query vector must be finite")
        norm = float(np.linalg.norm(q))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("query vector must be nonzero and finite")
        q = q / norm
        pairs = index.exact_dense_search(q, slot, candidate_rows, len(candidate_rows))
        per_slot_pairs[slot] = sorted(
            pairs, key=lambda pair: (-pair[1], index.ids[pair[0]])
        )

    if len(per_slot_pairs) == 1:
        _, pairs = next(iter(per_slot_pairs.items()))
        row_scores = {row: float(score) for row, score in pairs}
        score_type = "temporal_cosine"
    else:
        fused = reciprocal_rank_fusion(
            {
                slot: [row for row, _ in pairs]
                for slot, pairs in per_slot_pairs.items()
            }
        )
        row_scores = {row: float(score) for row, (score, _) in fused.items()}
        score_type = "temporal_rrf"

    frame_scores = [
        FrameScore(
            frame_id=index.ids[row],
            video_id=index.video_ids[row],
            timestamp=float(index.timestamps[row]),
            score=row_scores[row],
        )
        for row in candidate_rows
    ]
    frame_scores.sort(key=lambda fs: (fs.timestamp, fs.frame_id))
    return frame_scores, score_type


class TemporalBaselineRetriever:
    """Read-only P03 temporal moment localizer over an existing ``VideoIndexEntry``."""

    def __init__(
        self,
        entry: VideoIndexEntry,
        encoder_specs: Mapping[str, EncoderSpec],
        *,
        config: TemporalConfig = TemporalConfig(),
        dark_frame_ids: Sequence[str] | set[str] | frozenset[str] | None = None,
        frame_lumas: Mapping[str, float] | None = None,
    ) -> None:
        if not isinstance(config, TemporalConfig):
            raise ValueError("config must be a TemporalConfig instance")
        self.entry = entry
        self.encoder_specs = dict(encoder_specs)
        self.config = config
        self.dark_frame_ids = (
            frozenset(str(fid) for fid in dark_frame_ids)
            if dark_frame_ids is not None
            else None
        )
        self.frame_lumas = dict(frame_lumas) if frame_lumas is not None else None
        saved_models = getattr(entry.index, "encoder_metadata", {})
        for slot, spec in self.encoder_specs.items():
            saved = saved_models.get(slot)
            if saved is not None and saved != {
                "id": spec.model_id,
                "revision": spec.revision,
            }:
                raise ValueError(f"Index model mismatch for {slot}; rebuild index")

    def search_temporal_text(
        self,
        query: str,
        encoders: Mapping[str, Encoder],
        *,
        video_duration: float,
        config: TemporalConfig | None = None,
        video_id: str | None = None,
        dark_frame_ids: Sequence[str] | set[str] | frozenset[str] | None = None,
        frame_lumas: Mapping[str, float] | None = None,
    ) -> list[TemporalMoment]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        if not encoders:
            raise ValueError("query requires at least one encoder")
        vectors = {
            slot: encoder.encode_text(query.strip())
            for slot, encoder in encoders.items()
        }
        return self.search_temporal_vectors(
            vectors,
            video_duration=video_duration,
            config=config,
            video_id=video_id,
            dark_frame_ids=dark_frame_ids,
            frame_lumas=frame_lumas,
        )

    def search_temporal_vectors(
        self,
        query_vectors: Mapping[str, np.ndarray],
        *,
        video_duration: float,
        config: TemporalConfig | None = None,
        video_id: str | None = None,
        dark_frame_ids: Sequence[str] | set[str] | frozenset[str] | None = None,
        frame_lumas: Mapping[str, float] | None = None,
    ) -> list[TemporalMoment]:
        active_config = config if config is not None else self.config
        target_vid = video_id if video_id is not None else self.entry.video_id
        frame_scores, score_type = score_video_frames(
            self.entry,
            self.encoder_specs,
            query_vectors,
            video_id=video_id,
        )

        active_dark_ids = (
            dark_frame_ids if dark_frame_ids is not None else self.dark_frame_ids
        )
        active_lumas = (
            frame_lumas if frame_lumas is not None else self.frame_lumas
        )
        if active_config.dark_frame_luma_max is not None:
            excluded_dark = _resolve_excluded_dark_frame_ids(
                active_config,
                dark_frame_ids=active_dark_ids,
                frame_lumas=active_lumas,
            )
            if excluded_dark:
                frame_scores = [
                    fs for fs in frame_scores if fs.frame_id not in excluded_dark
                ]

        return propose_temporal_moments(
            frame_scores,
            video_duration=video_duration,
            config=active_config,
            video_id=target_vid,
            score_type=score_type,
        )
