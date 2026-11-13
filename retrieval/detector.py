"""P04 object detection adapter, detection cache, and candidate-frame integration.

Per `REWIND_LITE_IMPROVEMENT_PLAN.md` §1.1, §1.3, §3 P04:
- Pretrained lightweight YOLO nano/small model for object evidence.
- Run ONLY on candidate frames (from P03 TemporalMoment.evidence_frame_ids),
  never across the full video unless approved.
- Explicit coordinate conventions: both pixel `bbox_xyxy` and normalized
  `bbox_norm_xyxy` [0, 1]. Rejects invalid / degenerate / out-of-bounds boxes.
- Strict SHA-256 detection cache key parameterized by video_id, frame_id,
  timestamp, model_id, conf_threshold, iou_threshold, imgsz with invalidation.
- VRAM management: small batches, CPU fallback, and explicit `.unload()` to
  respect workstation VRAM budget (~4.29 GB).
- When `config.enabled=False`, moments retain `detections=None` without alteration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from retrieval.temporal import TemporalMoment


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class ObjectDetection:
    """Single validated bounding box detection per frame."""

    frame_id: str
    video_id: str
    timestamp: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_norm_xyxy: tuple[float, float, float, float]
    image_size: tuple[int, int]
    model_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a nonempty string")
        if not _is_finite_number(self.timestamp) or float(self.timestamp) < 0.0:
            raise ValueError(f"timestamp must be finite and >= 0, got {self.timestamp!r}")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError(f"class_id must be an integer >= 0, got {self.class_id!r}")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a nonempty string")
        if not _is_finite_number(self.confidence) or not (0.0 < float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in (0.0, 1.0], got {self.confidence!r}")

        # Validate image size (width, height)
        if (
            not isinstance(self.image_size, (tuple, list))
            or len(self.image_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.image_size)
        ):
            raise ValueError(f"image_size must be (width, height) with positive ints, got {self.image_size!r}")
        w, h = int(self.image_size[0]), int(self.image_size[1])

        # Validate bbox_xyxy
        if not isinstance(self.bbox_xyxy, (tuple, list)) or len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must be 4 coordinates (x1, y1, x2, y2)")
        x1, y1, x2, y2 = (float(v) for v in self.bbox_xyxy)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("bbox_xyxy coordinates must be finite numbers")
        if x1 < 0.0 or y1 < 0.0 or x2 > w + 1e-3 or y2 > h + 1e-3 or x1 >= x2 or y1 >= y2:
            raise ValueError(
                f"invalid bbox_xyxy [{x1}, {y1}, {x2}, {y2}] for image_size ({w}, {h})"
            )

        # Validate bbox_norm_xyxy
        if not isinstance(self.bbox_norm_xyxy, (tuple, list)) or len(self.bbox_norm_xyxy) != 4:
            raise ValueError("bbox_norm_xyxy must be 4 normalized coordinates (nx1, ny1, nx2, ny2)")
        nx1, ny1, nx2, ny2 = (float(v) for v in self.bbox_norm_xyxy)
        if not all(math.isfinite(v) for v in (nx1, ny1, nx2, ny2)):
            raise ValueError("bbox_norm_xyxy coordinates must be finite numbers")
        if nx1 < 0.0 or ny1 < 0.0 or nx2 > 1.0 + 1e-6 or ny2 > 1.0 + 1e-6 or nx1 >= nx2 or ny1 >= ny2:
            raise ValueError(
                f"invalid bbox_norm_xyxy [{nx1}, {ny1}, {nx2}, {ny2}]"
            )

        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "class_name", self.class_name.strip().lower())
        object.__setattr__(self, "image_size", (w, h))
        object.__setattr__(self, "bbox_xyxy", (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)))
        object.__setattr__(self, "bbox_norm_xyxy", (round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation compatible with P05 reranker and P08 UI."""
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "timestamp": self.timestamp,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_norm_xyxy": list(self.bbox_norm_xyxy),
            "image_size": list(self.image_size),
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for object detection and inference budget."""

    enabled: bool = False
    model_id: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    batch_size: int = 4
    device: str = "auto"
    half_precision: bool = True
    allowed_classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if not _is_finite_number(self.conf_threshold) or not (0.0 <= float(self.conf_threshold) <= 1.0):
            raise ValueError("conf_threshold must be in [0.0, 1.0]")
        if not _is_finite_number(self.iou_threshold) or not (0.0 < float(self.iou_threshold) <= 1.0):
            raise ValueError("iou_threshold must be in (0.0, 1.0]")
        if isinstance(self.imgsz, bool) or not isinstance(self.imgsz, int) or self.imgsz <= 0:
            raise ValueError("imgsz must be a positive int")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive int")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty string")
        if not isinstance(self.half_precision, bool):
            raise ValueError("half_precision must be a bool")
        if self.allowed_classes is not None:
            if not isinstance(self.allowed_classes, (tuple, list)) or any(
                not isinstance(c, str) or not c.strip() for c in self.allowed_classes
            ):
                raise ValueError("allowed_classes must be None or a sequence of nonempty strings")
            object.__setattr__(
                self,
                "allowed_classes",
                tuple(dict.fromkeys(c.strip().lower() for c in self.allowed_classes)),
            )


class DetectionCache:
    """Deterministic, parameterized cache for raw frame detections with invalidation."""

    SCHEMA_VERSION = "p04_v1"

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory_cache: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def make_key(
        cls,
        *,
        video_id: str,
        frame_id: str,
        timestamp: float,
        model_id: str,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
    ) -> str:
        payload = f"{cls.SCHEMA_VERSION}|{video_id}|{frame_id}|{timestamp:.6f}|{model_id}|{conf_threshold:.4f}|{iou_threshold:.4f}|{imgsz}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> list[dict[str, Any]] | None:
        with self._lock:
            if key in self._memory_cache:
                return [dict(d) for d in self._memory_cache[key]]
            if self.cache_dir is not None:
                path = self.cache_dir / f"{key}.json"
                if path.is_file():
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            self._memory_cache[key] = data
                            return [dict(d) for d in data]
                    except Exception:
                        pass
            return None

    def put(self, key: str, detections: Sequence[Mapping[str, Any] | ObjectDetection]) -> None:
        serialized = [
            d.to_dict() if isinstance(d, ObjectDetection) else dict(d)
            for d in detections
        ]
        with self._lock:
            self._memory_cache[key] = serialized
            if self.cache_dir is not None:
                path = self.cache_dir / f"{key}.json"
                try:
                    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
                except Exception:
                    pass

    def invalidate(self, key: str | None = None) -> None:
        with self._lock:
            if key is not None:
                self._memory_cache.pop(key, None)
                if self.cache_dir is not None:
                    path = self.cache_dir / f"{key}.json"
                    if path.is_file():
                        path.unlink(missing_ok=True)
            else:
                self._memory_cache.clear()
                if self.cache_dir is not None:
                    for p in self.cache_dir.glob("*.json"):
                        p.unlink(missing_ok=True)


@runtime_checkable
class DetectorBackend(Protocol):
    def detect_batch(
        self,
        frames: Sequence[tuple[str, str, float, np.ndarray]],
        config: DetectorConfig,
    ) -> list[list[ObjectDetection]]:
        ...

    def unload(self) -> None:
        ...


class UltralyticsYOLOBackend:
    """Pretrained YOLO backend with lazy loading, CPU fallback, and memory cleanup."""

    def __init__(self, model_id: str = "yolov8n.pt") -> None:
        self.model_id = model_id
        self._model = None
        self._device = None
        self._lock = threading.Lock()

    def _resolve_device(self, requested: str) -> str:
        if requested in ("cuda", "cpu"):
            return requested
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_model(self, config: DetectorConfig):
        if self._model is None or self.model_id != config.model_id:
            from ultralytics import YOLO
            self.model_id = config.model_id
            self._model = YOLO(self.model_id)
            self._device = self._resolve_device(config.device)
        return self._model

    def detect_batch(
        self,
        frames: Sequence[tuple[str, str, float, np.ndarray]],
        config: DetectorConfig,
    ) -> list[list[ObjectDetection]]:
        if not frames:
            return []
        with self._lock:
            model = self._get_model(config)
            device = self._device or "cpu"

            images = [f[3] for f in frames]
            use_half = config.half_precision and device == "cuda"
            try:
                results = model.predict(
                    source=images,
                    conf=float(config.conf_threshold),
                    iou=float(config.iou_threshold),
                    imgsz=int(config.imgsz),
                    device=device,
                    half=use_half,
                    verbose=False,
                )
            except Exception:
                # Fallback to CPU if CUDA fails (e.g. OOM)
                if device != "cpu":
                    device = "cpu"
                    results = model.predict(
                        source=images,
                        conf=float(config.conf_threshold),
                        iou=float(config.iou_threshold),
                        imgsz=int(config.imgsz),
                        device="cpu",
                        half=False,
                        verbose=False,
                    )
                else:
                    raise

        batch_detections: list[list[ObjectDetection]] = []
        for (frame_id, video_id, timestamp, img), res in zip(frames, results):
            h, w = img.shape[:2]
            frame_dets: list[ObjectDetection] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                names = getattr(res, "names", {})

                for box, conf, cls_idx in zip(xyxy, confs, classes):
                    bx1 = max(0.0, min(float(box[0]), float(w)))
                    by1 = max(0.0, min(float(box[1]), float(h)))
                    bx2 = max(0.0, min(float(box[2]), float(w)))
                    by2 = max(0.0, min(float(box[3]), float(h)))
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    cls_name = str(names.get(int(cls_idx), str(cls_idx))).strip().lower()
                    nx1 = bx1 / w
                    ny1 = by1 / h
                    nx2 = bx2 / w
                    ny2 = by2 / h
                    try:
                        det = ObjectDetection(
                            frame_id=frame_id,
                            video_id=video_id,
                            timestamp=timestamp,
                            class_id=int(cls_idx),
                            class_name=cls_name,
                            confidence=float(conf),
                            bbox_xyxy=(bx1, by1, bx2, by2),
                            bbox_norm_xyxy=(nx1, ny1, nx2, ny2),
                            image_size=(w, h),
                            model_id=self.model_id,
                        )
                        frame_dets.append(det)
                    except ValueError:
                        continue
            batch_detections.append(frame_dets)
        return batch_detections

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._device = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


_SHARED_DETECTOR_LOCK = threading.Lock()
_SHARED_BACKENDS: dict[str, DetectorBackend] = {}
_SHARED_DETECTION_CACHE = DetectionCache()


def get_shared_detector_backend(model_id: str = "yolov8n.pt") -> DetectorBackend:
    """Return a process-wide lazy-initialized detector backend for ``model_id``."""
    with _SHARED_DETECTOR_LOCK:
        backend = _SHARED_BACKENDS.get(model_id)
        if backend is None:
            backend = UltralyticsYOLOBackend(model_id=model_id)
            _SHARED_BACKENDS[model_id] = backend
        return backend


def get_shared_detection_cache() -> DetectionCache:
    """Return the process-wide in-memory detection cache."""
    return _SHARED_DETECTION_CACHE


def reset_shared_detector_state() -> None:
    """Clear shared default backends and in-memory detection cache."""
    with _SHARED_DETECTOR_LOCK:
        for b in _SHARED_BACKENDS.values():
            try:
                b.unload()
            except Exception:
                pass
        _SHARED_BACKENDS.clear()
        _SHARED_DETECTION_CACHE.invalidate()


def attach_moment_detections(
    moments: Sequence[TemporalMoment],
    frame_images: Mapping[str, np.ndarray],
    *,
    config: DetectorConfig = DetectorConfig(),
    backend: DetectorBackend | None = None,
    cache: DetectionCache | None = None,
    frame_timestamps: Mapping[str, float] | None = None,
) -> list[TemporalMoment]:
    """Attach object detections only to candidate frames of proposed moments.

    If ``config.enabled=False``, returns moments unchanged with ``detections=None``.
    """
    if not moments:
        return []
    if not config.enabled:
        return list(moments)

    detector_backend = (
        backend
        if backend is not None
        else get_shared_detector_backend(config.model_id)
    )
    if cache is not None:
        detector_cache = cache
    elif backend is None:
        detector_cache = get_shared_detection_cache()
    else:
        detector_cache = getattr(backend, "_default_detection_cache", None)
        if detector_cache is None:
            detector_cache = DetectionCache()
            try:
                setattr(backend, "_default_detection_cache", detector_cache)
            except Exception:
                pass

    # Collect unique candidate frame IDs
    needed_frame_ids = list(dict.fromkeys(
        fid
        for m in moments
        for fid in m.evidence_frame_ids
        if fid in frame_images or (frame_timestamps is not None and fid in frame_timestamps)
    ))

    # Determine frame metadata
    detections_by_frame: dict[str, list[dict[str, Any]]] = {}
    missing_frames: list[tuple[str, str, float, np.ndarray]] = []
    video_id = moments[0].video_id

    for fid in needed_frame_ids:
        ts = (
            float(frame_timestamps[fid])
            if frame_timestamps and fid in frame_timestamps
            else 0.0
        )
        key = DetectionCache.make_key(
            video_id=video_id,
            frame_id=fid,
            timestamp=ts,
            model_id=config.model_id,
            conf_threshold=float(config.conf_threshold),
            iou_threshold=float(config.iou_threshold),
            imgsz=int(config.imgsz),
        )
        cached = detector_cache.get(key)
        if cached is not None:
            detections_by_frame[fid] = cached
        elif fid in frame_images:
            missing_frames.append((fid, video_id, ts, frame_images[fid]))

    # Run inference for cache misses in small batches
    batch_size = max(1, config.batch_size)
    for i in range(0, len(missing_frames), batch_size):
        chunk = missing_frames[i : i + batch_size]
        chunk_results = detector_backend.detect_batch(chunk, config)
        for (fid, vid, ts, _), det_list in zip(chunk, chunk_results):
            key = DetectionCache.make_key(
                video_id=vid,
                frame_id=fid,
                timestamp=ts,
                model_id=config.model_id,
                conf_threshold=float(config.conf_threshold),
                iou_threshold=float(config.iou_threshold),
                imgsz=int(config.imgsz),
            )
            detector_cache.put(key, det_list)
            detections_by_frame[fid] = [d.to_dict() for d in det_list]

    # Filter by allowed_classes if specified in config
    allowed = set(config.allowed_classes) if config.allowed_classes is not None else None

    # Attach detections to each moment based on its evidence_frame_ids
    output_moments: list[TemporalMoment] = []
    for m in moments:
        moment_dets: list[dict[str, Any]] = []
        for fid in m.evidence_frame_ids:
            for d in detections_by_frame.get(fid, []):
                if allowed is None or d["class_name"] in allowed:
                    moment_dets.append(dict(d))
        output_moments.append(
            TemporalMoment(
                video_id=m.video_id,
                t_start=m.t_start,
                t_end=m.t_end,
                rank=m.rank,
                score=m.score,
                evidence_frame_ids=m.evidence_frame_ids,
                score_type=m.score_type,
                detections=tuple(moment_dets),
                peak_frame_id=getattr(m, "peak_frame_id", None),
                peak_timestamp=getattr(m, "peak_timestamp", None),
            )
        )

    return output_moments
