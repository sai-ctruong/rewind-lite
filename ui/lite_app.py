"""Small local Flask API for the Rewind Lite text-to-video flow.

Run ``python -m ui.lite_app``. The advanced ui.app endpoints are not registered.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import replace
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from scripts.lite_search import (DEFAULT_MODEL, DEFAULT_REVISION, _real_encoder,
                                 probe_video, sha256_file, validate_manifest)


_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class LiteService:
    def __init__(self, *, video_dir: str | Path, index_root: str | Path,
                 encoder_factory=None, model_id: str = DEFAULT_MODEL,
                 revision: str = DEFAULT_REVISION, device: str = "auto",
                 sample_every_s: float = 1.0, max_frames: int | None = None):
        from retrieval.video_engine import VideoSearchEngine

        self.video_dir = Path(video_dir).resolve()
        self.index_root = Path(index_root).resolve()
        self.encoder_factory = encoder_factory or _real_encoder
        self.model_id, self.revision, self.device = model_id, revision, device
        self.sample_every_s, self.max_frames = sample_every_s, max_frames
        self.engine = VideoSearchEngine(
            encoder_names=(model_id,), sample_every_s=sample_every_s,
            max_frames=max_frames, query_templates=("{q}",),
            enable_ocr=False, enable_asr=False, enable_caption=False,
            embed_batch_size=8, decode_backend="cv2", use_gpu_decode=False,
            parallel_index=False)
        self.encoder = None
        self.selected: str | None = None
        self.entries: dict = {}
        self.retrievers: dict = {}
        self.source_signatures: dict[str, tuple[int, int]] = {}
        self.index_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self._progress = {"status": "idle", "video_id": None,
                          "count": 0, "elapsed_s": 0.0, "error_code": None}

    def progress(self) -> dict:
        with self.state_lock:
            return dict(self._progress)

    def _set_progress(self, **values) -> None:
        with self.state_lock:
            self._progress.update(values)

    def _video_path(self, video_id: str) -> Path:
        if not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id):
            raise ApiError(400, "invalid_video_id", "Invalid video ID")
        path = (self.video_dir / f"{video_id}.mp4").resolve()
        if not path.is_relative_to(self.video_dir) or not path.is_file():
            raise ApiError(404, "video_not_found", "Video not found")
        return path

    def resolve_id(self, video_id: str | None) -> str:
        target = video_id if video_id is not None else self.selected
        if target is None:
            raise ApiError(409, "video_not_selected", "Select a video first")
        self._video_path(target)
        return target

    def list_videos(self) -> list[str]:
        if not self.video_dir.is_dir():
            return []
        return sorted(path.stem for path in self.video_dir.glob("*.mp4")
                      if _VIDEO_ID.fullmatch(path.stem) and path.is_file()
                      and path.resolve().is_relative_to(self.video_dir))

    def select(self, video_id: str) -> str:
        self._video_path(video_id)
        self.selected = video_id
        return video_id

    def upload(self, file) -> str:
        if not file or not file.filename or Path(file.filename).suffix.lower() != ".mp4":
            raise ApiError(415, "unsupported_video", "Upload an MP4 video")
        video_id = uuid.uuid4().hex
        path = self.video_dir / f"{video_id}.mp4"
        try:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            file.save(path)
        except OSError as exc:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ApiError(500, "upload_failed", "Upload could not be saved") from exc
        try:
            probe_video(path)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            raise ApiError(415, "unsupported_video", "MP4 video could not be decoded")
        self.selected = video_id
        return video_id

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = self.encoder_factory(
                self.model_id, self.revision, self.device)
        return self.encoder

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def _cache(self, video_id: str, path: Path, entry) -> None:
        from retrieval.lite_core import EncoderSpec, LiteRetriever

        retriever = LiteRetriever(
            entry, {"clip": EncoderSpec(self.model_id, self.revision)})
        if entry.video_id != video_id or any(
                entry.raws[rid].video_id != video_id for rid in entry.index.ids):
            raise ApiError(409, "index_expired", "Saved index does not match video")
        self.entries[video_id] = entry
        self.retrievers[video_id] = retriever
        self.source_signatures[video_id] = self._signature(path)

    def _load_saved(self, video_id: str, path: Path):
        from retrieval.video_engine import VideoIndexEntry

        directory = self.index_root / video_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            if (directory / "entry.pkl").exists():
                raise ApiError(409, "index_incomplete", "Saved index is incomplete")
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_manifest(manifest, path, self.model_id, self.revision)
            sampling = manifest.get("sampling", {})
            if (sampling.get("every_s") != self.sample_every_s or
                    sampling.get("max_frames") != self.max_frames):
                raise ValueError("sampling changed")
            entry = VideoIndexEntry.load(directory)
            self._cache(video_id, path, entry)
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            raise ApiError(409, "index_expired", "Saved index is missing or expired") from exc
        return entry

    def _checked_entry(self, video_id: str, path: Path):
        entry = self.entries.get(video_id)
        if entry is None:
            entry = self._load_saved(video_id, path)
        if entry is None:
            raise ApiError(409, "index_missing", "Index this video before searching")
        if self._signature(path) != self.source_signatures[video_id]:
            raise ApiError(409, "index_expired", "Video changed; rebuild its index")
        return entry

    def index(self, video_id: str, *, rebuild: bool = False) -> dict:
        path = self._video_path(video_id)
        if not self.index_lock.acquire(blocking=False):
            raise ApiError(409, "index_busy", "Indexing or search is already running")
        self._set_progress(status="indexing", video_id=video_id,
                           count=0, elapsed_s=0.0, error_code=None)
        try:
            if not rebuild:
                entry = self.entries.get(video_id)
                if entry is not None:
                    self._checked_entry(video_id, path)
                    self._set_progress(status="ready", count=entry.num_indexed)
                    return {"video_id": video_id, "cached": True,
                            "frames": entry.num_indexed, "sampled": entry.num_sampled}
                entry = self._load_saved(video_id, path)
                if entry is not None:
                    self._set_progress(status="ready", count=entry.num_indexed)
                    return {"video_id": video_id, "cached": True,
                            "frames": entry.num_indexed, "sampled": entry.num_sampled}
            source = probe_video(path)
            encoder = self._get_encoder()
            self.engine.set_encoders([encoder])
            started = time.perf_counter()

            def on_progress(count, elapsed):
                self._set_progress(count=count, elapsed_s=round(elapsed, 3))

            entry = self.engine.index_video(
                path, self.index_root / video_id / "frames", video_id=video_id,
                progress_cb=on_progress)
            index_seconds = round(time.perf_counter() - started, 3)
            entry.index.encoder_metadata = {
                "clip": {"id": self.model_id, "revision": self.revision}}
            directory = self.index_root / video_id
            entry.save(directory)
            manifest = {
                "schema_version": 1,
                "video": {"video_id": video_id, "sha256": sha256_file(path),
                          "size_bytes": path.stat().st_size, **source},
                "model": {"id": self.model_id, "revision": self.revision},
                "sampling": {"every_s": self.sample_every_s,
                             "max_frames": self.max_frames,
                             "decode_backend": "cv2",
                             "hist_duplicate_threshold": 0.985},
                "dedup": {"similarity_threshold": 0.97},
                "n_sampled": entry.num_sampled,
                "n_indexed": entry.num_indexed,
                "index_seconds": index_seconds,
            }
            temporary = directory / "manifest.json.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(directory / "manifest.json")
            self._cache(video_id, path, entry)
            self._set_progress(status="ready", count=entry.num_indexed,
                               elapsed_s=index_seconds)
            return {"video_id": video_id, "cached": False,
                    "frames": entry.num_indexed, "sampled": entry.num_sampled}
        except ApiError as exc:
            self._set_progress(status="error", error_code=exc.code)
            raise
        except (MemoryError, RuntimeError, OSError, ValueError) as exc:
            oom = isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
            code = "model_oom" if oom else "index_failed"
            self._set_progress(status="error", error_code=code)
            raise ApiError(503 if oom else 500, code,
                           "Model ran out of memory" if oom else "Indexing failed") from exc
        finally:
            self.index_lock.release()

    def search(self, video_id: str, query: str, top_k: int):
        path = self._video_path(video_id)
        if not self.index_lock.acquire(blocking=False):
            raise ApiError(409, "index_busy", "Indexing or search is already running")
        try:
            self._checked_entry(video_id, path)
            return self.retrievers[video_id].search_text(
                query, {"clip": self._get_encoder()}, top_k=top_k)
        except ApiError:
            raise
        except (MemoryError, RuntimeError, ValueError) as exc:
            oom = isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
            raise ApiError(503 if oom else 500,
                           "model_oom" if oom else "search_failed",
                           "Model ran out of memory" if oom else "Search failed") from exc
        finally:
            self.index_lock.release()

    def frame(self, video_id: str, frame_id: str) -> bytes:
        from ingestion.schemas import frame_jpeg_bytes

        path = self._video_path(video_id)
        if not self.index_lock.acquire(blocking=False):
            raise ApiError(409, "index_busy", "Indexing or search is already running")
        try:
            entry = self._checked_entry(video_id, path)
            if frame_id not in entry.index.ids or frame_id not in entry.raws:
                raise ApiError(404, "frame_not_found", "Frame not found")
            raw = entry.raws[frame_id]
            # Never follow image/source paths stored inside an index through HTTP.
            safe_raw = replace(raw, image_path=None, source_video=str(path))
            try:
                data = frame_jpeg_bytes(safe_raw)
            except (OSError, ValueError) as exc:
                raise ApiError(404, "frame_unavailable", "Frame image unavailable") from exc
            if not data:
                raise ApiError(404, "frame_unavailable", "Frame image unavailable")
            return data
        finally:
            self.index_lock.release()


def create_lite_app(*, video_dir: str | Path | None = None,
                    index_root: str | Path | None = None,
                    encoder_factory=None, model_id: str = DEFAULT_MODEL,
                    revision: str = DEFAULT_REVISION, device: str = "auto",
                    sample_every_s: float = 1.0, max_frames: int | None = None,
                    max_upload_bytes: int = 512 * 1024 * 1024) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes
    service = LiteService(
        video_dir=video_dir or _ROOT / "data" / "videos",
        index_root=index_root or _ROOT / "artifacts" / "indexes",
        encoder_factory=encoder_factory, model_id=model_id, revision=revision,
        device=device, sample_every_s=sample_every_s, max_frames=max_frames)
    app.extensions["lite_service"] = service

    @app.errorhandler(ApiError)
    def api_error(exc: ApiError):
        return jsonify(error={"code": exc.code, "message": exc.message}), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(exc):
        return jsonify(error={"code": "upload_too_large",
                              "message": "Video exceeds upload limit"}), 413

    def body() -> dict:
        data = request.get_json(silent=True)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ApiError(400, "invalid_request", "Expected a JSON object")
        return data

    @app.get("/api/lite/health")
    def health():
        return jsonify(ok=True, selected_video=service.selected,
                       indexed_videos=len(service.entries))

    @app.get("/api/lite/videos")
    def videos():
        return jsonify(videos=service.list_videos(), selected_video=service.selected)

    @app.post("/api/lite/upload")
    def upload():
        video_id = service.upload(request.files.get("video"))
        return jsonify(video_id=video_id, selected=True), 201

    @app.post("/api/lite/select")
    def select():
        video_id = service.select(body().get("video_id"))
        return jsonify(video_id=video_id, selected=True)

    @app.post("/api/lite/index")
    def index():
        data = body()
        video_id = service.resolve_id(data.get("video_id"))
        rebuild = data.get("rebuild", False)
        if not isinstance(rebuild, bool):
            raise ApiError(400, "invalid_rebuild", "rebuild must be a boolean")
        return jsonify(service.index(video_id, rebuild=rebuild))

    @app.get("/api/lite/progress")
    def progress():
        return jsonify(service.progress())

    @app.post("/api/lite/search")
    def search():
        data = body()
        query = data.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise ApiError(400, "invalid_query", "Enter a text query (1–512 characters)")
        top_k = data.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ApiError(400, "invalid_top_k", "top_k must be between 1 and 50")
        video_id = service.resolve_id(data.get("video_id"))
        hits = service.search(video_id, query.strip(), top_k)
        results = [{"rank": hit.rank, "score": hit.score,
                    "score_kind": hit.score_kind, "video_id": hit.video_id,
                    "frame_id": hit.frame_id, "timestamp": hit.timestamp,
                    "image_url": url_for("frame", video_id=video_id,
                                         frame_id=hit.frame_id)}
                   for hit in hits]
        return jsonify(video_id=video_id, query=query.strip(), results=results)

    @app.get("/api/lite/frames/<video_id>/<path:frame_id>")
    def frame(video_id: str, frame_id: str):
        return send_file(BytesIO(service.frame(video_id, frame_id)),
                         mimetype="image/jpeg")

    @app.get("/api/lite/videos/<video_id>/stream")
    def stream(video_id: str):
        path = service._video_path(video_id)
        return send_file(path, mimetype="video/mp4", conditional=True)

    return app


if __name__ == "__main__":
    create_lite_app().run(host="127.0.0.1", port=5000, threaded=True)
