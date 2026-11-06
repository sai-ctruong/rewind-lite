"""Single SigLIP encoder: MP4 -> keyframes -> cosine Top-K timestamps.

Run ``python -m scripts.lite_search video.mp4 'người đang đi bộ'``.
The local index is reused on later runs; no video embeddings are recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ingestion import model_cache  # noqa: F401 - set HF_HOME before HF imports

DEFAULT_MODEL = "google/siglip2-base-patch16-256"
DEFAULT_REVISION = "6328738b83b563ff8396fdb87f8ce130a7b60d74"


def validate_run_args(video: str | Path, query: str, top_k: int) -> Path:
    """Fail on invalid input before allocating a model or touching the index."""
    if not query or not query.strip():
        raise ValueError("query must be nonempty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    path = Path(video).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def dense_results(entry, query_vector, top_k: int) -> list[dict]:
    """Map single-encoder cosine hits to stable keyframe/video metadata."""
    index = entry.index
    hits = index.dense_search(query_vector, "clip", min(top_k, len(index.ids)))
    return [
        {"rank": rank, "frame_id": index.ids[row],
         "video_id": index.video_ids[row], "timestamp": index.timestamps[row],
         "score": score, "image_path": entry.raws[index.ids[row]].image_path}
        for rank, (row, score) in enumerate(hits, 1)
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: dict, video: str | Path, model_id: str,
                      revision: str) -> None:
    """Reject a stale index before searching or loading an encoder."""
    if manifest.get("video", {}).get("sha256") != sha256_file(video):
        raise ValueError("video checksum differs from saved index; use --rebuild")
    model = manifest.get("model", {})
    if model.get("id") != model_id or model.get("revision") != revision:
        raise ValueError("model differs from saved index; use --rebuild")


def probe_video(video: str | Path) -> dict:
    """Read basic source metadata and reject files OpenCV cannot decode."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"video cannot be opened: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError(f"video has no decodable frames: {video}")
        return {"fps": fps, "frame_count": frame_count,
                "duration_s": round(frame_count / fps, 3)}
    finally:
        cap.release()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "faiss-cpu", "opencv-python-headless",
                 "numpy", "rank-bm25"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _real_encoder(model_id: str, revision: str, device: str):
    import torch
    from huggingface_hub import snapshot_download
    from ingestion.embed_siglip import SiglipEncoder

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    snapshot = snapshot_download(repo_id=model_id, revision=revision, max_workers=1)
    return SiglipEncoder(model_name=snapshot,
                         device=None if device == "auto" else device)


def run_baseline(
    video: str | Path, query: str, *, index_dir: str | Path | None = None,
    sample_every_s: float = 1.0, max_frames: int = 64, top_k: int = 5,
    model_id: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION,
    device: str = "auto", rebuild: bool = False, encoder=None,
) -> dict:
    """Index a real MP4 once, then search its saved index with one text encoder."""
    video = validate_run_args(video, query, top_k)
    if not math.isfinite(sample_every_s) or sample_every_s <= 0:
        raise ValueError("sample_every_s must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    index_dir = Path(index_dir or Path("artifacts/indexes") / video.stem).resolve()
    manifest_path = index_dir / "manifest.json"
    existing = manifest_path.exists() and not rebuild
    if existing:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, video, model_id, revision)
        sampling = manifest.get("sampling", {})
        if sampling.get("every_s") != sample_every_s or sampling.get("max_frames") != max_frames:
            raise ValueError("sampling config differs from saved index; use --rebuild")
    else:
        source = probe_video(video)  # reject corrupt input before loading model

    if encoder is None:
        encoder = _real_encoder(model_id, revision, device)
    actual_device = str(getattr(encoder, "device", "test"))

    from retrieval.lite_core import EncoderSpec, LiteRetriever
    from retrieval.video_engine import VideoIndexEntry, VideoSearchEngine

    if existing:
        entry = VideoIndexEntry.load(index_dir)
        index_seconds = manifest["index_seconds"]
    else:
        engine = VideoSearchEngine(
            encoder_names=(model_id,), sample_every_s=sample_every_s,
            max_frames=max_frames, query_templates=("{q}",),
            enable_ocr=False, enable_asr=False, enable_caption=False,
            embed_batch_size=8, decode_backend="cv2", use_gpu_decode=False,
            parallel_index=False,
        )
        engine.set_encoders([encoder])
        started = time.perf_counter()
        entry = engine.index_video(video, index_dir / "frames")
        index_seconds = round(time.perf_counter() - started, 3)
        entry.index.encoder_metadata = {
            "clip": {"id": model_id, "revision": revision}}
        entry.save(index_dir)
        manifest = {
            "schema_version": 1,
            "video": {"video_id": entry.video_id, "source_path": str(video),
                      "sha256": sha256_file(video), "size_bytes": video.stat().st_size,
                      **source},
            "model": {"id": model_id, "revision": revision},
            "sampling": {"every_s": sample_every_s, "max_frames": max_frames,
                         "decode_backend": "cv2", "hist_duplicate_threshold": 0.985},
            "dedup": {"similarity_threshold": 0.97},
            "index": {"type": "faiss_hnsw_inner_product", "encoder_count": 1,
                      "vector_slot": "clip", "hnsw_m": entry.index.config.hnsw_m,
                      "ef_construction": entry.index.config.ef_construction,
                      "ef_search": entry.index.config.ef_search, "seed": None},
            "environment": {"packages": _package_versions(), "device": actual_device},
            "n_sampled": entry.num_sampled, "n_indexed": entry.num_indexed,
            "index_seconds": index_seconds,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    started = time.perf_counter()
    retriever = LiteRetriever(entry, {"clip": EncoderSpec(model_id, revision)})
    results = [hit.to_dict() for hit in retriever.search_text(
        query, {"clip": encoder}, top_k)]
    query_seconds = round(time.perf_counter() - started, 3)
    report = {
        "video_id": entry.video_id, "query": query.strip(),
        "model": manifest["model"], "device": actual_device,
        "index_reused": existing, "index_seconds": index_seconds,
        "query_seconds": query_seconds, "n_sampled": entry.num_sampled,
        "n_indexed": entry.num_indexed, "results": results,
        "manifest_path": str(manifest_path),
    }
    (index_dir / "last-run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewind Lite single-encoder baseline")
    parser.add_argument("video", help="source MP4 path")
    parser.add_argument("query", help="Vietnamese text query")
    parser.add_argument("--index-dir", help="saved index directory")
    parser.add_argument("--every", type=float, default=1.0, help="sample interval (s)")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_baseline(
            args.video, args.query, index_dir=args.index_dir,
            sample_every_s=args.every, max_frames=args.max_frames, top_k=args.top_k,
            model_id=args.model_id, revision=args.revision, device=args.device,
            rebuild=args.rebuild,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
