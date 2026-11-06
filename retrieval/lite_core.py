"""Small Lite retrieval contract over the existing CLIP/SigLIP Faiss slots.

The slot names are storage adapters, not claims about the underlying model: a
single SigLIP2 encoder may occupy ``clip`` in a T03 index. Model identity is
carried separately by EncoderSpec and the persisted manifest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

import numpy as np

from retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from ingestion.schemas import RawKeyframe
    from retrieval.video_engine import VideoIndexEntry


@dataclass(frozen=True)
class LiteKeyframe:
    frame_id: str
    video_id: str
    timestamp: float
    image_path: str | None = None


@runtime_checkable
class Encoder(Protocol):
    def embed(self, raw: RawKeyframe) -> np.ndarray: ...
    def encode_text(self, text: str) -> np.ndarray: ...


@runtime_checkable
class VectorIndex(Protocol):
    ids: list[str]
    video_ids: list[str]
    timestamps: list[float]

    def has_encoder(self, encoder: str) -> bool: ...
    def dimension(self, encoder: str) -> int: ...
    def dense_search(self, query_vec: np.ndarray, encoder: str,
                     top_k: int) -> list[tuple[int, float]]: ...
    def exact_dense_search(self, query_vec: np.ndarray, encoder: str,
                           candidate_rows: list[int], top_k: int
                           ) -> list[tuple[int, float]]: ...


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    revision: str

    @property
    def version(self) -> str:
        return f"{self.model_id}@{self.revision}"


@dataclass(frozen=True)
class SearchResult:
    frame_id: str
    video_id: str
    timestamp: float
    score: float
    rank: int
    image_path: str | None
    score_kind: str  # cosine for one encoder; RRF for multiple encoders
    encoder_versions: dict[str, str]
    source_ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class Retriever(Protocol):
    def search_text(self, query: str, encoders: Mapping[str, Encoder],
                    top_k: int = 5) -> list[SearchResult]: ...


class LiteRetriever:
    """Dense cosine for one slot, RRF for two; no sparse/rerank imports.

    Exact search for small indexes keeps ties stable and avoids HNSW overhead.
    Larger indexes use the persisted HNSW graph, then sort returned ties by ID.
    """

    def __init__(self, entry: VideoIndexEntry,
                 encoder_specs: Mapping[str, EncoderSpec], *,
                 exact_threshold: int = 512, fusion_depth: int = 1000):
        self.entry = entry
        self.encoder_specs = dict(encoder_specs)
        saved_models = getattr(entry.index, "encoder_metadata", {})
        for slot, spec in self.encoder_specs.items():
            saved = saved_models.get(slot)
            if saved is not None and saved != {"id": spec.model_id,
                                                "revision": spec.revision}:
                raise ValueError(f"Index model mismatch for {slot}; rebuild index")
        self.exact_threshold = exact_threshold
        self.fusion_depth = fusion_depth

    def search_text(self, query: str, encoders: Mapping[str, Encoder],
                    top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be nonempty")
        if not encoders:
            raise ValueError("query requires at least one encoder")
        vectors = {slot: encoder.encode_text(query.strip())
                   for slot, encoder in encoders.items()}
        return self.search_vectors(vectors, top_k)

    def search_vectors(self, query_vectors: Mapping[str, np.ndarray],
                       top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not query_vectors:
            raise ValueError("query requires at least one vector")
        index = self.entry.index
        if not index.ids:
            return []
        ranked: dict[str, list[tuple[int, float]]] = {}
        for slot, vector in query_vectors.items():
            if slot not in self.encoder_specs or slot not in {"clip", "siglip"}:
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
            if norm == 0 or not np.isfinite(norm):
                raise ValueError("query vector must be nonzero and finite")
            q = q / norm
            search_k = min(len(index.ids),
                           self.fusion_depth if len(query_vectors) > 1 else top_k)
            if len(index.ids) <= self.exact_threshold:
                pairs = index.exact_dense_search(q, slot, list(range(len(index.ids))),
                                                 len(index.ids))
            else:
                pairs = index.dense_search(q, slot, search_k)
            ranked[slot] = sorted(pairs, key=lambda pair: (-pair[1], index.ids[pair[0]]))

        if len(ranked) == 1:
            slot, pairs = next(iter(ranked.items()))
            rows = [(row, score, {slot: rank})
                    for rank, (row, score) in enumerate(pairs, 1)][:top_k]
            score_kind = "cosine"
        else:
            fused = reciprocal_rank_fusion(
                {slot: [row for row, _ in pairs] for slot, pairs in ranked.items()})
            rows = [(row, score, ranks) for row, (score, ranks) in fused.items()]
            rows.sort(key=lambda item: (-item[1], index.ids[item[0]]))
            rows = rows[:top_k]
            score_kind = "rrf"

        versions = {slot: self.encoder_specs[slot].version for slot in ranked}
        return [
            SearchResult(
                frame_id=index.ids[row], video_id=index.video_ids[row],
                timestamp=index.timestamps[row], score=score, rank=rank,
                image_path=self.entry.raws[index.ids[row]].image_path,
                score_kind=score_kind, encoder_versions=versions,
                source_ranks=ranks,
            )
            for rank, (row, score, ranks) in enumerate(rows, 1)
        ]
