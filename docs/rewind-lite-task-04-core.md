# Rewind Lite — T04 core contracts and compatibility review

Date: 2026-11-06 (Asia/Bangkok). Branch: `refactor/rewind-lite-t04`.

## Decision and contracts

T04 uses an adapter over the existing `KeyframeIndex` instead of migrating stored vectors. Its `clip` and `siglip` names remain storage slots; the T03 SigLIP2-only baseline occupies `clip`. `EncoderSpec` carries the actual model ID and revision separately, so a slot name does not misidentify a model. The legacy `VideoSearchEngine` is retained for ingestion and advanced retrieval; the CLI's basic search now uses `LiteRetriever`.

| Contract | Responsibility |
|---|---|
| `LiteKeyframe` | Stable frame ID, video ID, timestamp, optional image path |
| `Encoder` protocol | Image and text embeddings from the same encoder |
| `VectorIndex` protocol | Slot availability/dimension, exact and HNSW cosine search, row metadata |
| `Retriever` protocol | Nonempty text query to ranked results |
| `SearchResult` | Frame/video/timestamp, rank, score, `score_kind`, image path, model revisions and source ranks |

One active slot returns a **cosine** score. Two active slots use RRF and return an **rrf** score with each source rank. The RRF depth is fixed at 1,000 independently of requested `top_k`. Indexes with at most 512 rows use exact cosine search with stable frame-ID tie ordering; larger indexes use the existing Faiss HNSW graph and sort returned ties by ID. These thresholds are structural defaults, not measured quality claims.

`KeyframeIndex.build` rejects duplicate IDs, invalid timestamps, mixed embedding dimensions, zero vectors and NaN/Inf vectors before Faiss receives them. Query vectors get the same dimension/finite/nonzero checks. After deduplication, results use the index's retained row metadata, not the original sampling positions.

## Persistence and Lite import boundary

`meta.pkl` and `entry.pkl` now record `schema_version=1`. Files without that field are treated as version 0 for T03/legacy compatibility. Unknown future versions raise an explicit error. New indexes also save optional encoder model/revision metadata; callers may require matching model identities and vector dimensions when loading. The old T03 index has no embedded model metadata, so its CLI manifest continues to validate the pinned model/revision before loading. Pickle files are local trusted artifacts and must not be loaded from untrusted sources.

Importing the Lite core and `video_engine` no longer imports fine rerank, temporal checking, VLM rerank, dialogue or query understanding. Advanced features load their modules only when called. Basic Lite search uses dense cosine/RRF only and does not invoke OCR, ASR, captioning, VLM or Agent code.

## Verification

```powershell
.venv\Scripts\python.exe -m pytest -o addopts='' tests\test_lite_core_t04.py tests\test_lite_cli_t03.py tests\test_lite_cli_integration_t03.py tests\test_retrieval_phase3.py tests\test_video_engine.py -q
.venv\Scripts\python.exe -m pytest -o addopts='' -q
$env:HF_HUB_OFFLINE = '1'
.venv\Scripts\python.exe -m scripts.lite_search video-test.mp4 'người đang đi bộ' --index-dir artifacts/indexes/video-test --every 1 --max-frames 64 --top-k 5 --device cuda
```

Targeted tests: 83 passed. Full suite: 327 passed, 1 skipped (a legacy test conditioned on torch availability). The real T03 index was loaded without reembedding: `index_reused=true`, `device=cuda`, `score_kind=cosine`, and Top-2 timestamps 5.0 and 0.0 seconds. The source video has no verified relevance labels; these are smoke results, not accuracy measurements.

Known limits: old version-0 indexes lack saved model identity, so their external manifest must remain available for model checks. HNSW on larger indexes is approximate, including ties near the Top-K boundary. No retrieval-quality change is claimed from this refactor.
