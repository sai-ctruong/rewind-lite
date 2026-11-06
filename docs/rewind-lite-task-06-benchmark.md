# Rewind Lite — T06 baseline benchmark checkpoint

Date: 2026-11-06 (Asia/Bangkok). T06 partial checkpoint approved for `rewind-lite/main`.

## Protocol prepared

`evaluation/lite_benchmark.py` gates every real run through the T05 source/annotation validator **before** loading a model, building an index, or writing accuracy results. The current manifest has one authorized video and `data/annotations/labels.json` has zero verified rows. The actual command returns `DATA_BLOCKED`, `n_verified=0`, and exit code 3. Thus no E1/E2/E3 accuracy, latency, throughput, or VRAM measurement is claimed for T06.

When the gate reaches at least 50 validated, reviewed Vietnamese rows spanning independent `dev` and `test` videos, the harness will index each video once with both encoders. E1 searches the SigLIP2 slot, E2 the multilingual SigLIP slot, and E3 fuses their rankings with RRF. All three use the same sampled frames and the same index retained after histogram filtering and semantic deduplication. The current semantic dedup uses E1 vectors; this constraint must be considered when interpreting E2 as a standalone model. Explicit manifest `video_id` values are passed to the indexer, so labels need not depend on filenames.

| Baseline | Pinned model and revision | Retrieval |
|---|---|---|
| E1 | `google/siglip2-base-patch16-256@6328738b83b563ff8396fdb87f8ce130a7b60d74` | cosine |
| E2 | `google/siglip-base-patch16-256-multilingual@8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6` | cosine |
| E3 | E1 + E2 | RRF, depth 1000 |

E1 is the revision used by T03. E2 was pinned from the official [multilingual model repository](https://huggingface.co/google/siglip-base-patch16-256-multilingual). The [model card](https://huggingface.co/google/siglip-base-patch16-256-multilingual) describes image/text retrieval and shows `padding="max_length"`; the [SigLIP2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2) specifies max-length padding for its text encoder. The existing encoder applies raw Vietnamese text with `padding="max_length"`, `max_length=64`, and truncation to both models. This is a fixed preprocessing policy, not a claim that one model handles Vietnamese better.

The JSON report, when generated, contains each query's ranked predictions, sampled and searchable positive counts, T02 coverage/outcome, Hit@1, Hit@5, Recall@5, MRR, aggregate dev/test scores, Wilson intervals, paired McNemar comparisons, index duration and throughput, separate text embedding and retrieval time, GPU peak allocated memory, and HNSW recall against exact Top-5 when the index exceeds 512 rows. It records dataset hashes, model commits, source checksums, sampling/dedup/search settings, device and package versions. A Markdown table is written beside the JSON. The test split is reported without parameter tuning. Failure candidates are listed for manual review; language gap, attribute/action confusion, sampling miss, visual ambiguity, and label issue are taxonomy options, not automatically assigned causes. Five to ten verified failure diagnoses remain blocked on actual reviewed queries.

## Commands and current result

```powershell
.venv\Scripts\python.exe -m evaluation.lite_benchmark --check
.venv\Scripts\python.exe -m evaluation.lite_benchmark --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --output artifacts\benchmarks\t06.json --device cuda --every 1 --top-k 5
```

Both commands currently stop before model loading with `DATA_BLOCKED` and exit code 3. Outputs under `artifacts/` are ignored by Git. After T05 is complete, run the second command on the pinned environment, inspect per-query predictions and source videos, and manually adjudicate failure reasons before drawing a conclusion. Do not use the held-out test split for tuning. If fewer than 5 failures occur, report the actual count rather than inventing cases.

## Acceptance review

| Criterion | Status | Evidence |
|---|---|---|
| E1/E2/E3 same verified dataset, JSON/Markdown report | BLOCKED | Harness prepared; T05 gate has 0 verified queries |
| T02 metrics and per-query missing/coverage accounting | PASS as tooling | `evaluation/lite_benchmark.py`; synthetic regression tests only |
| Measured index/query/GPU latency and throughput | BLOCKED | Instrumentation prepared; no valid real run |
| Error taxonomy and 5–10 checked failure cases | PARTIAL | Taxonomy and candidate export prepared; human diagnoses await data |
| Confidence/paired comparison and untouched test | PASS as tooling | Wilson/McNemar fields and fixed config; no outcome claim |

T06 remains `PARTIAL / DATA BLOCKED`. No retrieval accuracy or model ranking is stated from the current video or synthetic tests.
