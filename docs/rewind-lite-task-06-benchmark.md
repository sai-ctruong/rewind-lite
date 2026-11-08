# Rewind Lite — T06 measured baseline benchmark

Measured 2026-11-08 (Asia/Bangkok) on five owner-supplied videos and 57 Vietnamese queries. The dataset validator accepted all 57 labels. Codex drafted and visually reviewed those labels, so these are **internal measurements**, not independently adjudicated accuracy claims. The [full JSON report](benchmark-results/t06.json) contains config, input hashes, every ranked prediction and timing; the [generated Markdown table](benchmark-results/t06.md) is a compact companion.

## Fixed protocol

- E1: `google/siglip2-base-patch16-256@6328738b83b563ff8396fdb87f8ce130a7b60d74`.
- E2: `google/siglip-base-patch16-256-multilingual@8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6`.
- E3: E1 and E2 fused with reciprocal rank fusion, depth 1000.
- Same 1 second sampling, histogram duplicate threshold 0.985, E1 semantic dedup threshold 0.97, five videos and T02 scoring for all baselines. Raw Vietnamese text uses max length 64 and max length padding. E1 drives semantic dedup, so E2 uses the same retained frames instead of an independently deduplicated E2 index.
- Dev has 43 queries across three videos; held-out test has 14 queries across two videos. No setting was tuned on test. The 215-row index uses exact dense search (threshold 512), so HNSW versus exact recall does not apply.

Run from the repository root in the pinned GPU environment:

```powershell
.venv\Scripts\python.exe -m evaluation.lite_benchmark --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --output artifacts\benchmarks\t06.json --device cuda --every 1 --top-k 5
```

The local run set `HF_HUB_CACHE=D:\Rewind\artifacts\hf-cache\hub`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after caching both pinned model snapshots. The JSON records model revisions, source checksums, input hashes, packages and command. The first attempt failed with Windows `os error 1455` during model loading; loading E1 and E2 individually succeeded, and an unchanged retry completed. No accuracy output was saved from the failed attempt.

## Results

| Split | Baseline | Coverage | Hit@1 | Hit@5 | MRR |
|---|---|---:|---:|---:|---:|
| Dev (43) | E1 | 42/43 | 24/43 (55.8%) | 33/43 (76.7%) | 0.645 |
| Dev (43) | E2 | 42/43 | 29/43 (67.4%) | 37/43 (86.0%) | 0.749 |
| Dev (43) | E3 | 42/43 | 29/43 (67.4%) | 36/43 (83.7%) | 0.745 |
| Test (14) | E1 | 14/14 | 12/14 (85.7%) | 13/14 (92.9%) | 0.871 |
| Test (14) | E2 | 14/14 | 10/14 (71.4%) | 13/14 (92.9%) | 0.810 |
| Test (14) | E3 | 14/14 | 11/14 (78.6%) | 14/14 (100%) | 0.863 |

The index sampled 407 frames and retained 215. Index construction took 47.744 s (8.52 sampled frames/s), after 17.403 s to load models. Peak PyTorch GPU allocation during indexing and queries was 1,596,314,624 bytes on an RTX 3060 Laptop GPU. Per query, E1 used 14.83 ms for text embedding plus 0.88 ms retrieval; E2 13.11 + 0.72 ms; E3 27.93 + 1.50 ms. These timings are this single local run, excluding model loading and video indexing from query latency.

The JSON includes Wilson 95% intervals for Hit@1 and paired McNemar tests for Hit@5. None of the paired comparisons reached p < 0.05 (dev p ≥ 0.289; test p = 1.0). With 14 test queries, related scenes and same-agent label review, the observed ordering does not establish which model is generally better.

## Checked E3 failures

All seven E3 Hit@5 misses were compared with sampled source frames and the top five predictions. Reasons describe the visible mismatch; they do not infer model internals. Timestamps below are seconds.

| Query | Source event | Top predictions | Checked reason |
|---|---|---|---|
| `vi-dev-012` | `video_test3` 3.25–14, person holding a clear glass bottle | `video_test2` 12–22, person handling a clear plastic bottle | Bottle material and scene confused; E2 alone retrieved the source at rank 1. |
| `vi-dev-014` | `video_test3` 6–9, person drinking from glass bottle | `video_test2` 19–22, person handling a plastic bottle | Returned frames show a bottle but not drinking; E2 alone hit the source. |
| `vi-dev-015` | `video_test3` 8.75–9.75, person lowering bottle | No source frame in that window; nearby extracted frames at 7 and 10 | Sampling/histogram filtering left zero positive frames. This is an index coverage failure. |
| `vi-dev-029` | `video-cook` 120–122.5, white cream poured from a can into soup | Same video 102 and 111–114, earlier white liquid/broth pouring | Visually similar pours at different recipe stages caused a temporal event mismatch. |
| `vi-dev-040` | `video-cook` 327–331, fried bacon sprinkled over plated soup | Same video 35–51, bacon frying in a pan | Same ingredient, different action and setting. |
| `vi-dev-041` | `video-cook` 334–339, green herbs sprinkled over plated soup | Same video 294–316, soup being plated before garnish | Similar bowl/food appearance, wrong preparation stage and no visible herb sprinkling. |
| `vi-dev-043` | `video-cook` 345–349 or 365.5–369.5, spoon lifting soup | Same video 292–302, soup being plated | Similar plated soup, but spoon lifting is absent. |

The taxonomy is `language_gap`, `attribute_action_confusion`, `sampling_miss`, `visual_ambiguity` and `label_issue`. The checked evidence supports attribute/action, temporal/visual and sampling errors here; it does not demonstrate a language gap. Source windows were left unchanged after viewing model output.

## Acceptance and limits

T06 has a measured E1/E2/E3 comparison, per-query predictions, coverage accounting, timing, GPU peak and seven checked failures. HNSW recall does not apply to the 215-row exact-search index. An independent reviewer should replay the five videos and adjudicate the 57 labels and time windows before external or academic accuracy claims. The two downloaded videos also lack original URL/license details in the manifest.

## Task T06 — Acceptance Review

| Criterion | Status | Evidence |
|---|---|---|
| E1/E2/E3 on one dataset and protocol with JSON/Markdown reports | PASS | [Full report](benchmark-results/t06.json), [generated table](benchmark-results/t06.md); pinned command and config above |
| T02 metrics, per-query predictions and missing/index coverage | PASS | 57 rows per baseline in JSON; 42/43 dev and 14/14 test covered |
| Measured latency, throughput and GPU peak | PASS | 47.744 s index, separate embedding/retrieval time, 1,596,314,624 peak allocated bytes |
| Error taxonomy and 5–10 checked failures | PASS | Seven visually checked E3 failures in the table above |
| Confidence, paired comparison and untouched test split | PASS | Wilson intervals and McNemar results in JSON; fixed model/sampling/dedup settings |

Tests run: dataset validator `--require-ready` (57 verified, 0 issues); full pytest suite (353 passed, 1 skipped); `git diff --check` (exit 0). Tests not run: optional 0.5/2 second sampling ablation, because it is not an acceptance gate; HNSW recall, because retrieval used exact search. Files changed: `README.md`, this report, `docs/task-status.md`, `docs/rewind-lite-task-09-release.md`, `docs/benchmark-results/t06.json` and `docs/benchmark-results/t06.md`. Risks: same-agent label review, two videos without source URLs/license terms, small test split and one transient Windows memory-loading error. Scope drift: T09 received a cross-reference to this later benchmark; no T09 release claim changed. Decision: the owner approved T06 commit and push to `rewind-lite/main` on 2026-11-08.
