# P02 — Temporal dataset and interval evaluation protocol

Prepared 2026-11-09 (Asia/Bangkok) for the owner-approved **internal-baseline Phase 2 track**. This protocol defines how future `[t_start,t_end]` predictions will be scored. It does **not** report temporal model accuracy; no Phase 2 interval predictions exist yet. The Phase 1 T06 keyframe JSON, labels and manifest remain unchanged.

## Sources and canonical annotation

`data/annotations/manifest.json` supplies each video's ID, relative path, SHA-256, duration in seconds, FPS, frame count, rights basis and `dev`/`test` split. `data/annotations/labels.json` supplies stable `query_id`, `video_id`, Vietnamese `query`, half-open `time_windows`, split and review status. The P02 adapter maps `query` to `query_vi` and attaches `duration_s` from the manifest **in memory**; it never rewrites the Phase 1 labels. Canonical scoring row:

```json
{"query_id":"vi-dev-001","video_id":"video-test","query_vi":"...","time_windows":[[0.0,23.86]],"duration_s":23.86,"split":"dev"}
```

The T05 validator checks unique IDs, readable sources, checksum and measured duration, rights records, valid nonoverlapping windows, review fields, and video/group split leakage. P02 rejects the dataset when it reports any issue. Rows marked `draft` or `rejected` are excluded and counted as `n_unreviewed`; structurally invalid rows block scoring and are reported as validation errors, never silently dropped. The current files yield **57 evaluable, 0 unreviewed, 0 invalid** rows: 43 dev across three videos and 14 test across two videos. All 57 were drafted and visually reviewed by Codex, so independent adjudication remains open. The two downloaded videos still lack original URL/license terms.

## Prediction contract and metric definitions

Input is a JSON object mapping each `query_id` to a ranked list. An absent query key or `[]` means **no prediction**. Unknown query IDs, wrong video IDs, duplicate/out-of-bounds/zero-length intervals, and nonsequential ranks are errors. Example:

```json
{"vi-dev-001":[{"video_id":"video-test","t_start":2.0,"t_end":8.0,"rank":1}]}
```

Intervals are `[start,end)` seconds with `0 <= start < end <= video_duration`; touching at one boundary has zero overlap. For one predicted interval `P` and one ground-truth interval `G`, `tIoU(P,G) = duration(P∩G) / duration(P∪G)`. For a query with multiple GT windows, each prediction is scored against its **best matching single GT window**. This avoids counting the empty gap between distinct valid moments as positive time. A query is a hit at `R@K, tIoU θ` when any of its first `K` predictions reaches `θ` against at least one GT window.

Report `R@1` and `R@5` at thresholds **0.3** and **0.5** for `dev`, `test` and all queries. `mIoU` is the mean **rank-1** best-GT tIoU, zero if no prediction. The primary denominator is every validated, evaluable query in that split, including no-prediction queries. Per-query output includes each predicted interval, its best tIoU, the four binary hit flags and rank-1 IoU. `n_no_prediction` is reported separately. No oracle best-of-Top-5 mIoU is substituted for rank-1 mIoU. Scores are ranking values, not calibrated probabilities.

The test split is held out by video. P03 thresholds and model choices must be fixed using dev only; P07 will evaluate test once under a declared configuration. The old T06 timestamp Hit@K and the new interval R@K/tIoU measure different events and must not be compared as if they were the same metric.

## Reproduce validation or score a future run

```powershell
.venv\Scripts\python.exe -m evaluation.temporal_metrics --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root .
.venv\Scripts\python.exe -m evaluation.temporal_metrics --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --predictions artifacts\phase2\predictions.json --output artifacts\phase2\scores.json
```

The first command validates and prints dataset counts only. The second requires **real** ranked interval predictions and writes per-query scores plus aggregates; no placeholder predictions are created. `artifacts/phase2/` stays local. Record the source-video and annotation hashes, baseline commit, encoder revision, sampling/dedup, segment thresholds and device with each future benchmark. The existing T06 JSON already records its own frozen inputs and configuration.

## P02 acceptance review

| Criterion | Status | Evidence |
|---|---|---|
| Dataset/manifest source and annotation policy | PASS for internal use | Existing T05 files and validator; adapter preserves originals; rights/review limitations above. |
| Validator rejects malformed intervals, IDs, source mismatches and leakage | PASS | T05 validator reused; P02 test rejects out-of-bounds window. |
| tIoU, R@1/5 at 0.3/0.5, mIoU and multiple GT policy | PASS | `evaluation/temporal_metrics.py`; synthetic boundary, multi-window and no-prediction tests. |
| Per-query output and explicit denominator | PASS | `evaluate_temporal_predictions`; empty prediction counts as zero. |
| Count of queries not evaluable | PASS | Current local gate: 57 evaluable, 0 unreviewed, 0 invalid; invalid input blocks scoring. |
| No fabricated temporal benchmark | PASS | Validation-only command above; no temporal performance numbers reported. |

Known limits: the current labels were reviewed by their author, and many Phase 1 windows are broad. They are adequate for an internal pipeline exercise but require independent review and tighter boundary adjudication before external temporal accuracy claims. P02 does not change sampling, retrieval, labels, T06 output or the live web UI.

Verification on 2026-11-09: full `pytest -o addopts='' -q` — 371 passed, 1 skipped; the validation-only P02 command — 57 evaluable (43 dev, 14 test), 0 unreviewed/invalid; T06 `--check` — READY with 57 verified labels; manifest and label SHA-256 still match T06 JSON; `git diff --check` — exit 0. A real temporal benchmark was not run because P03 has not produced interval predictions. Changed files for P02: `evaluation/temporal_metrics.py`, `tests/test_temporal_metrics_p02.py`, this protocol and `docs/task-status.md`. No changes to the 57 labels, source manifest or Phase 1 inference.
