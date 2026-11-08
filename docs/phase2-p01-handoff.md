# P01 — Phase 1 handoff and temporal design review

Prepared 2026-11-08 and approved for an **internal-baseline Phase 2 track** by the owner on 2026-11-09 (Asia/Bangkok). This handoff makes no Phase 2 runtime changes. The strict academic release gate in `REWIND_LITE_IMPROVEMENT_PLAN.md` §0/§4.1 remains open: `docs/task-status.md` records T05 as partial and T09 as ready for internal handoff, while independent label review and source/license details for two downloaded videos remain missing. The plan names TLCN-Rewind/`origin` as its handoff target; the owner instead chose `rewind-lite/main` and instructed work directly on `main`, overriding the proposed Phase 2 branch. P02 may proceed as internal work without treating Phase 1 as academically released.

## Frozen Phase 1 reference

| Item | Evidence and interpretation |
|---|---|
| Runtime/checkpoint | `21b87b05f20fc370f401a60824577fb7e0af3395` on `rewind-lite/main`; Phase 1 keyframe API/CLI and web quality fix are callable. This is the code reference for regression checks. |
| Measured evaluation | [`benchmark-results/t06.json`](benchmark-results/t06.json), 57 Codex-reviewed Vietnamese queries on five owner-supplied videos, split by video (43 dev, 14 test). It is an internal benchmark, not independent adjudication. |
| Input identity | Manifest SHA-256 `143e5cfe8d6da5541b789826c9fd2dc603df91e416de9fefc0b48086f2d2b468`; labels SHA-256 `73f1b72e3c6ba6960980b3ee4e83030678eb60e411b5dc22e9272999aa5874da`. Video hashes are in the JSON report and `data/annotations/manifest.json`. |
| E1 | `google/siglip2-base-patch16-256@6328738b83b563ff8396fdb87f8ce130a7b60d74`; proposed frozen encoder for the first temporal-only baseline. |
| E2/E3 comparators | `google/siglip-base-patch16-256-multilingual@8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6`; E3 is E1/E2 reciprocal-rank fusion. Keep as measured comparators, not a proven upgrade. |
| Measured T06 sampling | 1 second, no frame cap, histogram threshold 0.985, E1 semantic dedup 0.97, raw Vietnamese text at max length 64, exact search for at most 512 index rows. 407 sampled / 215 indexed frames. |
| E1 measured scores | Dev Hit@1 24/43, Hit@5 33/43; test Hit@1 12/14, Hit@5 13/14. These are **timestamp** metrics and cannot be read as temporal IoU. |
| Live web configuration | 512-frame budget distributed across the video, histogram 0.999, semantic dedup 0.99. This differs from measured T06; a future temporal benchmark must log which configuration produced each prediction. |

The Phase 1 demo and T06 report provide a reproducible **internal** baseline. The T09 release checklist does not verify clean installation from a fresh clone, independent annotation review, or source URLs/license terms for `video_test2` and `video_test3`. Those facts remain open; P01 does not rewrite their status.

## Current retrieval interfaces and integration points

| Existing interface | Phase 2 use |
|---|---|
| `RawKeyframe` / `VideoIndexEntry` in `ingestion/schemas.py` and `retrieval/video_engine.py` | Read `video_id`, `frame_id`, timestamp and source frame reference. `entry.raws` includes sampled frames before semantic dedup; `entry.index.ids` identifies searchable frames. Temporal scoring must use the latter and report sampling gaps. |
| `LiteRetriever.search_text` and `SearchResult` in `retrieval/lite_core.py` | Keep the keyframe Top-K contract unchanged. It returns only ranked hits, not a complete ordered frame-score series; temporal proposal needs a separate read-only all-frame scoring path over the same encoder/index. |
| `evaluation/dataset.py` and T02 protocol | Reuse source checksum, split, label validity and half-open `time_windows`. P02 must define interval-specific validation and scoring without changing T06 labels or metrics. |
| `/api/lite/search` in `ui/lite_app.py` | Keep its timestamp response unchanged. A later temporal endpoint may return intervals, but P01 adds no route and no UI behavior. |

Proposed path: `retrieval/temporal.py` for the moment contract and temporal-only proposal in P03; `evaluation/temporal_metrics.py` for P02 metrics. A later API adapter can map `TemporalMoment` to JSON. Detector output remains optional evidence and must not be a dependency of temporal-only inference.

## Proposed `TemporalMoment` contract

```python
@dataclass(frozen=True)
class TemporalMoment:
    video_id: str
    t_start: float
    t_end: float
    rank: int
    score: float
    evidence_frame_ids: tuple[str, ...]
    score_type: str = "temporal_relevance"
    detections: tuple[dict, ...] | None = None
```

Validation for P03: `video_id` is nonempty and matches the selected source; all times and score are finite; `0 <= t_start < t_end <= video_duration` within a documented floating tolerance; rank starts at 1 with no gaps; evidence IDs exist in the searchable index, belong to that video and lie in or next to the proposed interval according to a documented boundary policy. Results are ordered by descending score with deterministic tie-breaks and no duplicate interval. `score` is a ranking value, **not** a calibrated probability. `detections=None` means detector was not run; an empty tuple means it ran and found no object. No bbox or class may be inferred from similarity alone.

Interval convention: `[t_start, t_end)` in seconds, matching the existing label windows. `t_end` may equal the measured video duration. The contract fixes output validity, not the P03 smoothing, threshold, merging or boundary algorithm; those choices and their defaults must be recorded and tuned on dev only. Frame sampling cannot by itself prove exact action start/end.

## Test and evaluation plan for the next tasks

1. **P02 data/metrics:** keep stable `query_id`, `video_id`, Vietnamese query, `time_windows`, source duration and dev/test group. Validate finite half-open windows, bounds, overlaps, missing video/IDs and no prediction. Define tIoU for one predicted interval against the best matching GT window, then R@1/5 at tIoU 0.3/0.5 and mean IoU with all valid queries in the denominator. Write synthetic tests for exact overlap, disjoint, boundary touch, zero-length, multiple GT windows and empty predictions. Preserve original T06 artifacts.
2. **P03 temporal-only:** obtain a score for **every searchable frame** in one video, order by `(timestamp, frame_id)`, and propose intervals using configurable smoothing/threshold/merge/clipping. Test irregular gaps, duplicate timestamps, zero candidates, first/last-frame edges, deterministic ties, overlapping proposals and `duration` bounds. Compare per-query predictions on real video when the P02 gate permits; do not tune on the held-out test split.
3. **Phase 1 regression:** run the existing Lite core/API tests and a real keyframe search smoke test at the frozen commit/config before accepting a new temporal path. Detector OFF must yield identical temporal-only outputs to a build without detector code.

## P01 acceptance review

| Criterion | Status | Evidence / remaining work |
|---|---|---|
| Phase 1 reference commit, config, metrics and limits recorded | PASS for internal baseline | Commit and T06 JSON above; T09 release checklist explicitly limits claims. |
| Current retrieval/schema/evaluation/API interfaces audited | PASS | Interface table above, linked to concrete source files. |
| Temporal contract and integration design | PASS for internal track | Owner approved continuation on 2026-11-09; contract/invariants above are the P02/P03 design reference. |
| Phase 1 fully DONE per strict academic gate | OPEN | Owner authorized an internal-baseline track; independent label review and two source/license records remain open. |
| Single Phase 2 branch | SUPERSEDED BY OWNER | Owner instructed no new branches and direct work on `main`. No branch was created. |
| Phase 1 behavior unchanged | PASS for tested path | P01 changes only this design note and the checkpoint; full pytest and real Lite search smoke passed. |

Decision: owner approved proceeding with the internal baseline on 2026-11-09. This freezes the contract for internal P02/P03 work; external accuracy claims still require the independent review/provenance gates.

Checks run on 2026-11-08: `.venv\Scripts\python.exe -m pytest -o addopts='' -q` — 356 passed, 1 skipped; `evaluation.dataset --require-ready` — 57 verified, 0 issues; `evaluation.lite_benchmark --check` — READY, 57 verified; SHA-256 of current manifest/labels matches T06 JSON; local Flask `/api/lite/search` on `video-test` returned three timestamped results; `git diff --check` — exit 0. Not run: a new GPU benchmark or temporal metrics, because P01 changes no inference or evaluation code. The Phase 1 release limitations above remain.
