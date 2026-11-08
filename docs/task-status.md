# Rewind Lite — Task Status

Current branch: `main`; push target: `rewind-lite/main`.

T03 implementation commit: `1172c08`; checkpoint `4397d5e` is on the Rewind Lite default branch `main` and `refactor/rewind-lite-t03`.

T04 implementation commit: `412a6cd`, pushed to `main` and `refactor/rewind-lite-t04` on 2026-11-06.

Push remote `rewind-lite`: `https://github.com/sai-ctruong/rewind-lite.git`. The original `origin` remains `https://github.com/sai-ctruong/tlcn-rewind.git`.

Latest task: T09 release reconciliation on `main`, approved and pushed on 2026-11-08 after T06 commit `bc6fb22`. Five authorized videos and 57 Vietnamese labels have Codex visual review; T06 measured E1/E2/E3 and checked seven E3 misses against source frames. The same agent drafted and reviewed the labels, so independent human adjudication remains necessary for external accuracy claims.

| Task | Status | Evidence |
|---|---|---|
| T01 | DONE (audit only) | `docs/rewind-lite-task-01-audit.md` |
| T02 | DONE | `2e59f03`, checkpoint `3eced18`; `docs/rewind-lite-task-02-protocol.md` |
| T03 | DONE | `1172c08`; CUDA check, real `video-test.mp4` index/search and saved-index reuse; `docs/rewind-lite-task-03-environment.md`; 307 passed, 1 skipped |
| T04 | DONE | `412a6cd`; `docs/rewind-lite-task-04-core.md`; Lite contracts, versioned index/entry, import boundary and regression tests |
| T05 | PARTIAL / PROGRAM GATE READY; human adjudication pending | `docs/rewind-lite-task-05-dataset.md`; 5 owner-supplied videos in `dev`/`test`, 57 Codex-reviewed queries, 0 validator issues, `ready_for_accuracy=true`; no human review |
| T06 | MEASURED / INTERNAL BENCHMARK; pushed to `main` | `docs/rewind-lite-task-06-benchmark.md`; `docs/benchmark-results/t06.json`; 57 queries, 407 sampled / 215 indexed frames, E1/E2/E3 scores, timing and 7 checked E3 misses; independent label review pending |
| T07 | DONE / pushed to `main` | `96b5c91`; `ui/lite_app.py`, `tests/test_lite_api_t07.py`, `docs/rewind-lite-task-07-api.md`; real `video-test.mp4` API smoke passed |
| T08 | DONE / pushed to `main` | `4a87459`; `docs/rewind-lite-task-08-ui.md`; real SigLIP2 browser flow and 1440/768/390 screenshots |
| T09 | INTERNAL HANDOFF READY / UPDATE PUSHED TO `main` | Original checkpoint `175d301` on `main`; updated `docs/rewind-lite-task-09-release.md` and README reconcile T06 metrics, demo, tests and release checklist; independent label review still pending for academic claims |

Phase 2 P01: owner approved the internal-baseline handoff and `TemporalMoment` contract on 2026-11-09; checkpoint `3ef26ba` was pushed to `rewind-lite/main`. No new branch. Independent review/source-license gates remain open for external accuracy claims. The owner approved the P02 temporal dataset/metrics implementation and protocol for the `main` checkpoint on 2026-11-09 (`evaluation/temporal_metrics.py`, `docs/phase2-evaluation-protocol.md`); no temporal benchmark result or label change. The owner's standing instruction is to work on `main`.

Owner-approved search-quality checkpoint: the demo's old 64-frame cap indexed only the first 64 seconds of the 2037-second `walk-bangkok.mp4`. The web now spreads a 512-frame budget over the whole video and retains more subtle motion. The rebuilt local indexes have 510 frames spanning 0–2036 seconds for `walk-bangkok` and 17 searchable frames for `video-test` (previously 2). T06 and T03 measurements still describe the original CLI/benchmark configuration; no new accuracy claim has been made. The remaining Top-1 semantic error is recorded in `docs/lite-search-quality.md`. Source videos, model cache and indexes are local ignored artifacts; the compact T06 JSON remains tracked under `docs/benchmark-results/`.

Next action: owner tests the rebuilt local web demo with specific queries and expected scenes. For an academic accuracy claim, obtain independent human adjudication of the 57 labels, reproduce T06 and verify the two downloaded videos' source/license details.
