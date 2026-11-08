# Rewind Lite — Task Status

Current branch: `main`; push target: `rewind-lite/main`.

T03 implementation commit: `1172c08`; checkpoint `4397d5e` is on the Rewind Lite default branch `main` and `refactor/rewind-lite-t03`.

T04 implementation commit: `412a6cd`, pushed to `main` and `refactor/rewind-lite-t04` on 2026-11-06.

Push remote `rewind-lite`: `https://github.com/sai-ctruong/rewind-lite.git`. The original `origin` remains `https://github.com/sai-ctruong/tlcn-rewind.git`.

Latest task: T06 measured benchmark on `main`, approved and pushed on 2026-11-08. Five authorized videos and 57 Vietnamese labels have Codex visual review; the validator and T06 `--check` return ready. E1/E2/E3 were measured on these labels and seven E3 failures were checked against source frames. The same agent drafted and reviewed the labels, so independent human adjudication is still needed for external claims. The preceding T05 checkpoint was pushed to `rewind-lite/main` as `caeead0`.

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
| T09 | PARTIAL / RELEASE REVIEW PENDING; pushed to `main` | `175d301`; `docs/rewind-lite-task-09-release.md`; README, demo and release checklist; T06 internal report exists, independent label review and release reconciliation pending |

Known limit: only two keyframes survived the duplicate filters on the original `video-test.mp4`; the measured T06 result depends on same-agent reviewed labels and five videos. Source videos, model cache and indexes are local ignored artifacts; the compact report JSON is tracked under `docs/benchmark-results/`.

Next action: for an academic release, obtain independent human adjudication of the 57 labels, reproduce the report and update T09 release evidence.
