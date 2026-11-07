# Rewind Lite — Task Status

Current branch: `main`; push target: `rewind-lite/main`.

T03 implementation commit: `1172c08`; checkpoint `4397d5e` is on the Rewind Lite default branch `main` and `refactor/rewind-lite-t03`.

T04 implementation commit: `412a6cd`, pushed to `main` and `refactor/rewind-lite-t04` on 2026-11-06.

Push remote `rewind-lite`: `https://github.com/sai-ctruong/rewind-lite.git`. The original `origin` remains `https://github.com/sai-ctruong/tlcn-rewind.git`.

Active task: T05 dataset partial checkpoint approved by the owner for `rewind-lite/main`; 2 source videos and 14 draft labels have Codex visual review. T09 hardening/report partial checkpoint was pushed to `rewind-lite/main` as `175d301`. T05 remains data blocked until independent label review; T06 has no measured accuracy.

| Task | Status | Evidence |
|---|---|---|
| T01 | DONE (audit only) | `docs/rewind-lite-task-01-audit.md` |
| T02 | DONE | `2e59f03`, checkpoint `3eced18`; `docs/rewind-lite-task-02-protocol.md` |
| T03 | DONE | `1172c08`; CUDA check, real `video-test.mp4` index/search and saved-index reuse; `docs/rewind-lite-task-03-environment.md`; 307 passed, 1 skipped |
| T04 | DONE | `412a6cd`; `docs/rewind-lite-task-04-core.md`; Lite contracts, versioned index/entry, import boundary and regression tests |
| T05 | PARTIAL / REVIEW BLOCKED; checkpoint approved | `docs/rewind-lite-task-05-dataset.md`; 2 authorized videos in `dev`/`test`, 14 draft queries, 0 verified; validator and contact sheets |
| T06 | PARTIAL / DATA BLOCKED | `docs/rewind-lite-task-06-benchmark.md`; E1/E2/E3 harness and synthetic tests checkpoint on `main`; no real accuracy result |
| T07 | DONE / pushed to `main` | `96b5c91`; `ui/lite_app.py`, `tests/test_lite_api_t07.py`, `docs/rewind-lite-task-07-api.md`; real `video-test.mp4` API smoke passed |
| T08 | DONE / pushed to `main` | `4a87459`; `docs/rewind-lite-task-08-ui.md`; real SigLIP2 browser flow and 1440/768/390 screenshots |
| T09 | PARTIAL / DATA BLOCKED; pushed to `main` | `175d301`; `docs/rewind-lite-task-09-release.md`; README, demo and release checklist; real CLI/API smoke; accuracy and error analysis await T05/T06 |

Known limit: only two keyframes survived the duplicate filters on the original `video-test.mp4`; no verified labels, so no retrieval accuracy claim. Both source videos, model cache, and indexes are local ignored artifacts.

Next action: independently review the 14 draft query/window pairs against source playback, record reviewer and date, then add more diverse verified data toward the 50-query gate. Run the pinned T06 benchmark only after T05 is ready. Do not claim an academic release until those gates pass.
