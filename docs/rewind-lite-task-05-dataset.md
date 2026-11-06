# Rewind Lite — T05 dataset preparation and annotation protocol

Date: 2026-11-06 (Asia/Bangkok). Partial checkpoint target: `rewind-lite/main`.

## Current evidence

`data/annotations/manifest.json` records the one local source video supplied by the repository owner: `video-test.mp4`. The owner confirmed permission to use it for evaluation. Its measured metadata are 1,193 frames at 50 FPS (23.86 seconds), with SHA-256 `8410556fea9c755db18ebd29316405065e4330af52aa21530604bb72abdc448c`. The video remains local and ignored by Git. The manifest assigns it to `dev`; a single video cannot provide an independent video-level `test` split.

`data/annotations/labels.json` is an empty array: **0 annotated, 0 verified Vietnamese queries**. The 78 rows in `evaluation/labels.json` and 51 rows in `evaluation/labels_en.json` remain unverified legacy material because their source videos are not present. They are not included in T05 counts. The owner has said more authorized data are available and will send paths; those files have not yet been inspected.

The timestamped contact sheet at `artifacts/contact-sheets/video-test.jpg` was generated locally with 24 samples at one-second spacing and visually inspected. It is an ignored review aid, not a committed dataset or proof that any proposed query/window is correct. No retrieval accuracy is reported.

## Format and review workflow

Each manifest video has a stable `video_id`, a path relative to the repository root, SHA-256, measured `duration_s`, `fps`, `frame_count`, a source description, confirmed rights with a basis, a `dev` or `test` split, and a `group_id` for related scenes. Store source videos locally; only JSON annotations and metadata are tracked. Recalculate the checksum and metadata when a file changes.

Each label has `query_id`, `video_id`, a Vietnamese `query`, `language: "vi"`, one or more half-open `time_windows` (`[start, end)` in seconds), `category`, `split`, `paraphrase_group`, `annotator`, `review_status`, `reviewer`, and `reviewed_at`. Allowed categories are `object`, `color`, `action`, `count`, `context`, and `short_event`; review statuses are `draft`, `verified`, and `rejected`. Only `verified` rows with valid source and review metadata contribute to the completion count. The language declaration is checked as metadata; a human must check the Vietnamese wording and whether it actually describes the video.

1. Receive each video with provenance and usage permission; record them in the manifest and verify the file, checksum, frame rate, frame count, and duration.
2. Generate a contact sheet, then inspect each candidate segment and its boundaries in the original video. When the frame cap is reached, samples are spread across the whole video. At one-second grid spacing, a brief event can still be missed; use a shorter interval and playback around it. Never infer a narrow window from a single thumbnail.
3. Draft natural Vietnamese queries from observed content. Include objects, colors, actions, counts, context, and short events where the sources genuinely support them. Record all distinct appearances as separate windows. Do not manufacture category balance from one scene.
4. Have a reviewer inspect the source video and each window independently, resolve uncertain or overlapping descriptions, then set `review_status` to `verified` and record reviewer/date. Keep rejected and draft material separate from reported results.
5. Assign related scenes and every paraphrase to the same split. Keep each video in only one split, reserve independent videos for `test`, and avoid editing test labels based on retrieval output. Re-run the validator after every manifest or annotation change.

Generate a sheet and validate the current files from the repository root:

```powershell
.venv\Scripts\python.exe -m scripts.contact_sheet video-test.mp4 artifacts\contact-sheets\video-test.jpg --every 1 --max-frames 64
.venv\Scripts\python.exe -m evaluation.dataset --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --every 1
.venv\Scripts\python.exe -m evaluation.dataset --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --every 1 --require-ready
```

The validator exits 2 for structural/source errors. Without `--require-ready`, it exits 0 when the available files are structurally sound, even if the dataset is incomplete. With `--require-ready`, it exits 3 until at least 50 valid verified labels exist across both `dev` and `test` with no validation issues. `n_nominal_covered` counts valid windows intersecting a uniform sampling grid **before** histogram filtering or duplicate removal; it is not searchable index coverage. Category counts include all structurally supplied labels, including drafts. A passing metadata validator cannot certify visual semantics or language quality.

## T05 acceptance status

| Criterion | Status | Evidence |
|---|---|---|
| Source manifest, rights, real file checks | PASS for one video | `data/annotations/manifest.json`; validator checks file, SHA-256 and OpenCV metadata |
| At least 50 verified Vietnamese queries | BLOCKED | `data/annotations/labels.json` has 0 rows |
| Human-checked windows, categories, review metadata | BLOCKED | Contact-sheet workflow exists; no reviewed labels yet |
| Independent split without query/paraphrase leakage | BLOCKED for actual dataset | One video is assigned to `dev`; validator checks group, duplicate/near-duplicate cross-split queries |
| Validator and nominal sampling coverage reporting | PASS as tooling | `evaluation/dataset.py`, `scripts/contact_sheet.py`, `tests/test_dataset_t05.py`; current coverage 0/0 |

T05's dataset gate remains blocked until additional source videos and human-reviewed labels are available. This checkpoint is implementation work only; it does not justify retrieval accuracy claims or a T06 benchmark conclusion.
