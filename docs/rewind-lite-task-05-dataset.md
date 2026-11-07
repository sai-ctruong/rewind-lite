# Rewind Lite — T05 dataset preparation and annotation protocol

Date: 2026-11-06 (Asia/Bangkok). Updated 2026-11-07 with two additional local videos and draft annotations; this is still a partial T05 checkpoint.

## Current evidence

`data/annotations/manifest.json` records three local source videos supplied by the repository owner. The owner confirmed permission to use the downloaded `video_test2.mp4` and `video_test3.mp4` freely. All three files remain local and ignored by Git. Original URLs and specific license terms for the downloaded files have not been supplied.

| Video | Split | SHA-256 | OpenCV metadata | Scene observed |
|---|---|---|---|---|
| `video-test.mp4` | `dev` | `8410556fea9c755db18ebd29316405065e4330af52aa21530604bb72abdc448c` | 1,193 frames, 50 FPS, 23.86 s | Crowded tree-lined walkway |
| `video_test2.mp4` | `test` | `a0de57a8a60d9c87d4aebf3f9407b44fdfedb9a2d3bdd2dc82cdc1a1653d83b8` | 663 frames, 29.970 FPS, 22.122 s | Person picking up and carrying bottles by rocks |
| `video_test3.mp4` | `dev` | `ab9bf9811ef8fb79cd5048c65bd933c2933ad2467bbf26a57f4cb99207ba8341` | 468 frames, 25 FPS, 18.72 s, 3840×2160 | Person at a computer drinking from a clear bottle |

The scenes are visually distinct and assigned different `group_id` values. The third video joins `dev` to preserve the held-out `test` video. Two short `dev` clips and one short `test` clip are still too small to support a generalization claim.

`data/annotations/labels.json` now contains **23 candidate Vietnamese queries**: 16 `dev`, 7 `test`. Every row is explicitly `draft`, with no reviewer or review date. The draft windows were proposed from timestamped sampled frames and selected full-resolution frames; they have **not** received an independent review against playback. The validator reports 3 videos, 23 labels, 23 nominally sampled windows, **0 verified**, and 0 structural issues. The 78 rows in `evaluation/labels.json` and 51 in `evaluation/labels_en.json` remain unverified legacy material because their source videos are absent; they are excluded from T05 counts.

The ignored contact sheets are `artifacts/contact-sheets/video-test.jpg` (24 samples at 1 second) and `artifacts/contact-sheets/video_test2.jpg` (45 samples at 0.5 seconds). For visual review, Codex generated `video-test-review-05.jpg` (48 samples at 0.5 seconds), `video_test2-review-025.jpg` (89 samples at 0.25 seconds), `video_test3-review-05.jpg` (38 samples at 0.5 seconds), and `video_test3-review-025.jpg` (75 samples at 0.25 seconds), plus selected full-resolution frames. These are review aids, not proof of exact event boundaries. No retrieval accuracy is reported.

## Codex visual review of the draft labels

The owner designated Codex to review the candidates. Codex performed a second pass over timestamped frames **without viewing retrieval predictions**. This checked visible content and improved proposed windows, but it is not an independent human playback review. The rows therefore remain `draft`, with `reviewer` and `reviewed_at` unset; `n_verified=0` is intentional.

| Query IDs | Second-pass observation | Edit / remaining uncertainty |
|---|---|---|
| `vi-dev-001`, `002`, `006` | Crowd, globe lamps, bicycle sign persist in the static street view | Kept broad windows; exact presence at every boundary still needs playback review |
| `vi-dev-003` | Dark-clothed cyclist with white chest stripes approaches on the left | Removed inferred gender, distinguished him from later cyclists, extended window to 14 s |
| `vi-dev-004` | Blue-jacketed person approaches throughout the clip | Removed inferred gender; expanded to the beginning |
| `vi-dev-005` | Pigeon is visible on the walkway through the inspected earlier frames | Shortened window to 21 s rather than assert presence at the very end |
| `vi-dev-007` | Group walks away down the center | Removed inferred age |
| `vi-test-001` | Green glass bottle rests on rocks before pickup | Kept `[0, 6.5)` |
| `vi-test-002` | Beige-coated person bends toward the rocks | Expanded start to 2.5 s; end 7.5 s |
| `vi-test-003` | Hand lifts green bottle from the rocks around 6–7 s | Narrowed short event to `[6, 7.25)` |
| `vi-test-004`, `005` | Crumpled clear plastic bottle and brown paper bag enter view before 10 s | Moved starts to 8.75 s and 8.5 s |
| `vi-test-006` | Green bottle is moved into the paper bag | Narrowed proposed action to `[9.75, 16)`; exact completion still needs playback review |
| `vi-test-007` | Large plastic bottle with blue label hangs beside the bag | Moved start to 9.25 s |

For `video_test3.mp4`, Codex inspected the whole 0.25-second contact sheet and selected original-resolution frames without retrieval predictions. The nine new drafts cover distinct visible details and events; proposed windows still need playback and independent review:

| Query IDs | Visual evidence | Proposed window / remaining uncertainty |
|---|---|---|
| `vi-dev-008`, `009`, `011`, `016` | One white-shirted person at a computer; pink foreground logo and crossed wooden beams remain visible | Broad `[0, 18.72)` scene windows; a reviewer must check every boundary |
| `vi-dev-010` | Silver-colored chain visible against white shirt | `[2.5, 16)`; material is not inferred from color |
| `vi-dev-012` | Clear bottle held in front of the person | `[3.25, 14)`; material appears glass in inspected full-resolution frames |
| `vi-dev-013`, `014`, `015` | Bottle rises to mouth, person drinks, then lowers it | Proposed `[5.75, 6.25)`, `[6, 9)`, `[8.75, 9.75)`; exact transitions require playback |

The owner described both downloaded videos as free to use. No original URL, publisher, or license text was supplied, so the manifest records those statements and the provenance gap explicitly. A human reviewer still needs to play all three full clips, correct any missed appearances or boundaries, and record name/date before any row becomes `verified`. More diverse videos and genuine query moments are needed to reach the 50-query gate without padding three short scenes with near-duplicates.

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
.venv\Scripts\python.exe -m scripts.contact_sheet video_test2.mp4 artifacts\contact-sheets\video_test2.jpg --every 0.5 --max-frames 64 --columns 6
.venv\Scripts\python.exe -m scripts.contact_sheet video_test3.mp4 artifacts\contact-sheets\video_test3-review-025.jpg --every 0.25 --max-frames 80 --columns 8
.venv\Scripts\python.exe -m evaluation.dataset --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --every 1
.venv\Scripts\python.exe -m evaluation.dataset --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --every 1 --require-ready
```

The validator exits 2 for structural/source errors. Without `--require-ready`, it exits 0 when the available files are structurally sound, even if the dataset is incomplete. With `--require-ready`, it exits 3 until at least 50 valid verified labels exist across both `dev` and `test` with no validation issues. `n_nominal_covered` counts valid windows intersecting a uniform sampling grid **before** histogram filtering or duplicate removal; it is not searchable index coverage. Category counts include all structurally supplied labels, including drafts. A passing metadata validator cannot certify visual semantics or language quality.

## T05 acceptance status

| Criterion | Status | Evidence |
|---|---|---|
| Source manifest, rights, real file checks | PASS for three owner-supplied files; provenance detail pending | `data/annotations/manifest.json`; validator checks file, SHA-256 and OpenCV metadata |
| At least 50 verified Vietnamese queries | BLOCKED | 23 drafts, 0 verified; three short clips do not justify manufacturing near-duplicate queries |
| Human-checked windows, categories, review metadata | BLOCKED | Contact sheets and candidate windows exist; no independent reviewer yet |
| Independent split without query/paraphrase leakage | PASS structurally, small sample | Distinct scenes and group IDs; validator found 0 issues; one held-out `test` clip limits inference |
| Validator and nominal sampling coverage reporting | PASS as tooling | `evaluation/dataset.py`, `scripts/contact_sheet.py`, `tests/test_dataset_t05.py`; 23/23 draft windows intersect the nominal 1 s grid before dedup |

T05's dataset gate remains blocked until independent human review and at least 50 verified queries are available, ideally with more source-video diversity. The reviewer should play each source clip near the listed windows, correct all appearances and boundaries, then add their name/date before changing `review_status` to `verified`. Do not use retrieval output to revise the held-out test labels. This checkpoint does not justify retrieval accuracy claims or a T06 benchmark conclusion.
