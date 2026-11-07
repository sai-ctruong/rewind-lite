# Rewind Lite — T05 dataset preparation and annotation protocol

Date: 2026-11-06 (Asia/Bangkok). Updated 2026-11-08 after the repository owner requested Codex review of all 57 labels.

## Current evidence

`data/annotations/manifest.json` records five local source videos supplied by the repository owner. The owner confirmed permission to use the downloaded `video_test2.mp4` and `video_test3.mp4` freely and stated that they filmed `video-test4.mp4` and `video-cook.mp4` and allow both to be used for evaluation. All five files remain local and ignored by Git. Original URLs and specific license terms for the downloaded files have not been supplied.

| Video | Split | SHA-256 | OpenCV metadata | Scene observed |
|---|---|---|---|---|
| `video-test.mp4` | `dev` | `8410556fea9c755db18ebd29316405065e4330af52aa21530604bb72abdc448c` | 1,193 frames, 50 FPS, 23.86 s | Crowded tree-lined walkway |
| `video_test2.mp4` | `test` | `a0de57a8a60d9c87d4aebf3f9407b44fdfedb9a2d3bdd2dc82cdc1a1653d83b8` | 663 frames, 29.970 FPS, 22.122 s | Person picking up and carrying bottles by rocks |
| `video_test3.mp4` | `dev` | `ab9bf9811ef8fb79cd5048c65bd933c2933ad2467bbf26a57f4cb99207ba8341` | 468 frames, 25 FPS, 18.72 s, 3840×2160 | Person at a computer drinking from a clear bottle |
| `video-test4.mp4` | `test` | `47c8745b5d94f59b0279950f873a4c8479d4c27bb149e2e827bc82c0a2e039ba` | 341 frames, 24 FPS, 14.208 s, 1280×720 | Helmeted person raises a phone to the ear at a worksite |
| `video-cook.mp4` | `dev` | `e3b9e4d3e2d7004a246b132199086189dcfac2967a5fc09292abefae7dd64c7d` | 11,432 frames, 29.970 FPS, 381.448 s, 1280×720 | Preparing creamy soup, pan-toasting bread cubes, and plating |

The scenes are visually distinct and assigned different `group_id` values. The fifth video is assigned wholly to `dev`, while the two short `test` clips remain held out. This is still too small to support a generalization claim.

`data/annotations/labels.json` now contains **57 reviewed Vietnamese queries**: 43 `dev`, 14 `test`. All 57 rows record `review_status: verified`, reviewer `Codex (owner-requested visual review)`, and review date `2026-11-08`. The validator reports 5 videos, 57 labels, 57 nominally sampled windows, **57 verified**, 0 structural issues, and `ready_for_accuracy=true`. This status records Codex's visual review at sampled frames; the original annotator and reviewer are the same agent, so it is **not independent human adjudication**. The 78 rows in `evaluation/labels.json` and 51 in `evaluation/labels_en.json` remain unverified legacy material because their source videos are absent; they are excluded from T05 counts.

The ignored contact sheets are `artifacts/contact-sheets/video-test.jpg` (24 samples at 1 second) and `artifacts/contact-sheets/video_test2.jpg` (45 samples at 0.5 seconds). For visual review, Codex generated `video-test-review-05.jpg` (48 samples at 0.5 seconds), `video_test2-review-025.jpg` (89 samples at 0.25 seconds), `video_test3-review-05.jpg` (38 samples at 0.5 seconds), `video_test3-review-025.jpg` (75 samples at 0.25 seconds), `video-test4-review-025.jpg` (57 samples at 0.25 seconds), and `video-cook-review-2s.jpg` (191 samples at 2 seconds), plus selected full-resolution frames. These are review aids, not proof of exact event boundaries. No retrieval accuracy is reported.

## Codex visual review of the draft labels

The following tables record the draft-stage visual pass before the owner-requested final review on 2026-11-08. Codex inspected timestamped frames **without viewing retrieval predictions**. At that stage the rows remained `draft`; the final review and changes are recorded below.

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

For `video-test4.mp4`, Codex inspected the whole 0.25-second contact sheet and original-resolution frames at 0, 2.75, 3.25, 8, and 14 seconds. The seven new drafts cover visible clothing, structure, and the phone movement; they do not infer what the person says or hears:

| Query IDs | Visual evidence | Proposed window / remaining uncertainty |
|---|---|---|
| `vi-test-008`, `009`, `010`, `011`, `014` | Yellow hard hat, reflective vest, blue shirt, construction structure, and a partly turned back persist | Broad `[0, 14.208333)` scene windows need playback confirmation |
| `vi-test-012` | Black phone moves up to the person's ear around 3 s | Proposed `[2.75, 3.5)`; exact transition needs playback review |
| `vi-test-013` | Phone stays beside the ear through the inspected end frame | Proposed `[3.25, 14.208333)`; reviewer should inspect the ending boundary |

For `video-cook.mp4`, Codex inspected the entire 2-second contact sheet and full-resolution frames at 14, 50, 98, 112, 202, 228, 252, and 328 seconds without viewing retrieval predictions. The 27 new drafts cover distinct steps instead of repeating the same plated food description. The 2-second sample may miss exact event transitions, so these ranges need playback review:

| Query IDs | Visual evidence | Proposed windows / remaining uncertainty |
|---|---|---|
| `vi-dev-017`–`019` | Ingredient counter, clam cans held up, and bacon package | 4–30 s; individual display moments 12–16 s and 18–20 s |
| `vi-dev-020`–`026` | Bacon added and removed; carrots, celery, onion, then white powder added to pot | 32–88 s; exact starts and stops need playback |
| `vi-dev-027`–`032` | Liquids poured from separate containers; potatoes and clams added; leaf visible in creamy soup | 100–210 s; ingredient names are limited to what the frames support |
| `vi-dev-033`–`037` | Butter melts, bread cubes are toasted, oil and chopped herbs added | 218–282 s; avoid inferring oil type or seasoning from the thumbnails |
| `vi-dev-038`–`043` | Soup ladled, bread and bacon topped, herbs sprinkled, spoon scoops the final bowl | 292–370 s, plus plated-soup opening at 0–4 s; credits at 378–380 s are excluded |

The owner described the two downloaded videos as free to use and said they filmed `video-test4` and `video-cook` and permit their use for evaluation. No original URL, publisher, or license text was supplied for the downloaded files, so the manifest records those statements and the provenance gap explicitly.

## Owner-requested final review — 2026-11-08

Codex compared every query with the source video's timestamped contact sheet and selected original-resolution frames, without viewing retrieval predictions. The four short clips were reviewed at 0.5- or 0.25-second sampling. For the 381-second cooking clip, Codex reviewed the whole 2-second sheet, then 8-frame event rows for each of its 27 labels and full-resolution frames around ambiguous ingredients and actions. The review tightened **25 window sets** (24 `dev`, 1 `test`) and classified `vi-dev-018`, whose query explicitly counts two clam cans, as `count`. No query text was changed. All 57 labels were marked `verified` with Codex and the date recorded in each row.

| Labels | Review finding and action |
|---|---|
| `vi-dev-001`–`007` | Street scene, cyclist, blue jacket, pigeon, lamps, sign, and group remain visible in the proposed windows; kept all seven windows |
| `vi-dev-008`–`016` | Computer scene and bottle actions match the 0.25-second sequence; narrowed necklace visibility in `010` and bottle-raising event in `013` |
| `vi-test-001`–`007` | Bottle pickup, crushed plastic bottle, and bag are visible; narrowed `006` to the interval where the green bottle is handled beside/into the bag |
| `vi-test-008`–`014` | Hard hat, vest, blue shirt, background structure, and phone movement match the 0.25-second sequence; windows retained |
| `vi-dev-017`–`030` | Ingredient display and successive additions to the pot match the event rows; short display/pour windows were tightened |
| `vi-dev-031`–`043` | Leaf, clams, butter, bread cubes, oil, herbs, and plating match the event rows; `038` and `043` now use separate windows for repeated actions |

This review used sampled frames rather than continuous playback, so subsecond transitions and appearances between samples may be missed. Because Codex also drafted the labels, an independent human pass is still recommended before using the dataset as externally validated ground truth in a thesis or publication. The validator's `ready_for_accuracy=true` means the recorded metadata and count pass its programmed gate; it does not certify independent semantic accuracy.

The reviewed category distribution is 21 `action`, 11 `object`, 9 `color`, 8 `short_event`, 6 `context`, and 2 `count`. Count examples are sparse because the available clips rarely show an unambiguous count; this limits category-level conclusions.

## Format and review workflow

Each manifest video has a stable `video_id`, a path relative to the repository root, SHA-256, measured `duration_s`, `fps`, `frame_count`, a source description, confirmed rights with a basis, a `dev` or `test` split, and a `group_id` for related scenes. Store source videos locally; only JSON annotations and metadata are tracked. Recalculate the checksum and metadata when a file changes.

Each label has `query_id`, `video_id`, a Vietnamese `query`, `language: "vi"`, one or more half-open `time_windows` (`[start, end)` in seconds), `category`, `split`, `paraphrase_group`, `annotator`, `review_status`, `reviewer`, and `reviewed_at`. Allowed categories are `object`, `color`, `action`, `count`, `context`, and `short_event`; review statuses are `draft`, `verified`, and `rejected`. Only `verified` rows with valid source and review metadata contribute to the programmed completion count. The language declaration is checked as metadata; the reviewer must check wording and visual truth, and independent human review remains advisable for external claims.

1. Receive each video with provenance and usage permission; record them in the manifest and verify the file, checksum, frame rate, frame count, and duration.
2. Generate a contact sheet, then inspect each candidate segment and its boundaries in the original video. When the frame cap is reached, samples are spread across the whole video. At one-second grid spacing, a brief event can still be missed; use a shorter interval and playback around it. Never infer a narrow window from a single thumbnail.
3. Draft natural Vietnamese queries from observed content. Include objects, colors, actions, counts, context, and short events where the sources genuinely support them. Record all distinct appearances as separate windows. Do not manufacture category balance from one scene.
4. Have a reviewer inspect the source video or dense contact sheets and each window without retrieval predictions, resolve uncertain or overlapping descriptions, then set `review_status` to `verified` and record reviewer/date. Record whether the reviewer is independent from the annotator. Keep rejected and draft material separate from reported results.
5. Assign related scenes and every paraphrase to the same split. Keep each video in only one split, reserve independent videos for `test`, and avoid editing test labels based on retrieval output. Re-run the validator after every manifest or annotation change.

Generate a sheet and validate the current files from the repository root:

```powershell
.venv\Scripts\python.exe -m scripts.contact_sheet video-test.mp4 artifacts\contact-sheets\video-test.jpg --every 1 --max-frames 64
.venv\Scripts\python.exe -m scripts.contact_sheet video_test2.mp4 artifacts\contact-sheets\video_test2.jpg --every 0.5 --max-frames 64 --columns 6
.venv\Scripts\python.exe -m scripts.contact_sheet video_test3.mp4 artifacts\contact-sheets\video_test3-review-025.jpg --every 0.25 --max-frames 80 --columns 8
.venv\Scripts\python.exe -m scripts.contact_sheet video-test4.mp4 artifacts\contact-sheets\video-test4-review-025.jpg --every 0.25 --max-frames 80 --columns 8
.venv\Scripts\python.exe -m scripts.contact_sheet video-cook.mp4 artifacts\contact-sheets\video-cook-review-2s.jpg --every 2 --max-frames 200 --columns 8
.venv\Scripts\python.exe -m evaluation.dataset --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --every 1
.venv\Scripts\python.exe -m evaluation.dataset --manifest data\annotations\manifest.json --labels data\annotations\labels.json --root . --every 1 --require-ready
```

The validator exits 2 for structural/source errors. Without `--require-ready`, it exits 0 when the available files are structurally sound, even if the dataset is incomplete. With `--require-ready`, it exits 3 until at least 50 valid verified labels exist across both `dev` and `test` with no validation issues. `n_nominal_covered` counts valid windows intersecting a uniform sampling grid **before** histogram filtering or duplicate removal; it is not searchable index coverage. Category counts include all structurally supplied labels, including drafts. A passing metadata validator cannot certify visual semantics or language quality.

## T05 acceptance status

| Criterion | Status | Evidence |
|---|---|---|
| Source manifest, rights, real file checks | PASS for five owner-supplied files; provenance detail for downloaded files pending | `data/annotations/manifest.json`; validator checks file, SHA-256 and OpenCV metadata |
| At least 50 verified Vietnamese queries | PASS for programmed gate, with same-agent caveat | 57 rows marked `verified` after owner-requested Codex visual review; independent human adjudication absent |
| Human-checked windows, categories, review metadata | BLOCKED for human check; Codex visual pass complete | 25 window sets tightened, one category corrected, reviewer/date on all rows; sampled frames cannot establish exact subsecond boundaries |
| Independent split without query/paraphrase leakage | PASS structurally, small sample | Each video and group belongs to one split; two short held-out clips limit inference |
| Validator and nominal sampling coverage reporting | PASS | `evaluation/dataset.py`, `scripts/contact_sheet.py`, `tests/test_dataset_t05.py`; 57/57 reviewed windows intersect the nominal 1 s grid before dedup; 0 issues, `ready_for_accuracy=true` |

T05's programmed dataset gate now passes for an internal T06 run. Treat its accuracy results as provisional because the same agent drafted and reviewed the labels and the review used sampled frames. Before an externally validated thesis or release claim, have an independent human replay all five clips, adjudicate boundaries and semantics, and update reviewer metadata if corrections are made. Do not use retrieval output to revise the held-out test labels.
