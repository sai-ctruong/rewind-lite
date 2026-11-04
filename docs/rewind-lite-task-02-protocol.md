# Rewind Lite — T02: annotation audit and evaluation protocol

Audit date: 2026-11-05. This is a **static audit** of the two JSON files, not a review of source-video content. No video files or measured model results were used.

## Static audit

| File | Rows | Vietnamese syntax | English syntax | Mixed syntax | Video IDs |
|---|---:|---:|---:|---:|---|
| `evaluation/labels.json` | 78 | 66 | 12 | 0 | `beach_walking` 11, `walking` 7, `walk_rain_seoul` 60 |
| `evaluation/labels_en.json` | 51 | 0 | 51 | 0 | `beach_walking` 11, `walking` 7, `walk_rain_seoul` 33 |

Classification uses the language of the descriptive syntax; proper names, store signs, and loanwords do not by themselves make a Vietnamese sentence mixed. English rows in `labels.json` (1-based positions): **3, 12, 18, 21, 23, 29, 73–78**. Examples: row 3, “Sydney Opera House by the water”; row 12, “yellow taxi cab in Times Square”; row 73, “red curved sculpture on the sidewalk”. Rows 5 and 26 include English proper-name/sign text in Vietnamese syntax and deserve manual language review if a stricter code-switching definition is wanted. No annotation was edited or reclassified in the JSON.

Both files have nonempty `video_id` values, finite nonempty positive-duration windows, and no exact duplicate `(query, video_id)` pair. Their three video ID strings agree, but **source file existence, actual duration, and whether each scene appears in its window are unverified**. The current files have no row `id`; uniqueness of future IDs is enforced by the validator when IDs are provided, and can be made mandatory with `require_ids=True` after migration.

Identical-window groups in `labels.json` are rows **2/3, 11/12, 17/18, 20/21, 22/23, 28/29, 58/77, 61/78, 62/75, 65/73, 67/74, 69/76**. The first six occur in `labels_en.json` too. Exact `(query, video_id, window)` duplicates **across** files are rows 3, 12, 18, 21, 23, 29 in both files. These are potential evaluation leakage if the files are treated as separate splits; they are not automatically bad annotations. Within each file, 13 pairs have partially overlapping windows: 1/2, 1/3, 4/5, 5/6, 6/7, 8/35, 9/10, 9/36, 9/37, 10/36, 11/13, 12/13, 36/37. Review these pairs against video before deciding whether overlaps are correct. Rows 73–78 are English paraphrases of earlier rows and should be reviewed together with their counterparts. All rows ultimately need source-video confirmation before reporting accuracy.

## Annotation contract

`evaluation/labels.schema.json` describes legacy `time_window: [start, end]` and new `time_windows: [[start, end], ...]`. `evaluation.dataset.validate_labels` enforces the conditions JSON Schema cannot express: `0 <= start < end`, finite numbers, no overlapping windows **within a single annotation**, unique supplied IDs, and `video_id` membership when given an independently checked manifest ID set. Windows are half-open: a keyframe at `start` counts; one at `end` does not. Adjacent windows may touch. Multiple nonoverlapping windows are a union for one query; a keyframe counted in two windows still counts once. Legacy rows remain loadable without an `id`; new annotations should use stable IDs such as `vi-001`, which must never be reused.

Manual annotation procedure: verify each video checksum and duration; use timestamped contact sheets to find candidate moments; inspect frames and nearby playback at original resolution; enter the narrowest supported window(s); check boundary frames; record annotator, date, and review decision in a separate annotation log; have a second reviewer adjudicate uncertain content and overlaps. Do not change window values based on search output. If a scene appears multiple times, add separate windows after video review. Keep an immutable copy of the originally reviewed labels.

## Scoring policy

For each valid label, compute sampled positives from `entry.raws`, then searchable positives as the intersection with `entry.index.ids` after dedup. `relevant_ids()` now returns only searchable positives. A hit is a returned ID in that set. Hit@K, Recall@K, and MRR use those IDs; recall's denominator is the count of searchable positives. Report `annotation_validity`, `index_coverage`, `retrieval_hit`, and `outcome` per query:

1. Invalid annotation: record issues; do not search; exclude from metric denominator, but count in `n_invalid`.
2. No sampled positive in any window: `no_sampled_positive`; include as zero in primary metrics.
3. Sampled positives exist but all are removed from the searchable index: `dedup_removed_positive`; include as zero.
4. Searchable positive exists but no returned hit: `retrieval_rank_miss`; include as zero.
5. Searchable hit: `retrieval_hit`.

The **primary denominator is all structurally valid annotations**, including uncovered queries. Report `n_queries`, `n_valid`, `n_invalid`, `n_covered`, and `coverage = n_covered / n_valid`. Secondary `hit@K_covered` and `mrr_covered` divide by covered queries only and are `null` if none are covered. Do not compare covered-only scores without coverage; they can improve when dedup silently drops hard queries. Model accuracy must also include video-source validation, which is unavailable here.

## Reproducible dataset manifest and split plan

Before a real run, write a versioned manifest containing, for **each actual video**: stable `video_id`, relative path, SHA-256, measured duration, frame rate, and split/group. Record label-file SHA-256, annotation revision/reviewer, and split assignment. Record the experiment config: frame sampling interval and cap, histogram/dedup method and thresholds, encoder model ID **and revision**, preprocessing, index type and parameters, retrieval/fusion config, seed, Python/dependency versions, device/GPU, command, timing, and peak VRAM. Fail the run if files or checksums disagree. No manifest with invented checksums or durations is included.

Only three video IDs are present, with 60/78 labels concentrated in one video. Use **grouped video-level dev/test folds**, rotating which video is held out, and report each fold plus a query-weighted aggregate. Keep all paraphrases, languages, and overlapping scene annotations in the same video fold. Predeclare model/config selection using dev videos only; do not tune on a held-out video. If additional independent videos become available, reserve a final untouched test set at video level. No train split is needed for frozen pretrained encoders; if training is introduced later, obtain additional videos and split by video before training.

## Remaining limit

The repository lacks the source videos and a verified manifest. Static validation passes for both files, but semantic label correctness, duration bounds, actual index coverage, and retrieval accuracy remain unverified. GPU is not needed for this audit.
