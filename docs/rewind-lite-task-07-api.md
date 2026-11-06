# Rewind Lite — T07 API and integration review

Date: 2026-11-06 (Asia/Bangkok). T07 acceptance reviewed; user approved commit and push to `rewind-lite/main`.

## Run and HTTP contract

Start the local Flask server with `.venv\Scripts\python.exe -m ui.lite_app`. It listens on `127.0.0.1:5000`, reads videos from `data/videos`, and stores indexes in `artifacts/indexes`. The app factory allows overriding these locations, device, sampling, and the 512 MiB upload limit. Indexing is synchronous; another request can poll progress while the threaded server handles the index request. The service uses one engine and lazily loads one model per process.

The Lite app registers only the following routes. JSON errors have the form `{"error":{"code":"index_missing","message":"Index this video before searching"}}` with an appropriate HTTP status.

| Method | Route | Request | Success |
|---|---|---|---|
| GET | `/api/lite/health` | none | `{"ok":true,"selected_video":null,"indexed_videos":0}` |
| GET | `/api/lite/videos` | none | `{"videos":["walk"],"selected_video":null}` |
| POST | `/api/lite/upload` | multipart field `video`, decodable `.mp4` | 201 `{"video_id":"<generated-id>","selected":true}` |
| POST | `/api/lite/select` | `{"video_id":"walk"}` | `{"video_id":"walk","selected":true}` |
| POST | `/api/lite/index` | `{"video_id":"walk","rebuild":false}`; `video_id` may be omitted after select | `{"video_id":"walk","cached":false,"frames":2,"sampled":2,"poster_url":"/api/lite/frames/walk/walk/0"}` |
| GET | `/api/lite/progress` | none | `{"status":"ready","video_id":"walk","count":2,"elapsed_s":1.2,"error_code":null}` |
| POST | `/api/lite/search` | `{"video_id":"walk","query":"người đi bộ","top_k":5}` | ranked results, example below |
| GET | `/api/lite/frames/<video_id>/<frame_id>` | frame ID returned by search | JPEG bytes |
| GET | `/api/lite/videos/<video_id>/stream` | optional `Range: bytes=0-99` | MP4 bytes; 206 for a valid range |

Example search response (scores and timestamps are illustrative):

```json
{
  "video_id": "walk",
  "query": "người đi bộ",
  "results": [{
    "rank": 1,
    "score": 0.42,
    "score_kind": "cosine",
    "video_id": "walk",
    "frame_id": "walk/0",
    "timestamp": 5.0,
    "image_url": "/api/lite/frames/walk/walk/0"
  }]
}
```

`/api/lite/progress` uses `idle`, `indexing`, `ready`, or `error`. During indexing it exposes the sampled-frame count and elapsed seconds; failure exposes a stable `error_code`. Only one index/search/frame decode runs at a time within a process. Concurrent index, search, or frame requests receive 409 `index_busy`.

Uploads require a `.mp4` name and a decodable video, with a generated server-side ID. Unsupported or undecodable files return 415 `unsupported_video`; over-limit requests return 413 `upload_too_large`; write failure returns 500 `upload_failed`. Empty or oversized queries return 400 `invalid_query`; `top_k` outside 1–50 returns 400 `invalid_top_k`. Invalid, unknown, or unselected IDs return `invalid_video_id` (400), `video_not_found` (404), or `video_not_selected` (409). Missing, incomplete, or expired indexes return 409 with `index_missing`, `index_incomplete`, or `index_expired`. Frame IDs absent from the index return 404 `frame_not_found`; decode failures return 404 `frame_unavailable`. Model memory exhaustion returns 503 `model_oom`; other indexing/search failures return 500 `index_failed` or `search_failed`. These responses do not include source or index filesystem paths.

Saved indexes use the T03 manifest. On restart, matching video checksum, pinned model revision, and sampling settings permit reuse without reembedding. A changed source video is rejected. Index files are trusted local artifacts because the existing entry format uses Python pickle; do not import index directories from untrusted sources.

T08 adds `poster_url` to the index response so the UI can show a real saved frame before playback. The existing frame route serves it. Frame decoding now releases the index lock after validating the frame, allowing multiple result thumbnails to load concurrently while index/search operations still return `index_busy`.

## Acceptance review

| Criterion | Status | Evidence |
|---|---|---|
| HTTP contract with request/response examples | PASS | Route table and JSON example above |
| Clear input, missing-index, OOM, index/write/decode failure responses | PASS | `tests/test_lite_api_t07.py` |
| Reuse saved index after restart; truthful progress | PASS | Restart and concurrent-progress tests; real-video smoke below |
| API tests and path/ID guards; no advanced Lite routes | PASS | `tests/test_lite_api_t07.py` |
| End-to-end API on a real video | PASS | Real SigLIP2/CUDA smoke below |

Real smoke used the authorized local `video-test.mp4`, the T03 saved index, `HF_HUB_OFFLINE=1`, and `create_lite_app(video_dir=Path.cwd(), index_root=Path('artifacts/indexes'), max_frames=64, device='cuda')`. Select, index reuse, Vietnamese text search, frame JPEG, and 100-byte video range returned HTTP 200, 200 (`cached=true`), 200 (two results), 200, and 206 respectively. This verifies the route-to-engine path; it is not a retrieval-accuracy result. The current source has no verified labels, so T05/T06 accuracy gates remain blocked.

Tests run: `.venv\Scripts\python.exe -m pytest -o addopts='' tests/test_lite_api_t07.py -q` (8 passed); `.venv\Scripts\python.exe -m pytest -o addopts='' -q` (351 passed, 1 skipped); `git diff --check` (clean).

Tests not run and why: no scored accuracy benchmark because T05 has zero verified labels and no independent test video.

Files changed: `ui/lite_app.py`, `tests/test_lite_api_t07.py`, this document, `docs/task-status.md`.

Risks/known issues: the synchronous local server is a single-process implementation; progress and model cache are per process. A source copied into the default `data/videos` directory must be indexed with the same sampling settings used by that app. Existing `ui.app` remains a separate legacy application.

Scope drift: none.

Decision: user approved T07 commit and push to `rewind-lite/main`.
