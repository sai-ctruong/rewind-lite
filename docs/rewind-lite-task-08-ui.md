# Rewind Lite — T08 UI acceptance review

Date: 2026-11-06 (Asia/Bangkok). T08 accepted for commit and push to `rewind-lite/main`.

## Demo flow

Run `.venv\Scripts\python.exe -m ui.lite_app` and open `http://127.0.0.1:5000/`. With the default directories, upload an MP4 once; the UI automatically selects and indexes it. On later launches it loads the saved index when the video, model revision, and sampling settings match. For the current T03 demo artifact, launch an app with `video_dir=Path.cwd()`, `index_root=Path('artifacts/indexes')`, `max_frames=64`, and `device='cuda'` through `create_lite_app`; this selects `video-test.mp4` and reuses its index.

The sidebar lists videos and provides MP4 upload. The search form accepts Vietnamese text and requests Top-5 from the T07 API. Results show real frame thumbnails, rank, timestamp, and cosine score. Clicking a result or its amber timeline mark sets the player's `currentTime` and attempts playback. The first saved frame appears as the video poster. The index status is visible above the player, including frame count and elapsed time while indexing. Loading, empty, upload error, index error, search error, and missing-result states have their own messages. The mobile sidebar folds into a keyboard-accessible menu.

No mock scores, Agent, VQA, image search, temporal search, or chatbot routes are included in the Lite app. The legacy `ui.app` remains separate.

## Design and browser check

Used the installed `design-taste-frontend` skill. The drawn app style was **true-black-flat**, seed `201858199`. Its first shell had no sidebar, so it was redrawn to **Ruled T-frame** to meet the required video selector. The six-column detail pattern was adapted to a video player and a results column because the product has no KPI data. The product's own frame is the dominant image; amber marks on the timeline show actual returned timestamps. Cards use flat lightness steps without gradients or shadows. Inter has a system-font fallback for offline use.

The skill's `lint.mjs --app` found 0 failures and 3 warnings from zero-padding in clock formatting (`00:05`), which is intentional. Its `inspect.mjs --app` could not complete on either installed Chrome or Edge: both timed out on `Page.enable` before page navigation. Headless Chrome screenshots were inspected instead at desktop 1440 px, tablet 768 px, and a true 390 px mobile iframe. Local ignored captures are under `.taste-skill/`. The 390 px frame reported no horizontal overflow. A browser harness exercised menu open/close, index reuse, Vietnamese query, both thumbnail images, poster, and click-to-seek. It passed first with a fake encoder isolated in the ignored QA harness, then with the real offline SigLIP2 encoder and T03 saved index. The real run returned 2 results and seeking a result moved playback to 5 seconds. This is a flow check, not a retrieval-accuracy claim; T05 still has no verified labels.

## Acceptance review

| Criterion | Status | Evidence |
|---|---|---|
| Vietnamese input, Top-5 thumbnails, timestamps and video seeking | PASS | Real SigLIP2 browser harness: 2 live results, both thumbnails loaded, seek to 00:05; `ui/lite.js` |
| Clear progress, error, empty and loading states; responsive and keyboard focus | PASS | `ui/lite_index.html`, `ui/lite.css`; 1440/768/390 screenshots; mobile menu and no-overflow browser checks |
| No out-of-scope features or invented scores | PASS | Lite route set, API-only result rendering; no production mock data |
| API integration and visual inspection | PASS with tool limitation | Flask client tests and Chrome screenshot/browser harness; `inspect.mjs` could not connect to Chrome/Edge CDP |
| Demo flow without technical operations during presentation | PASS | Video auto-selection, saved-index reuse, direct search and click-to-seek; server is started before presenting |

Tests run: `.venv\Scripts\python.exe -m pytest -o addopts='' -q` (353 passed, 1 skipped); `node --check ui/lite.js` (pass); taste-skill lint (0 FAIL, 3 clock-format warnings); `git diff --check` (clean); real and fake browser harness runs (PASS).

Tests not run and why: the taste-skill visual inspector could not attach to either browser (`CDP timeout: Page.enable`); headless screenshots and a same-origin browser harness supplied the visual and interaction checks. No scored benchmark was run because T05 has zero verified labels.

Files changed: `.gitignore`, `ui/lite_app.py`, `ui/lite_index.html`, `ui/lite.css`, `ui/lite.js`, `tests/test_lite_ui_t08.py`, `tests/test_lite_api_t07.py`, `docs/rewind-lite-task-07-api.md`, this document, `docs/task-status.md`.

Risks/known issues: the UI assumes one local Flask process, as the T07 model/index cache is per process. The Google-hosted Inter font may fall back to the system face offline. The current sample video yields only two retained keyframes. Browser screenshots and QA harness files are local ignored artifacts.

Scope drift: T08 added a `poster_url` field to the T07 index response and fixed a concurrent-thumbnail 409 found during browser testing; no new advanced routes were added.

Decision: the user approved T08 commit and push to `rewind-lite/main`.
