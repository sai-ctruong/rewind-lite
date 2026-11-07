# Rewind Lite — T09 hardening và bàn giao

Ngày kiểm tra: 2026-11-07 (Asia/Bangkok). Đây là bản kiểm tra để review **checkpoint T09**, chưa phải tuyên bố benchmark hoặc phát hành học thuật hoàn tất. T05 sau đó nhận thêm `video_test2.mp4`; hiện có 2 video, 14 nhãn draft và 0 verified. Dữ liệu T05/T06 vẫn bị chặn.

## Kịch bản demo 3–5 phút

**Chuẩn bị trước khi trình bày:** cài môi trường theo [README](../README.md), chuẩn bị MP4 được phép dùng, chạy `.venv\Scripts\python.exe -m ui.lite_app`, mở `http://127.0.0.1:5000/`. Dùng sidebar tải MP4 và chờ index xong; kiểm tra player, hai thumbnail hiện có và âm thanh/video trên máy trình chiếu. Có thể dùng lại index đã tạo khi video và cấu hình không đổi. Không cần chạy lệnh trong lúc trình bày.

| Thời lượng | Thao tác / lời dẫn | Bằng chứng trên màn hình |
|---|---|---|
| 0:00–0:40 | Nêu bài toán tiếng Việt và phạm vi keyframe timestamp | Video trong player; không gọi đây là full temporal grounding |
| 0:40–1:20 | Chọn video, giải thích sampling → encoder → index | Trạng thái `ready`, số frame đã index |
| 1:20–2:40 | Nhập truy vấn tiếng Việt, tìm Top-5 | Rank, thumbnail, timestamp và cosine score thật; mẫu hiện trả 2 kết quả |
| 2:40–3:20 | Chọn một thumbnail và phát video từ timestamp | Player tua tới keyframe đã chọn |
| 3:20–4:20 | Nêu dataset và giới hạn đánh giá | 2 video `dev`/`test`, 14 nhãn draft, 0 verified; accuracy chưa đo |
| 4:20–5:00 | Tóm tắt đóng góp và hướng hoàn thành benchmark | Validator, benchmark gate và kế hoạch bổ sung dữ liệu |

Nếu trình duyệt không phát video, kiểm tra MP4/codec bằng cách phát file gốc trước buổi demo. Nếu model chưa có cache, tải và index từ trước. Không trình bày score cosine của một query như accuracy.

## Dàn ý báo cáo / slides

1. Bài toán, động cơ và ranh giới: keyframe retrieval thay vì dự đoán `[start,end]`.
2. Thành phần pretrained và tái sử dụng: SigLIP2, PyTorch, OpenCV, Faiss; module ingestion/encoder/index từ Rewind gốc.
3. Đóng góp Lite: đường single-encoder, manifest và cache, schema/rank contract, API/UI, validator và benchmark gate.
4. Kiến trúc và preprocessing: sampling, hai lớp dedup, text max-length 64, chuẩn hóa L2, cosine Top-K.
5. Thiết kế dữ liệu và protocol: nguồn/quyền, split theo video, nhãn review độc lập, coverage và các metric T02.
6. Kết quả quan sát được: smoke trên video 23.86 giây, 2 keyframe, index lưu/nạp, API/UI hoạt động. Bảng accuracy để `chưa đo`.
7. Phân tích lỗi dự kiến: sampling miss, nhầm thuộc tính/hành động, mơ hồ hình ảnh, language gap, label issue. Chỉ gán nguyên nhân sau khi có nhãn và xem lại video.
8. Giới hạn, rủi ro và bước tiếp: thêm video/nhãn verified, benchmark E1/E2/E3, review 5–10 ca lỗi, khóa kết quả.

## Kiểm tra acceptance T02–T08

| Task | Trạng thái tại T09 | Bằng chứng / giới hạn |
|---|---|---|
| T02 | PASS về protocol | [T02](rewind-lite-task-02-protocol.md), unit tests metrics/dataset; video legacy chưa xác minh |
| T03 | PASS đường model thật | CLI CUDA dùng lại index, 2 keyframe tại 5.0 và 0.0 giây; [T03](rewind-lite-task-03-environment.md) |
| T04 | PASS core và import boundary | Full suite 353 pass, 1 skip; [T04](rewind-lite-task-04-core.md) |
| T05 | DATA BLOCKED | Bản T09 ban đầu: 1 video, 0 labels. Sau cập nhật T05: 2 video, 14 draft, 0 verified, `ready_for_accuracy=false`; [T05](rewind-lite-task-05-dataset.md) |
| T06 | DATA BLOCKED | `evaluation.lite_benchmark --check` trả `DATA_BLOCKED`; không có accuracy report; [T06](rewind-lite-task-06-benchmark.md) |
| T07 | PASS API smoke | Flask client trên model thật: select 200, index cached 200, search 200, frame 200, stream range 206; [T07](rewind-lite-task-07-api.md) |
| T08 | PASS UI flow trong phạm vi đã kiểm | Trang UI 200; browser/desktop/mobile check và click-to-seek ghi ở [T08](rewind-lite-task-08-ui.md). Không lặp lại visual browser check ở T09. |

## Phiên bản, bảo mật và release checklist

`requirements-gpu.txt` pin PyTorch 2.5.1+cu121, NumPy 1.26.4, Transformers 4.57.3, sentencepiece 0.2.1 và protobuf 5.29.5. Các gói core còn lại dùng lower bound; manifest index và benchmark report lưu version thực. Môi trường hiện tại: Python 3.11.15, CUDA khả dụng trên RTX 3060 Laptop, `uv pip check` báo 40 package tương thích. Lưu toàn bộ freeze trong `artifacts/dependency-freeze.txt` cục bộ trước phép đo mới.

Kiểm tra file handling: API giới hạn upload 512 MiB, tạo ID trên server, kiểm tra MP4 decode, giới hạn ID/path và chỉ cho tải frame thuộc index; test client xác nhận các lỗi liên quan. Index pickle là artifact **tin cậy cục bộ**, không nạp index từ người lạ. Lite server chỉ bind `127.0.0.1`. `.gitignore` loại video, model weights, cache, artifact, `.env.*` và key. `git ls-files` không phát hiện video/weight/key lớn theo tên hoặc file tracked trên 100 MiB; grep các mẫu private key/token phổ biến không có match. Đây là rà soát cơ bản, không thay cho kiểm tra secret chuyên dụng.

| Gate phát hành | Kết quả |
|---|---|
| README tiếng Việt, lệnh PowerShell, kiến trúc, giới hạn, provenance | PASS |
| Setup sạch hoặc tương đương; CLI/API/UI | PASS trên venv hiện có, `uv pip check`, CUDA và model thật; **chưa thử clone/cài mới** |
| Full suite, validator, dependency check | PASS: 353 passed, 1 skipped; validator không có lỗi cấu trúc |
| Benchmark accuracy trên tập verified độc lập | BLOCKED: 0 nhãn verified; nay có một video `test` nhưng chưa review |
| 5–10 failure cases đã xem lại video | BLOCKED: chưa có predictions được chấm hợp lệ |
| Audit file nhạy cảm và artifact lớn | PASS ở mức kiểm tra cơ bản nêu trên |
| Demo script, dàn ý và limitations | PASS trong tài liệu này |

**Chưa thể phát hành kết luận accuracy.** Sau khi nhận video bổ sung và nhãn có reviewer: hoàn thành T05, chạy T06 trên tập dev/test cố định, kiểm tra per-query results và failure cases, rồi cập nhật README/report bằng số đo thực cùng config và artifact JSON/Markdown. Không sửa test split theo đầu ra model.

## T09 — Acceptance Review

| Criterion | PASS / FAIL / BLOCKED | Evidence |
|---|---|---|
| Setup và CLI/API/UI hoạt động | PASS trong môi trường tương đương | `uv pip check`; CUDA; CLI và Flask smoke model thật; README |
| Tests cần thiết, phần chưa chạy ghi rõ | PASS | 353 passed, 1 skipped; clean setup và visual browser T09 chưa chạy |
| README metrics thật, config, artifact links | PARTIAL / DATA BLOCKED | README có bảng quan sát thật và config; chưa thể liên kết report T06 vì artifact hợp lệ chưa tồn tại |
| Security/basic file handling, không tracked media/keys | PASS mức cơ bản | API tests, Git ignore/track/grep audit |
| Demo script, report outline, limitations | PASS | Tài liệu này |
| Benchmark accuracy và error analysis đầy đủ để release | BLOCKED | T05/T06 cần thêm video và nhãn verified |

Tests run: `.venv\Scripts\python.exe -m pytest -o addopts='' -q` (353 passed, 1 skipped); `uv pip check --python .venv\Scripts\python.exe` (40 compatible); `scripts.lite_search` dùng model thật/index cũ trên CUDA; Flask real-path smoke; `evaluation.dataset` (0 issues, 0 verified); `evaluation.lite_benchmark --check` (`DATA_BLOCKED`); Git ignore/tracked-file/secret-pattern audit; `git diff --check`.

Tests not run and why: không cài lại từ clone sạch trong T09; venv hiện có và package check là môi trường kiểm thử tương đương. Không chạy benchmark có nhãn, accuracy/VRAM/throughput hoặc phân tích lỗi xác minh vì T05 chưa có tập dữ liệu. Không lặp lại browser screenshot/interaction check của T08.

Files changed: `README.md`, `requirements.txt`, `requirements-full.txt`, `.gitignore`, tài liệu này, `docs/task-status.md`.

Risks/known issues: phụ thuộc core chưa lock toàn bộ; model cache/index/video không chuyển cùng repository. Một video hiện chỉ giữ hai keyframe. Check Git theo tên/mẫu không bảo đảm phát hiện mọi bí mật. Clean install chưa được chạy lại.

Scope drift: không thay đổi logic runtime hoặc nhãn; chỉ tài liệu, comment dependency và Git ignore.

Decision: người dùng đã duyệt checkpoint T09 `PARTIAL / DATA BLOCKED` để commit và push lên `rewind-lite/main`. Đây chưa phải release học thuật hoàn tất.
