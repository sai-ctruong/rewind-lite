# Rewind Lite — T09 bàn giao và checklist phát hành

Cập nhật 2026-11-08 (Asia/Bangkok) sau benchmark T06. Bản Lite đã sẵn sàng cho **demo nội bộ** và có hướng dẫn chạy lại trên năm video cục bộ. Kết quả accuracy chưa đủ cơ sở để công bố học thuật: Codex vừa tạo vừa review 57 nhãn; hai video tải về chưa có URL gốc hoặc điều khoản cấp phép cụ thể. [README](../README.md) là hướng dẫn cài/chạy; [T06](rewind-lite-task-06-benchmark.md) là giao thức, số đo và phân tích lỗi; [JSON T06](benchmark-results/t06.json) lưu từng dự đoán, hash đầu vào và cấu hình.

## Kịch bản demo 3–5 phút

Trước buổi trình bày, chuẩn bị `video-test.mp4` được phép dùng, model cache và index cục bộ theo README. Chạy `.venv\Scripts\python.exe -m ui.lite_app`, mở `http://127.0.0.1:5000/`, chọn video và kiểm tra phát video/âm thanh. Index sẵn giúp tránh đợi tải model khi trình bày.

| Thời gian | Thao tác / lời dẫn | Bằng chứng |
|---|---|---|
| 0:00–0:40 | Nêu bài toán tìm keyframe từ câu tiếng Việt | Giao diện hiển thị video; kết quả là timestamp, không phải ranh giới đoạn `[start,end]` |
| 0:40–1:20 | Chọn video và giải thích sampling → SigLIP2 → dedup → index | Trạng thái `ready`, số khung hình đã index |
| 1:20–2:40 | Nhập truy vấn tiếng Việt và tìm Top-K | Thumbnail, rank, timestamp và điểm cosine; video demo hiện có 2 keyframe sau lọc trùng |
| 2:40–3:20 | Chọn thumbnail để tua player | Player phát từ timestamp đã chọn |
| 3:20–4:20 | Trình bày benchmark nội bộ T06 | 5 video, 57 truy vấn; Hit@5 test E1/E2/E3 = 13/14, 13/14, 14/14; test nhỏ và nhãn chưa review độc lập |
| 4:20–5:00 | Nêu đóng góp và giới hạn | Đường Lite/API/UI, giao thức đánh giá, 7 ca lỗi đã đối chiếu khung hình; không suy ra mô hình tốt hơn nói chung |

Nếu trình duyệt không phát MP4, kiểm tra codec bằng cách mở file nguồn trước buổi demo. Không dùng điểm cosine của một truy vấn như chỉ số accuracy.

## Dàn ý báo cáo / slides

1. Bài toán và phạm vi: retrieval keyframe, không phải full temporal sentence grounding.
2. Thành phần pretrained/tái sử dụng: SigLIP2/SigLIP, PyTorch, OpenCV, Faiss và ingestion/encoder/index từ Rewind gốc.
3. Đóng góp riêng của Lite: đường single-encoder, manifest/cache, hợp đồng rank, API/UI, validator và benchmark E1/E2/E3.
4. Kiến trúc: sampling 1 giây, lọc trùng histogram 0.985, semantic dedup 0.97, text max length 64, chuẩn hóa L2, exact Top-K trên index nhỏ.
5. Dữ liệu và protocol: nguồn/quyền, split theo video, 43 dev/14 test, coverage và cách chấm T02; nêu giới hạn cùng tác giả tạo/review nhãn.
6. Kết quả T06: bảng Hit@1/Hit@5/MRR, thời gian index/query và GPU peak từ [report](rewind-lite-task-06-benchmark.md); không đọc chênh lệch 1–2 query như bằng chứng vượt trội.
7. Bảy ca E3 trượt Hit@5: nhầm vật thể/giai đoạn hành động và một ca không có khung hình dương sau lấy mẫu/lọc trùng.
8. Giới hạn và bước tiếp: review nhãn độc lập, bổ sung nguồn/giấy phép hai video tải về, thêm video test và tái chạy benchmark trước công bố.

## Acceptance T02–T08 và kiểm tra T09

| Mục | Trạng thái | Bằng chứng hiện tại |
|---|---|---|
| T02 protocol | PASS | [Giao thức](rewind-lite-task-02-protocol.md), validator và unit tests |
| T03 model/index thật | PASS | CLI SigLIP2/CUDA dùng lại index `video-test.mp4`, trả hai frame 5.0 và 0.0 giây |
| T04 core | PASS | Hợp đồng index/retrieval và full test suite |
| T05 dữ liệu | READY cho benchmark nội bộ | 5 video, 57 nhãn `verified`, 0 lỗi validator; review độc lập còn thiếu |
| T06 benchmark | MEASURED, nội bộ | [JSON](benchmark-results/t06.json) có E1/E2/E3, từng query, coverage, latency, GPU, paired test; [7 ca lỗi](rewind-lite-task-06-benchmark.md) |
| T07 API | PASS | Flask test client: select/index/search/frame 200; video byte range 206; index `cached=true` |
| T08 UI | PASS trong phạm vi đã kiểm | Trang/CSS/JS 200 hiện tại; kiểm tra hình ảnh desktop/mobile và click-to-seek ở [T08](rewind-lite-task-08-ui.md) chưa lặp lại trong T09 |
| T09 setup | PASS trong môi trường tương đương | Python 3.11.15, CUDA RTX 3060 Laptop, `uv pip check` 40 gói tương thích; chưa tạo clone/venv sạch từ đầu |
| T09 tests | PASS | `pytest`: 353 passed, 1 skipped; validator 57 verified, 0 issue; T06 `--check`: `READY` |

Benchmark T06 lấy mẫu 407 frame, giữ 215 trong index. Hit@5 dev E1/E2/E3 là 33/43, 37/43, 36/43; test là 13/14, 13/14, 14/14. So sánh cặp Hit@5 không đạt p < 0.05. [README](../README.md) liên kết cấu hình, artifact và số đo; JSON T06 lưu SHA-256 manifest và labels, đã đối chiếu với file hiện tại. Video nguồn, model cache và index là artifact cục bộ bị Git ignore; người chạy lại cần có đúng video theo checksum manifest.

## Phiên bản, file handling và giới hạn

`requirements-gpu.txt` pin PyTorch 2.5.1+cu121, NumPy 1.26.4, Transformers 4.57.3, sentencepiece 0.2.1 và protobuf 5.29.5. Các gói core khác dùng lower bound; JSON T06 log phiên bản thực. `UV_CACHE_DIR` cần đặt trong workspace như README khi cache mặc định Windows không thể truy cập. Index pickle chỉ được nạp từ nguồn tin cậy; API giới hạn upload 512 MiB, kiểm tra MP4/ID/path và chỉ phục vụ frame thuộc index. Server mặc định bind `127.0.0.1`.

Audit Git cơ bản hiện có 119 file tracked: không có video, model weights, file key/`.env` theo đuôi/tên đã kiểm và không có file tracked trên 100 MiB. Rà soát tên/mẫu không thay cho secret scan chuyên dụng. Full clean install chưa thử; lần đầu cần tải hai model pinned và chuẩn bị năm video cục bộ. Không chuyển video hoặc model weights vào Git.

## Release checklist

| Gate | Trạng thái | Việc còn lại |
|---|---|---|
| Demo nội bộ với CLI/API/UI và video được phép dùng | READY | Kiểm tra MP4/âm thanh và cache trên máy trình bày trước giờ demo |
| Báo cáo T06 có số đo, config, predictions và phân tích lỗi | READY cho đánh giá nội bộ | Giữ JSON và checksums cùng đúng phiên bản labels/video |
| Người khác cài và tái chạy từ clone mới | CHƯA XÁC MINH | Thử clean install, chuẩn bị đúng năm video theo manifest và cache model |
| Công bố accuracy học thuật | BLOCKED | Người độc lập adjudicate 57 nhãn; xác minh nguồn/license hai video tải về; tái chạy T06 sau mọi chỉnh sửa nhãn |

## T09 — Acceptance Review

| Criterion | Status | Evidence |
|---|---|---|
| Setup sạch hoặc môi trường tương đương; CLI/API/UI hoạt động | PASS với giới hạn | Venv hiện có, CUDA CLI và Flask real-path smoke; clean install chưa chạy |
| Tests cần thiết và phần chưa chạy ghi rõ | PASS | 353 passed, 1 skipped; browser visual T08 không lặp lại, clean install không chạy |
| README có metrics thật, config và link artifact | PASS | [README](../README.md), [T06 report](rewind-lite-task-06-benchmark.md), [JSON](benchmark-results/t06.json) |
| Security/basic file handling, không tracked media/keys lớn | PASS ở mức audit cơ bản | API tests, `.gitignore`, 119 tracked files đã rà theo tên/đuôi/kích thước |
| Demo script, report outline và limitations | PASS | Hai phần đầu tài liệu này; các giới hạn ở trên |

Tests run trong lượt cập nhật: full pytest, `uv pip check` với cache workspace, validator `--require-ready`, T06 `--check`, CLI CUDA dùng lại index, Flask real-path smoke và kiểm tra hash JSON đầu vào/Git tracked files. Tests not run: clean install từ clone mới, browser visual T08 lặp lại và full T06 benchmark lặp lại; report T06 đã được đo, hash đầu vào còn khớp. Files changed: tài liệu này, README và `docs/task-status.md`. Risks: nhãn chưa được người độc lập adjudicate, hai URL/license nguồn thiếu, 14 test queries và các video không nằm trong Git. Scope drift: không đổi runtime hay ground truth.

**Kết luận phát hành:** bàn giao nội bộ/demo đã có bằng chứng chạy thật; công bố accuracy học thuật vẫn chờ review dữ liệu độc lập và xác minh provenance. Chủ repository đã duyệt commit và push checkpoint T09 lên `rewind-lite/main` ngày 2026-11-08.
