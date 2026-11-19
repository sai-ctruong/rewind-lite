# Điểm dừng bước 2 — corpus v2

**Cập nhật sau khi tiếp tục:** Đã tải đủ model, sửa lỗi dấu thời gian trên Windows, chạy benchmark 21 video và lưu [kết quả đo](corpus-v2-baseline.md) cùng [JSON](benchmark-results/corpus-v2.json). Nội dung dưới đây là ảnh chụp tiến trình tại thời điểm người dùng yêu cầu tạm dừng, không phải trạng thái hiện tại.

Ngày 2026-11-19 (Asia/Bangkok). Người dùng yêu cầu lưu tiến trình và dừng. Làm việc trực tiếp trên `main`; **chưa commit hoặc push** các thay đổi của bước 1 và bước 2.

## Đã làm

- Bước 1: `data/annotations/corpus_v2_manifest.json`, `corpus_v2_labels.json`, `corpus_v2_absent.json`, `evaluation/absent_dataset.py`, `tests/test_absent_dataset.py`, `docs/corpus-v2-evaluation.md` và cập nhật `README.md`. Có 62 video trong manifest, 84 truy vấn có cảnh khớp, 20 truy vấn vắng mặt; dữ liệu video cục bộ tại `data/videos/corpus_v2/` và bị Git ignore.
- Bước 2: thêm `evaluation/corpus_v2_benchmark.py` và `tests/test_corpus_v2_benchmark.py`. Benchmark dự kiến đánh giá từng video được gán nhãn bằng cấu hình lấy mẫu/index giống web, chấm Hit@1/5, MRR, temporal tIoU và từ chối truy vấn vắng mặt. Ngưỡng cosine được chọn chỉ trên `dev`, sau đó áp dụng cho `test`.
- Ba kiểm thử mới của benchmark đã chạy qua: `.venv\Scripts\python.exe -m pytest tests\test_corpus_v2_benchmark.py -q`. Hai validator dữ liệu và 17 bài kiểm thử của bước 1 đã qua trước đó.

## Điểm dừng và nguyên nhân

Chưa có số đo benchmark. Lệnh thử đường model → index → tìm kiếm trên `video_62` dừng khi nạp `google/siglip2-base-patch16-256@6328738b83b563ff8396fdb87f8ce130a7b60d74`: cache trong `artifacts/hf-cache/` mới có 3/9 tệp, thiếu `preprocessor_config.json` và trọng số model. Tải tiếp snapshot đã được khởi động, nhưng người dùng ngắt lượt làm việc. Tiến trình tải đã nhận Ctrl-C và thoát; không còn tiến trình tải của lượt này. Cache có thể chứa tệp `.incomplete`; chưa có index mới hoặc báo cáo đo.

## Tiếp tục từ đây

1. Tải hoàn chỉnh snapshot model đã khóa vào `D:\Rewind\artifacts\hf-cache` (ổ D còn khoảng 26 GB khi kiểm tra); xác nhận có processor, tokenizer và trọng số model.
2. Chạy lại thử index/tìm kiếm `video_62` bằng `LiteService`. Nếu qua, chạy benchmark đầy đủ:

   ```powershell
   .venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --output artifacts\corpus-v2-benchmark\report.json
   ```

3. Kiểm tra báo cáo từng truy vấn và các trường hợp lỗi nhãn; ghi kết quả và giới hạn vào tài liệu. Chỉ dùng `dev` để chọn ngưỡng, giữ `test` làm phép đo cuối.
4. Chạy lại validator, kiểm thử liên quan và `git diff --check` trước khi tuyên bố hoàn thành. Không thêm `run_web_local.py`, `web.stderr.log` hoặc `web.stdout.log` vào thay đổi bước này; chúng đã có sẵn và không liên quan.

Các nhãn hiện là review nội bộ của Codex trên khung hình lấy mẫu, chưa có người thẩm định độc lập. Không dùng số liệu tương lai như kết quả ground truth đã xác nhận độc lập trước khi review nguồn video.
