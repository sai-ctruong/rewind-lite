# Rewind Lite

**Định vị khoảnh khắc trong video bằng truy vấn ngôn ngữ tự nhiên tiếng Việt**

Repository phục vụ tiểu luận chuyên ngành Artificial Intelligence. Mục tiêu là nhận một câu mô tả tiếng Việt và trả về các keyframe phù hợp, kèm video và timestamp. Hệ thống dùng pretrained Vision–Language Models và hướng tới chạy trên GPU NVIDIA.

> **Trạng thái:** đang refactor từ Rewind. Task 01 đã hoàn thành audit; chưa có baseline hiệu năng mới được đo và xác nhận cho Rewind Lite. Đọc [báo cáo audit](docs/rewind-lite-task-01-audit.md) trước khi dùng các số liệu hoặc cấu hình cũ.

## Phạm vi nghiên cứu

- Video Moment Retrieval ở mức keyframe và timestamp.
- Truy vấn tiếng Việt, mã hóa ảnh và văn bản bằng pretrained Vision–Language Models.
- Baseline, benchmark, metrics và error analysis trên nhãn thời gian có thể kiểm chứng.
- Web demo đơn giản để nạp video, tìm kiếm và xem kết quả.

Repository hiện còn các nhánh chức năng từ Rewind như agent, VQA, tìm bằng ảnh và temporal search. Chúng chưa phải phạm vi cốt lõi của Rewind Lite và chưa được xóa trong Task 01.

## Pipeline hiện có

```text
Video → lấy mẫu keyframe → SigLIP image embeddings → dedup → Faiss HNSW
                                                        ↘ BM25 từ OCR/ASR/caption (tùy chọn)
Truy vấn tiếng Việt → SigLIP text embeddings → dense + sparse retrieval → RRF
                                                            → Qwen2-VL rerank (tùy chọn)
                                                            → keyframe + video ID + timestamp
```

Các điểm vào chính:

| Thành phần | File |
|---|---|
| Trích keyframe | [`ingestion/video_ingest.py`](ingestion/video_ingest.py) |
| Encoder ảnh và text | [`ingestion/embed_siglip.py`](ingestion/embed_siglip.py) |
| Chỉ mục | [`ingestion/build_index.py`](ingestion/build_index.py) |
| Điều phối index và tìm kiếm | [`retrieval/video_engine.py`](retrieval/video_engine.py) |
| Benchmark theo nhãn thời gian | [`evaluation/bench_retrieval.py`](evaluation/bench_retrieval.py) |
| Metrics | [`evaluation/metrics.py`](evaluation/metrics.py) |
| Web demo hiện tại | [`ui/app.py`](ui/app.py) |

## Chạy thử mã nguồn hiện có

Yêu cầu Python 3.10 trở lên. `requirements.txt` dùng cho test và web UI; `requirements-full.txt` thêm model và OCR để xử lý video thật. Hai file này chưa khóa phiên bản cho một protocol thực nghiệm tái lập.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

Để chạy model thật, cài `requirements-full.txt` và một bản PyTorch hỗ trợ CUDA phù hợp với máy nếu dùng GPU NVIDIA. Kiểm tra bằng `torch.cuda.is_available()` trước khi đo hiệu năng. Lần đầu chạy có thể cần tải weights của model.

```powershell
python -m pip install -r requirements-full.txt
python -m retrieval.video_search_demo path\to\video.mp4 "người đang đi bộ trên phố" --topk 5
python -m ui.app
```

Web demo hiện có tại `http://127.0.0.1:5000`; giao diện còn chứa chức năng từ Rewind và sẽ được thu gọn trong task riêng. Đặt video trong `data/videos/` nếu dùng luồng nạp video của web UI. Thư mục `data/` và artifact sinh ra khi index không được đưa vào Git.

## Đánh giá

[`evaluation/labels.json`](evaluation/labels.json) hiện có 78 truy vấn tiếng Việt với `video_id` và `time_window`. [`evaluation/labels_en.json`](evaluation/labels_en.json) có 51 truy vấn tiếng Anh. Video nguồn không có trong repository này; vì vậy không thể chạy lại benchmark thực chỉ từ bản clone.

[`evaluation/bench_retrieval.py`](evaluation/bench_retrieval.py) có hàm chấm Hit@K, Recall@K và MRR theo cửa sổ thời gian. Trước khi công bố baseline mới cần thống nhất cách chấm sau dedup, kiểm tra nhãn với video nguồn, lưu cấu hình/model/dependency và ghi nhận lỗi theo từng truy vấn. **Không coi số đo trong lịch sử Rewind là kết quả của Rewind Lite.**

## Tiến độ

1. **Task 01 — Audit & Baseline Freeze:** [báo cáo](docs/rewind-lite-task-01-audit.md).
2. Task 02 — Đóng băng protocol thực nghiệm và kiểm tra nhãn.
3. Task 03 — Xác lập baseline Lite trên video thật.
4. Task 04 — Benchmark, metrics và error analysis.
5. Task 05 — Thu gọn web demo.

Mỗi task được thực hiện và review riêng. Kết quả benchmark chỉ được ghi sau khi chạy thực nghiệm trên dữ liệu và môi trường đã mô tả.
