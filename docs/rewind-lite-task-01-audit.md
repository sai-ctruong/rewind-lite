# Rewind Lite — Task 01: Audit & Baseline Freeze

Ngày ghi nhận: 25/10/2026 (Asia/Bangkok)

## 1. Phạm vi và kết luận

Đề tài: **Định vị khoảnh khắc trong video bằng truy vấn ngôn ngữ tự nhiên tiếng Việt**. Phạm vi Rewind Lite là Video Moment Retrieval ở mức keyframe và timestamp, dùng pretrained Vision–Language Models, hỗ trợ GPU NVIDIA, có baseline, benchmark, metrics, error analysis và web demo đơn giản.

Repository đã có phần lõi có thể tái sử dụng: trích keyframe, mã hóa ảnh và truy vấn bằng SigLIP, lập chỉ mục, tìm kiếm và trả `video_id` cùng timestamp. Có thể thu gọn đường chạy quanh lõi này theo từng task; Task 01 chỉ khảo sát, không sửa source code, không xóa module và không tạo kết quả benchmark.

## 2. Baseline được ghi nhận

| Mục | Trạng thái tại thời điểm audit |
|---|---|
| Git | Nhánh `main`, commit `20a20262cfd37bc928d4b715915519c5f23a9f54`; working tree sạch trước khi tạo báo cáo này |
| Python / pytest | Python 3.14.5; pytest 9.1.1 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB; driver 546.92 |
| PyTorch | `2.14.0+cpu`; `torch.cuda.is_available()` trả `False` |
| Dependency thiếu trong môi trường | `flask`, `cv2`, `faiss`, `rank_bm25`, `transformers` |
| Bộ nhãn | `evaluation/labels.json`: 78 truy vấn tiếng Việt trên 3 video ID; `evaluation/labels_en.json`: 51 truy vấn |
| Dữ liệu thực nghiệm | Không thấy video nguồn hoặc báo cáo benchmark được Git theo dõi trong checkout |

`python -m pytest` dừng ở bước thu thập do thiếu Flask; 9 tệp test được skip do thiếu OpenCV hoặc Faiss. Chạy riêng `tests/test_metrics_phase9.py` với `--basetemp` trong workspace cho **14 pass, 2 fail**; hai lỗi do thiếu Faiss. Chưa đủ căn cứ kết luận toàn bộ test của repository đạt hay không đạt.

Đây là **baseline trạng thái repository và môi trường**, chưa phải baseline độ chính xác của mô hình. Không chạy benchmark trên video thật và không suy ra số đo mới từ README.

## 3. Dependency graph và pipeline hiện tại

```text
Video → ingestion/video_ingest.py → RawKeyframe
                                 → SigLIP image embeddings → dedup → KeyframeIndex
                                 → OCR / ASR / caption (tùy chọn) ───────┘

Truy vấn tiếng Việt → SigLIP text embeddings → CoarseRetriever
                                            → dense + BM25 + RRF
                                            → Qwen2-VL rerank (tùy chọn)
                                            → keyframe ID + video ID + timestamp
                                                   ↓
                                           Flask UI / benchmark
```

### Các thành phần chính

| Thành phần | File | Vai trò / quan hệ phụ thuộc |
|---|---|---|
| Video I/O | [`ingestion/video_ingest.py`](../ingestion/video_ingest.py) | Lấy mẫu theo thời gian, loại frame gần trùng bằng histogram, giữ timestamp và nguồn video để đọc lại frame |
| Schema | [`ingestion/schemas.py`](../ingestion/schemas.py) | `RawKeyframe`, `KeyframeRecord` và truy xuất ảnh |
| Encoder | [`ingestion/embed_siglip.py`](../ingestion/embed_siglip.py) | Mã hóa ảnh và text trong cùng không gian; lazy import PyTorch/Transformers, tự chọn CUDA khi khả dụng |
| Chỉ mục | [`ingestion/build_index.py`](../ingestion/build_index.py) | Faiss HNSW cho dense retrieval, BM25 cho text, metadata video/timestamp |
| Điều phối video | [`retrieval/video_engine.py`](../retrieval/video_engine.py) | Index một video hoặc dataset, dedup, tìm kiếm, rerank tùy chọn, lưu/nạp index |
| Truy xuất | [`retrieval/coarse_retriever.py`](../retrieval/coarse_retriever.py), [`retrieval/fusion.py`](../retrieval/fusion.py) | Kết hợp các danh sách dense và sparse bằng RRF |
| Rerank | [`retrieval/fine_rerank.py`](../retrieval/fine_rerank.py), [`retrieval/vlm_rerank.py`](../retrieval/vlm_rerank.py) | Qwen2-VL local tùy chọn; engine có fallback về coarse khi rerank lỗi |
| Web demo | [`ui/app.py`](../ui/app.py), [`ui/index.html`](../ui/index.html) | Flask API và giao diện; có sẵn nạp video, truy vấn chữ, keyframe/timestamp |
| Đánh giá | [`evaluation/bench_retrieval.py`](../evaluation/bench_retrieval.py), [`evaluation/metrics.py`](../evaluation/metrics.py) | Nhãn theo cửa sổ thời gian; Hit@K, Recall@K, MRR và kết quả từng query |

Luồng video hiện dùng mặc định hai encoder SigLIP; OCR bật, ASR và caption tắt; VLM rerank chỉ chạy khi yêu cầu. Ngoài đường chạy này, repository còn có dialogue, agent, VQA, tìm bằng ảnh, phản hồi relevance và truy xuất chuỗi thời gian. Các nhánh đó chưa cần xóa hoặc sửa để xác lập Lite.

### Dependency và cấu hình

- [`requirements.txt`](../requirements.txt) chứa các thư viện lõi và UI; [`requirements-full.txt`](../requirements-full.txt) thêm PyTorch, Transformers, EasyOCR và tiện ích cho VLM. Phiên bản chỉ có ngưỡng tối thiểu, chưa khóa môi trường thực nghiệm.
- [`pyproject.toml`](../pyproject.toml) khai báo Python `>=3.10` và cấu hình pytest, chưa khai báo bộ dependency cài đặt đầy đủ.
- [`configs/settings.yaml`](../configs/settings.yaml) lưu nhiều tham số, nhưng chính file này ghi rằng `VideoSearchEngine` không parse cấu hình đó. Các mặc định của engine đang được đồng bộ thủ công.
- [`ingestion/model_cache.py`](../ingestion/model_cache.py) mặc định đặt cache model ở `D:\hf_cache` nếu chưa có `HF_HOME`; cần làm rõ tính di động trong task cấu hình sau này.

## 4. Khả năng tách thành Rewind Lite

**Có thể giữ làm lõi:** `video_ingest`, schema keyframe, SigLIP encoder, dedup, index, coarse retrieval, metrics và benchmark theo nhãn thời gian. API trả `video_id`, `timestamp` và ảnh keyframe đã có.

**Có thể để nguyên ngoài đường demo Lite:** dialogue, agent, VQA, tìm bằng ảnh, phản hồi relevance và temporal search. Việc không dùng chúng trong đường chạy Lite không đồng nghĩa với xóa module hoặc dependency.

**Cần tách rõ bằng cấu hình/giao diện ở các task sau:** một baseline đơn giản, biến thể có BM25 và biến thể có VLM rerank; tập tham số được ghi lại cho từng lần đo; giao diện chỉ trình bày các chức năng thuộc đề tài.

## 5. Rủi ro và khoảng trống trước khi công bố baseline

1. **Đáp án sau dedup:** `VideoIndexEntry` giữ metadata cả frame bị loại, còn `relevant_ids()` trong `bench_retrieval.py` tính đáp án từ toàn bộ `entry.raws`. Cần kiểm tra ảnh hưởng lên mẫu số Recall@K và trường hợp cửa sổ nhãn không còn keyframe có thể truy xuất.
2. **Nhãn và tài liệu lệch nhau:** [`evaluation/labels.README.md`](../evaluation/labels.README.md) ghi 51 nhãn cho `labels.json`, nhưng file này hiện có 78. Các cửa sổ thời gian có cấu trúc hợp lệ theo kiểm tra tĩnh; chưa đối chiếu với video nguồn.
3. **Tính tái lập:** chưa có bộ thực nghiệm được khóa chung gồm video/dataset manifest, nhãn, cấu hình lấy mẫu, phiên bản model, dependency và báo cáo đo. Các số trong [`README.md`](../README.md) là kết quả được tài liệu hóa trước đây, không phải kết quả đo lại trong Task 01.
4. **GPU:** code có nhánh CUDA và cơ chế chuyển encoder khỏi VRAM khi rerank, nhưng môi trường hiện tại chỉ có PyTorch CPU. Chưa xác nhận bằng lần index/search thực trên NVIDIA GPU.
5. **Error analysis:** benchmark có dòng kết quả từng query nhưng chưa có báo cáo phân loại lỗi như lỗi lấy mẫu, encoder, xếp hạng, OCR hoặc nhãn.
6. **Web demo:** backend hiện chứa nhiều chức năng AIC ngoài phạm vi Lite. Cần giới hạn luồng trình bày khi đến task UI, không thay đổi chúng trong Task 01.

## 6. Kế hoạch refactor đề xuất

Mỗi hàng dưới đây là **một task riêng**, chỉ triển khai sau khi được yêu cầu và review task trước.

| Thứ tự | Task | Đầu ra để review |
|---|---|---|
| 02 | Đóng băng protocol thực nghiệm | Dataset manifest, kiểm tra nhãn/cửa sổ thời gian, quy tắc chấm sau dedup, cấu hình và test cho các quy tắc đó |
| 03 | Xác lập baseline Lite | Đường chạy keyframe → pretrained VLM → timestamp, cấu hình GPU rõ ràng; chỉ báo số đo sau khi chạy với video và nhãn thật |
| 04 | Benchmark và error analysis | Hit@1/5, Recall@K, MRR, thời gian index/query, kết quả từng query và phân loại lỗi có dẫn chứng |
| 05 | Web demo đơn giản | Nạp video, nhập truy vấn tiếng Việt, xem keyframe và timestamp; test API quan trọng |

## 7. Trạng thái Task 01

- Chỉ khảo sát repository và môi trường; không sửa source code, module, dependency hoặc kiến trúc.
- Không tạo kết quả benchmark hay khẳng định các số đo cũ đã được tái lập.
- File Markdown này là bản ghi audit để review trước task tiếp theo.
