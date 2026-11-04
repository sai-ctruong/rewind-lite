# evaluation/labels.json — bộ nhãn đánh giá truy xuất video (B1)

Hiện có **78** mục trong `labels.json` và **51** mục trong `labels_en.json`.
`labels.json` có cả truy vấn tiếng Anh. Xem [audit và protocol T02](../docs/rewind-lite-task-02-protocol.md)
trước khi dùng để đo Recall@K / hit@K / MRR. Video nguồn chưa có trong checkout này.

## Cách tạo bộ nhãn này (để tin được số đo)

Tài liệu cũ mô tả việc xác định cửa sổ bằng contact sheet và xem frame lân cận.
T02 chưa thể kiểm chứng quá trình đó hay nội dung nhãn vì không có video nguồn.
Trước khi công bố số đo, đối chiếu từng mục theo quy trình thủ công trong audit T02.

Định dạng mỗi mục (khớp `bench_retrieval.load_labels`):
```json
{"query": "...", "video_id": "<tên file không đuôi>", "time_window": [t_start, t_end]}
```
`time_window` là **giây**; keyframe rơi vào [t_start, t_end) được coi là ĐÚNG.
Nhãn mới có thể dùng `time_windows` cho nhiều khoảng rời nhau và `id` ổn định;
xem `labels.schema.json` và `evaluation.dataset.validate_labels`.

## Nội dung 3 video (tham chiếu nhanh)

| video_id | thời lượng | nội dung |
|---|---|---|
| `beach_walking` | 22.7s | tour Sydney (montage nhanh): Opera House, Harbour Bridge, hồ bơi xanh, tram đỏ, JB HI-FI |
| `walking` | 10.5s | Quảng trường Thời Đại New York: taxi vàng, Raising Cane's, LANEIGE, đám đông |
| `walk_rain_seoul` | 53.3 phút | đi bộ Seoul trời mưa: phố neon, người che ô, Starbucks, xe buýt 2 tầng vàng… |

## Chạy benchmark

```powershell
$env:PYTHONUTF8=1
python -m evaluation.bench_retrieval --labels evaluation/labels.json
```

Lệnh này (trên GPU) sẽ:
1. Index 3 video (ensemble SigLIP, batch embedding A1, pipeline A4).
2. Đo **throughput embedding** (frame/giây LẺ vs THEO LÔ → lượng hoá A1).
3. Chấm **Recall@K / hit@K / MRR** trên các nhãn của file được chọn, sau khi
   xác minh video nguồn và manifest.
4. Lưu `evaluation/benchmarks/retrieval_benchmark.json`.

## Lưu ý & cách mở rộng

- Cửa sổ các cảnh phố/biển hiệu (tồn tại lâu) đặt ±7s; mục tiêu thoáng qua đặt hẹp
  (đã xác minh). Nếu muốn CHẶT hơn, xem lại frame quanh mốc rồi thu hẹp `time_window`.
- Thêm nhãn: cứ nối thêm phần tử vào mảng JSON (nhớ đúng `video_id` = tên file).
- Muốn quét tham số (sample_every_s / efSearch / embed_batch_size) để chọn điểm
  "khuỷu tay" (Mục 11.3): tạo nhiều `VideoSearchEngine` cấu hình khác nhau rồi gọi
  `evaluate_labeled` cho từng cái — hạ tầng đã sẵn trong `bench_retrieval.py`.
