# Đánh giá tìm kiếm trên toàn kho video

Phép đo này chạy đúng đường `LiteService.search_all` của web: một câu tiếng Việt được tìm trên **62/62 video đã có chỉ mục**, rồi lấy 5 khung hình đứng đầu của toàn kho. Bộ truy vấn gồm 64 câu `dev` từ `corpus_v3_labels.json`; 15 câu `test` không được chạy hoặc chấm. Tệp chỉ mục và video ở máy cục bộ, còn [pool dự đoán](benchmark-results/corpus-global-dev-pool.json), [nhãn cho từng kết quả](../data/annotations/corpus_global_dev_judgments.json) và [bảng điểm](benchmark-results/corpus-global-dev-score.json) được lưu trong repository. Pool ghi mã model, revision, cấu hình lấy mẫu, thời điểm chạy và SHA-256 của manifest/nhãn nguồn.

## Quy tắc nhãn

Mỗi truy vấn gốc chỉ xác nhận một khoảng thời gian trong một video. Kết quả ở video khác có thể cũng đúng. Vì vậy, kết quả nằm trong khoảng nguồn được gán `relevant` từ nhãn nguồn; các kết quả còn lại bắt đầu ở trạng thái `unjudged`, kể cả kết quả trong cùng video nhưng ngoài khoảng đó. Không suy diễn `irrelevant` từ việc video khác tên nguồn.

Codex xem các khung hình quanh **kết quả hạng 1** và khung hình nguồn: trước tiên lấy tại `t−1, t, t+1`; với trường hợp khó, xem thêm `t−4, t−2, t, t+2, t+4`. Chỉ những trường hợp nhìn rõ được gán `relevant` hoặc `irrelevant` kèm người review, ngày và cách review trong JSON. Ba truy vấn `v2-vi-007`, `v2-vi-053`, `v2-vi-055` còn `unjudged` ở hạng 1. Các kết quả hạng 2–5 phần lớn chưa được review. Contact sheet sinh từ video cá nhân chỉ lưu cục bộ tại `artifacts/corpus-global-dev/review/`.

## Kết quả hiện tại

| Chỉ số | Cận dưới | Cận trên | Truy vấn đã xác định |
| --- | ---: | ---: | ---: |
| Hit@1 | 24/64 = 37,50% | 27/64 = 42,19% | 61/64 |
| Hit@5 | 26/64 = 40,63% | 64/64 = 100% | 26/64 |

Có 60 phán đoán thủ công ở hạng 1; 253/320 kết quả trong pool vẫn `unjudged`. Cận dưới tính `unjudged` là chưa chứng minh đúng; cận trên tính chúng là có thể đúng. Đây là **khoảng của phép đo nội bộ**, chưa phải điểm chính xác cuối cùng. Cận trên Hit@5 còn rất rộng vì chưa review đủ hạng 2–5. Không dùng số Hit@5 này để khẳng định chất lượng hoặc so sánh mô hình. Nhãn nguồn và review đều do Codex thực hiện, chưa có người kiểm tra độc lập. Video thuộc split `test` vẫn nằm trong 62 video được tìm, nên kết quả này chỉ là chẩn đoán nội bộ; không dùng để tune rồi gọi 15 truy vấn `test` là đánh giá độc lập hoàn toàn.

Trong 64 kết quả hạng 1 có 24 `relevant`, 37 `irrelevant`, 3 `unjudged`. Theo loại câu hỏi, kết quả hạng 1 chưa xác định gồm một câu vật thể, một câu hành động và một sự kiện ngắn. Cảnh màu sắc đúng 4/6, nhưng mẫu nhỏ; không suy rộng tỷ lệ này cho video khác.

## Chạy lại

Sau khi đặt 62 video tại `data/videos/corpus_v2/` và có chỉ mục hợp lệ tại `artifacts/corpus-v2-benchmark/indexes/`:

```powershell
.venv\Scripts\python.exe -m evaluation.corpus_global_benchmark collect
```

Lệnh tạo `artifacts/corpus-global-dev/pool.json` và một mẫu `judgments.json` mới. Nó **không mang theo** phán đoán thủ công cũ; cần đối chiếu các hit theo `(query_id, video_id, frame_id, timestamp)` trước khi chuyển review cũ sang pool mới. Để chấm snapshot đã lưu trong Git mà không cần video/model:

```powershell
.venv\Scripts\python.exe -m evaluation.corpus_global_benchmark score --pool docs/benchmark-results/corpus-global-dev-pool.json --judgments data/annotations/corpus_global_dev_judgments.json --output artifacts/corpus-global-dev/recheck.json
```

Validator từ chối pool thiếu video được tìm, nhãn không khớp hit và phán đoán thủ công thiếu thông tin review. Muốn thu hẹp Hit@5, cần tiếp tục xem các kết quả hạng 2–5 cho những truy vấn chưa có hit đúng; chỉ sau khi toàn bộ ứng viên cần thiết được xác định mới báo điểm Hit@5 duy nhất. Điều này không thay thế việc một người khác xem clip nguồn và xác nhận nhãn.
