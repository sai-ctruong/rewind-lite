# Thử nghiệm ranh giới thời gian trên video chưa gán nhãn trước đó

Đo ngày 2026-11-19 bằng đường tìm kiếm của web. Quy tắc giới hạn mỗi kết quả quanh đỉnh điểm số tối đa 8 giây được chọn từ `dev` trước khi tạo và đo `test` mới. Trên `test`, quy tắc này cải thiện nhẹ R@5/tIoU≥0,5 nhưng làm kết quả đầu tiên kém đi. **Web giữ thuật toán temporal cũ làm mặc định**; giới hạn 8 giây chỉ còn là tham số thí nghiệm.

## Dữ liệu và cách đo

- Giữ [manifest 62 video](../data/annotations/corpus_v2_manifest.json) và 64 nhãn dương `dev` từ v2. [Nhãn v3](../data/annotations/corpus_v3_labels.json) thay toàn bộ nhãn `test` cũ bằng 15 truy vấn hành động trên năm video chưa từng được dùng trong các phép chọn cấu hình: `video_2`, `video_3`, `video_12`, `video_13`, `video_25`. Các video này được chọn trước khi xem dự đoán của chúng. Người trong video thuộc ba nhóm `test` cũ; đây là phép đo video mới trong cùng kho, không phải một miền dữ liệu mới.
- [Nhãn vắng mặt v3](../data/annotations/corpus_v3_absent.json) giữ 11 hàng `dev` và bỏ `v2-abs-012` đã được xác nhận gán sai. Chưa tạo nhãn vắng mặt `test` mới vì cần xem liên tục toàn video mới xác nhận được sự vắng mặt. Vì vậy phép đo này không có specificity hay balanced accuracy `test`.
- Codex tự soạn và rà soát nhãn dương từ contact sheet cách 10 giây trên toàn video và cách 2 giây quanh sự kiện. Metadata ghi rõ cách review. Một số hành động lặp lại có thể còn thiếu khoảng đúng; chưa có người thứ hai xem liên tục video. Các số liệu là **đánh giá nội bộ**, không phải ground truth độc lập.
- Cùng model SigLIP2, index lấy mẫu 0,5 giây, tối đa 512 frame/video, detector tắt, top 5 và keyframe cách nhau ít nhất 5 giây. Mỗi câu chỉ tìm trong video nguồn của nó. Hai lượt đo khác nhau duy nhất ở `temporal_peak_cap_s`: `None` và `8.0`. Ngưỡng presence được chọn chỉ từ `dev`.

Validator báo 79/79 nhãn dương `verified` có video, checksum và khoảng thời gian hợp lệ, 11 nhãn vắng mặt đúng cấu trúc. Kiểm tra cấu trúc không chứng minh nội dung nhãn chính xác. [JSON baseline](benchmark-results/corpus-v3-baseline.json) và [JSON giới hạn 8 giây](benchmark-results/corpus-v3-cap8.json) lưu checksum, cấu hình, dự đoán và điểm từng truy vấn.

## Kết quả

| Split và chỉ số temporal | Không giới hạn | Giới hạn 8 giây |
|---|---:|---:|
| `dev` R@1, tIoU≥0,5 | 10/64 | 15/64 |
| `dev` R@5, tIoU≥0,5 | 14/64 | 21/64 |
| `dev` R@5, tIoU≥0,3 | 43/64 | 40/64 |
| `dev` mIoU top 1 | 0,242 | 0,251 |
| **`test` R@1, tIoU≥0,5** | **1/15** | **0/15** |
| **`test` R@5, tIoU≥0,5** | **6/15** | **7/15** |
| `test` R@5, tIoU≥0,3 | 8/15 | 8/15 |
| **`test` mIoU top 1** | **0,068** | **0,033** |

Trên `test`, giới hạn 8 giây giúp ba câu (`v3-vi-001`, `010`, `012`) vượt ngưỡng R@5/tIoU≥0,5 và làm hai câu (`003`, `013`) tụt xuống dưới ngưỡng. `v3-vi-003` từng có top 1 tIoU 0,865, sau khi cắt còn 0,491. Do top 1 kém hơn và mẫu chỉ có 15 câu, chưa có bằng chứng đủ để đổi mặc định cho người dùng. Keyframe không chịu tác động: Hit@1 là 2/15, Hit@5 là 9/15 ở cả hai lượt. Presence sensitivity là 6/15 ở cả hai lượt; không suy ra khả năng từ chối câu vắng mặt trên `test`.

Không dùng 15 câu `test` này để chọn thêm ngưỡng hoặc sửa nhãn theo dự đoán. Nếu tiếp tục thay thuật toán, cần chọn cấu hình trên `dev` và đo trên các video `test` khác chưa từng xem dự đoán. Các kết quả cũng chưa phản ánh tìm kiếm toàn kho, vì mỗi truy vấn chỉ chấm trong một video và các cảnh ăn sáng lặp lại nhiều.

## Chạy lại

Đặt MP4 theo đường dẫn trong manifest và chạy:

```powershell
.venv\Scripts\python.exe -m evaluation.dataset --manifest data/annotations/corpus_v2_manifest.json --labels data/annotations/corpus_v3_labels.json --root . --every 1 --require-ready
.venv\Scripts\python.exe -m evaluation.absent_dataset --manifest data/annotations/corpus_v2_manifest.json --absent data/annotations/corpus_v3_absent.json --positives data/annotations/corpus_v3_labels.json
.venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --labels data/annotations/corpus_v3_labels.json --absent data/annotations/corpus_v3_absent.json --keyframe-min-gap-s 5 --output artifacts/corpus-v3-baseline.json
.venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --labels data/annotations/corpus_v3_labels.json --absent data/annotations/corpus_v3_absent.json --keyframe-min-gap-s 5 --temporal-peak-cap-s 8 --output artifacts/corpus-v3-cap8.json
```
