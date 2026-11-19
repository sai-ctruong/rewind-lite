# Bộ dữ liệu đánh giá corpus v2

Ngày lập: 2026-11-19 (Asia/Bangkok). Mục đích: bước 1 của kế hoạch nâng chất lượng đồ án — lập bộ video và truy vấn có thể kiểm tra lại, tách khỏi năm video T05 cũ.

Bước 2 đã đo baseline trên 21 video có nhãn: xem [báo cáo và giới hạn](corpus-v2-baseline.md), cùng [JSON dự đoán từng truy vấn](benchmark-results/corpus-v2.json).

## Nguồn và phạm vi

Người dùng xác nhận họ tự quay và cho phép dùng 62 tệp `video_1.mp4` đến `video_62.mp4` trong `D:\Videos\Videos` để đánh giá và trích hình cho báo cáo. Bản sao phục vụ đánh giá nằm tại `data/videos/corpus_v2/`; video không được đưa lên Git. [`corpus_v2_manifest.json`](../data/annotations/corpus_v2_manifest.json) ghi đường dẫn, SHA-256 từng tệp, thời lượng, FPS, số khung hình, nguồn, quyền sử dụng và split. Tổng thời lượng đo bằng OpenCV là 138,7 phút; các video đều 640×480, 25 FPS. Quyền sử dụng được ghi theo lời xác nhận của người cung cấp, chưa có giấy tờ độc lập.

Video được chia theo **nhóm người xuất hiện**, không chia ngẫu nhiên theo khung hình: 42 `dev`, 20 `test`, gồm chín nhóm người dự kiến. Nhóm của cùng một người được giữ trọn trong một split. Split `test` gồm ba nhóm, trong đó một nhóm chiếm 13/20 video; khi báo cáo nên kèm kết quả theo nhóm thay vì chỉ gộp tất cả câu hỏi. Việc nhận diện người qua các clip là quan sát thị giác sơ bộ; cần chủ video hoặc người kiểm tra độc lập xác nhận trước khi công bố kết quả. Một số clip có thêm người bước vào; người đó cũng có thể xuất hiện ở clip khác, nên reviewer cần kiểm tra khả năng rò rỉ qua nhân vật phụ.

Các cảnh chính là ăn sáng, ăn trái cây và làm việc với giấy/máy tính trong cùng một phòng. Đây là phân bố hẹp; bộ dữ liệu không đại diện cho video ngoài đời hoặc các bối cảnh khác.

## Truy vấn và nhãn

[`corpus_v2_labels.json`](../data/annotations/corpus_v2_labels.json) chứa **84 truy vấn tiếng Việt có cảnh khớp**: 64 `dev`, 20 `test`. Các truy vấn bao phủ 21 video: `1, 6, 8, 11, 16, 18, 21, 24, 26, 28, 30, 33, 35, 38, 40, 43, 48, 50, 53, 58, 62`. Mỗi truy vấn ghi video, mốc `[start, end)` theo giây, loại câu hỏi và thông tin người rà soát. Có 46 câu hành động, 14 câu vật thể, 11 câu bối cảnh, 7 câu sự kiện ngắn và 6 câu màu sắc. Chưa có câu đếm đủ chắc chắn; không nên báo cáo chất lượng riêng cho loại đếm.

[`corpus_v2_absent.json`](../data/annotations/corpus_v2_absent.json) chứa **20 truy vấn tiếng Việt về cảnh vắng mặt**: 12 `dev`, 8 `test`. Câu truy vấn dùng cách diễn đạt tìm kiếm bình thường; `expected_presence: "absent"` và `time_windows: []` là nhãn kỳ vọng. Tệp này được kiểm tra riêng vì benchmark T06 hiện hành chỉ chấm truy vấn có khoảng thời gian đúng; không đưa 20 câu vắng mặt vào T06 như thể chúng có khoảng `[start, end)`.

Tổng cộng có **104 truy vấn** trên 21/62 clip; 41 clip chưa có truy vấn riêng nhưng vẫn được ghi trong manifest và có thể mở rộng về sau. Ảnh tổng quan và contact sheet theo mốc thời gian lưu cục bộ tại `artifacts/dataset-v2/`. Codex đã xem 16 khung hình phân bố đều trên mỗi video được chọn, khung hình giữa khoảng của cả 84 nhãn dương và khung hình gốc cho một số trường hợp không rõ. Lượt xem giữa khoảng dẫn tới việc sửa 11 mô tả hoặc loại câu chưa được khung hình hỗ trợ. Codex soạn và tự rà soát cả 104 câu **không xem dự đoán tìm kiếm**. Trạng thái `verified` trong JSON chỉ có nghĩa là lượt rà soát nội bộ này; nó **không** đồng nghĩa với người gán nhãn độc lập hoặc xem liên tục toàn bộ video. Ranh giới sự kiện dưới một giây, các lần lặp khác của cùng hành động và sự vắng mặt giữa các khung hình lấy mẫu chưa được xác nhận đầy đủ.

## Kiểm tra lại

Chạy từ thư mục gốc dự án sau khi đặt 62 MP4 theo đường dẫn trong manifest:

```powershell
.venv\Scripts\python.exe -m evaluation.dataset --manifest data\annotations\corpus_v2_manifest.json --labels data\annotations\corpus_v2_labels.json --root . --every 1 --require-ready
.venv\Scripts\python.exe -m evaluation.absent_dataset --manifest data\annotations\corpus_v2_manifest.json --absent data\annotations\corpus_v2_absent.json --positives data\annotations\corpus_v2_labels.json
```

Kết quả ngày lập: 62 video, 84/84 nhãn dương có thông tin review, 84/84 cửa sổ chạm lưới lấy mẫu danh nghĩa 1 giây, 0 lỗi cấu trúc, `ready_for_accuracy=true` theo gate T05; 20/20 truy vấn vắng mặt có thông tin review, 0 lỗi cấu trúc. Gate chỉ kiểm tra nguồn, metadata, split và hình thức nhãn; không xác nhận sự thật ngữ nghĩa.

SHA-256 của ba tệp để khóa phiên bản đầu tiên:

| Tệp | SHA-256 |
|---|---|
| `corpus_v2_manifest.json` | `4e872ae58a9df1ead886a8c772e1b3a0dc65de7cc7c7d1fbced4241c851b6fa6` |
| `corpus_v2_labels.json` | `74a875b72dc2158ca8abb5881a02aa758bc93e46f1d2f32c5b67416625f06be5` |
| `corpus_v2_absent.json` | `b3789e952f6135379f9484e66da69feddffe7822f6840445dc9b77ff73c3ccf7` |

## Quy tắc dùng trong bước đánh giá tiếp theo

1. Khóa test và các checksum ở trên trước khi điều chỉnh mô hình/ngưỡng. Chỉ dùng `dev` để chọn tham số.
2. Với **định vị trong một video**, giới hạn tìm kiếm vào `video_id` của câu hỏi và chấm top-k giao với một trong các khoảng đúng. Với **tìm kiếm toàn kho**, phải gán nhãn mọi clip có thể khớp với cùng câu hỏi trước khi chấm; các clip cùng người và hành động tương tự có thể là đáp án đúng ngoài video được ghi ở một hàng.
3. Truy vấn vắng mặt cần một ngưỡng từ chối và chỉ số false-positive/abstention riêng; bộ benchmark T06 chưa có phép đo này.
4. Trước khi đưa số liệu vào báo cáo như ground truth đã thẩm định, nhờ một người khác xem toàn bộ 21 clip đã gán nhãn, xác nhận các mốc và sự vắng mặt, kiểm tra nhóm người ở cả 62 clip, sửa các hàng sai rồi chạy lại hai validator. Ghi tên người review, ngày và các thay đổi. Không dùng kết quả dự đoán trên test để sửa nhãn test theo hướng có lợi cho mô hình.

Các video gốc và contact sheet vẫn ở máy cục bộ; Git chỉ lưu metadata và nhãn. Khi sao chép dự án sang máy khác, cần cung cấp video bằng kênh được chủ video cho phép, rồi xác nhận lại checksum bằng validator.
