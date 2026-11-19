# Kiểm tra chất lượng tìm kiếm web — 2026-11-08

## Cập nhật lấy mẫu độ phủ cao — 2026-11-19

Hai kiểm thử hồi quy tạo video MP4 thật bằng OpenCV: (1) chi tiết chỉ xuất hiện từ 1,2 đến 1,8 giây, nằm giữa các mốc lấy mẫu 1 giây; (2) một đốm màu nhỏ xuất hiện đúng mốc 1 giây nhưng bị lọc theo histogram toàn khung. Trước thay đổi, cả hai truy vấn đều trả Top-1 tại giây 0 thay vì lúc chi tiết xuất hiện. Web nay lấy mẫu mỗi 0,5 giây khi đủ ngân sách và chỉ gộp khung hình có tương quan histogram hoặc cosine embedding bằng 1.0. Hai truy vấn tổng hợp trả đúng Top-1 sau khi xây lại chỉ mục. CLI và benchmark T06 không đổi.

Giới hạn: kiểm thử tổng hợp xác nhận cảnh ngắn còn trong chỉ mục và có thể tìm thấy; nó không đo cải thiện độ chính xác trên 57 truy vấn thật. Cần video T05 cục bộ để chạy lại benchmark với đúng model revision và so sánh Top-1, R@1/tIoU, thời gian lập chỉ mục. Giữ nhiều frame hơn có thể làm tăng thời gian lập chỉ mục và số kết quả gần trùng; trên video dài, bước lấy mẫu vẫn tăng theo ngân sách 512 frame.

Đã kiểm tra thêm trên ba clip người dùng cung cấp tại `D:\Videos\Videos` bằng SigLIP2 revision cục bộ `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` (khác revision đã chốt trong T06). Số frame trong index trước → sau: `video_6` 39 → 122, `video_57` 32 → 108, `video_62` 10 → 31. Tôi xem trực tiếp các khung hình ở mốc kết quả và chấm thăm dò bốn truy vấn tiếng Anh theo timestamp Top-1 của chế độ keyframe:

| Clip và truy vấn | Cảnh đúng đã xem | Trước | Sau |
|---|---:|---:|---:|
| `video_6`: `a woman drinking from a cup` | 26–27 s | 16 s, sai | 16,8 s, sai |
| `video_57`: `a woman leaving the room` | 50–52 s | 51 s, đúng | 50,88 s, đúng |
| `video_62`: `a woman opening a laptop` | 1–3 s | 1 s, đúng | 0,96 s, đúng |
| `video_62`: `a woman typing on a laptop` | 8–14 s | 9 s, đúng | 13,44 s, đúng |

Như vậy, kiểm tra nhỏ này **không cho thấy tăng Top-1** (3/4 ở cả hai cấu hình). Với truy vấn uống cốc, chế độ temporal đưa cảnh đúng vào hạng 2 ở cả hai cấu hình; lỗi xếp hạng hạng 1 còn tồn tại. Encoder đa ngôn ngữ E2 đã thử trên cùng clip cũng xếp một cảnh sai lên đầu, nên chưa có cơ sở đổi model mặc định. Đây là kiểm tra thăm dò trên clip khác bộ T05, không phải benchmark độc lập; video T05 đã bị xóa nên không thể chạy lại 57 nhãn cũ. Lợi ích đã chứng minh của sửa đổi là khả năng giữ và tìm cảnh ngắn bị bộ lấy mẫu/lọc trùng cũ bỏ qua, không phải tăng điểm Top-1 tổng quát.

## Lỗi đã tái hiện

Phiên web dùng `max_frames=64` theo nghĩa **dừng sau 64 keyframe đầu**. `walk-bangkok.mp4` dài 2037,467 giây; manifest cũ chỉ chứa 64 keyframe của phút đầu. Truy vấn về phần sau video vì thế không thể trả đúng thời điểm. Trên `video-test.mp4` dài 23,86 giây, histogram 0.985 và semantic dedup 0.97 chỉ để lại 2 keyframe ở giây 0 và 5.

## Thay đổi và kiểm tra

Web dùng ngân sách mặc định 512 keyframe, tăng bước lấy mẫu theo thời lượng để phủ toàn bộ video, và dùng ngưỡng histogram 0.999 / semantic 0.99. Manifest ghi bước lấy mẫu thực tế; index có cấu hình cũ phải xây lại. CLI và benchmark T06 vẫn dùng cấu hình đã đo trước đây.

Đã xây lại trên máy demo: `walk-bangkok` có 510 keyframe từ giây 0 đến 2036, bước 4 giây; `video-test` có 19 frame sau lọc histogram và 17 frame trong index. API tìm kiếm `walk-bangkok` trả được timestamp ở phút 27–31; API tìm kiếm `video-test` và đọc lại index cache đều hoạt động. Bộ test có ca hồi quy về chuyển động nhỏ, phủ toàn thời lượng và từ chối cache cũ. Dải ảnh 24 mốc một giây của `video-test` đã được xem trực tiếp.

## Giới hạn còn thấy ở `video-test`

Bảy truy vấn đã gắn nhãn đều có ít nhất một kết quả Top‑5 trong cửa sổ nhãn, nhưng phần lớn cửa sổ trải gần hết video; chỉ số đó không đo tốt độ đúng của thứ hạng. Ví dụ truy vấn “người mặc áo khoác xanh dương đi gần máy quay” trả giây 1 ở hạng 1 và giây 21 ở hạng 4, dù chủ thể gần máy quay hơn ở đoạn cuối. Thử encoder E2 đa ngôn ngữ trên cùng video cũng xếp giây 1 ở hạng 1; thử chấm điểm crop nửa dưới/trái/giữa/phải cũng xếp crop bên phải của giây 1 cao hơn giây 21. Hai thử nghiệm này chưa hỗ trợ thay model hay thêm crop reranking. Chỉ mục mới sửa việc thiếu thời điểm; encoder ảnh–văn bản trên keyframe vẫn có thể nhầm đối tượng nhỏ, quan hệ không gian và hành động. Video dài lấy mẫu thưa hơn nên sự kiện rất ngắn giữa hai mốc cũng có thể bị bỏ sót. Chưa đo lại benchmark T06 với cấu hình web mới và chưa đưa ra tuyên bố tăng độ chính xác.

## Acceptance Review

| Tiêu chí | Trạng thái | Bằng chứng |
|---|---|---|
| Phủ thời lượng video dài | PASS | `walk-bangkok` 510 frame, timestamp 0–2036 trên video 2037,467 giây; test ngân sách 3 frame trải đều trên clip 12 giây |
| Giữ thêm chuyển động nhỏ | PASS | `video-test` 17 frame searchable thay vì 2; test video tổng hợp có vật thể nhỏ di chuyển |
| Cache cũ không dùng sai cấu hình | PASS | Test API nhận `index_expired`, xây lại thành công và lần sau `cached=true` |
| Độ đúng ngữ nghĩa Top‑1 trên từng cảnh | CHƯA ĐẠT | Ví dụ áo xanh nêu trên vẫn xếp thời điểm chưa phù hợp lên đầu |

Tests run: `.venv\Scripts\python.exe -m pytest -o addopts='' -q` — 356 passed, 1 skipped; `git diff --check` — exit 0; Flask API thật — trang 200, index/search/cache hoạt động. Tests not run: benchmark T06 với cấu hình web mới, vì T06 là baseline cố định và chưa có bộ nhãn độc lập cho tuning. Files changed: `retrieval/video_engine.py`, `ui/lite_app.py`, `tests/test_lite_api_t07.py`, `README.md`, `docs/rewind-lite-task-09-release.md`, `docs/task-status.md`, tài liệu này. Risks: lỗi thứ hạng ngữ nghĩa còn tồn tại; lấy mẫu 4 giây trên video dài có thể bỏ lỡ hành động ngắn. Scope drift: sửa cấu hình web và cách áp dụng ngân sách frame ở web; CLI và benchmark T06 không đổi. Decision: chủ repository đã duyệt push checkpoint này lên `rewind-lite/main` ngày 2026-11-08, với giới hạn Top‑1 đã ghi rõ.
