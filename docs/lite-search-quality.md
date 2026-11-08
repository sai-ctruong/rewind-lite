# Kiểm tra chất lượng tìm kiếm web — 2026-11-08

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
