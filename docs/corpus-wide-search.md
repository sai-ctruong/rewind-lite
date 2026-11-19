# Tìm kiếm trên toàn bộ video có chỉ mục

Web có lựa chọn **Toàn bộ video có chỉ mục · Khung hình**. Chế độ này mã hóa câu truy vấn một lần, tìm trong từng chỉ mục hợp lệ bằng cùng encoder, gộp kết quả theo điểm cosine và chọn tối đa 5 mốc khác nhau. Khoảng cách tối thiểu 5 giây chỉ áp dụng giữa các mốc trong cùng một video. Khi bấm kết quả, giao diện mở video được ghi trên kết quả rồi tua tới timestamp.

`POST /api/lite/search_all` nhận JSON `{"query":"người đang đi bộ","top_k":5}`. `top_k` từ 1 đến 50. Response có `results` với `video_id`, `frame_id`, `timestamp`, `rank`, `score`, `score_kind`, `image_url`; `total_videos`, `searched_videos` và `skipped` chứa `video_id` cùng `reason` (`index_missing`, `index_incomplete`, `index_expired`). Không có chỉ mục hợp lệ thì API trả 409 `index_missing`. `GET /api/lite/videos` trả thêm `saved_index_videos` để giao diện báo độ phủ trước khi tìm; danh sách này chỉ xác nhận có manifest, còn kết quả tìm sẽ xác minh checksum và cấu hình.

Chỉ mục không nằm trong Git. Trên máy mới, đặt MP4 trong thư mục video mà server cấu hình và lập chỉ mục từng video trước khi tìm toàn kho. Chế độ này không tự lập chỉ mục trong lúc tìm, tránh một truy vấn HTTP phải chờ xử lý cả kho. Tìm khoảng thời gian và bằng chứng vật thể YOLO vẫn giới hạn ở video đang chọn.

Độ phủ và thời gian phản hồi chỉ là kiểm tra chức năng, không phải độ chính xác. Lượt chạy cục bộ ngày 2026-11-20 với 62 video có chỉ mục trả `searched_videos=62`, `skipped=[]`, 5 kết quả và ảnh đầu tiên HTTP 200. Truy vấn lặp sau khi mô hình và chỉ mục đã nạp mất khoảng 0,25 giây trên máy thử; lần đầu phụ thuộc vào thời gian nạp mô hình và kiểm tra checksum. Điểm cosine không biểu thị xác suất và xếp hạng toàn kho chưa có benchmark nhãn xuyên video độc lập.
