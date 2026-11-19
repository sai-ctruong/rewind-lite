# Corpus v2 — phép đo baseline trên video thật

Đo ngày 2026-11-19 bằng model và đường tìm kiếm của web Rewind Lite. [JSON đầy đủ](benchmark-results/corpus-v2.json) lưu cấu hình, checksum đầu vào, chỉ mục từng video, dự đoán từng truy vấn và các chỉ số. Đây là **kết quả trên nhãn do Codex tự rà soát**, chưa phải ground truth được thẩm định độc lập.

## Giao thức

- Nguồn: [manifest 62 video](../data/annotations/corpus_v2_manifest.json), [84 truy vấn có cảnh khớp](../data/annotations/corpus_v2_labels.json), [20 truy vấn vắng mặt](../data/annotations/corpus_v2_absent.json). Chỉ 21 video có truy vấn được lập chỉ mục cho phép đo này. Mỗi truy vấn tìm **trong video nguồn của chính nó**; kết quả không đo tìm kiếm toàn kho 62 video.
- Split theo nhóm người: 64 truy vấn dương và 12 truy vấn âm ở `dev`; 20 dương và 8 âm ở `test`. Khóa dữ liệu bằng ba SHA-256 trong JSON. Ngưỡng từ chối được chọn bằng balanced accuracy trên `dev`, rồi áp dụng một lần trên `test`.
- Cùng cấu hình web: SigLIP2 `google/siglip2-base-patch16-256` tại revision `6328738b83b563ff8396fdb87f8ce130a7b60d74`; lấy mẫu 0,5 giây, tối đa 512 frame/video, tắt lọc trùng histogram và semantic, top 5. Tìm khoảng thời gian bằng `LiteService.search_temporal` với detector tắt.
- Keyframe đúng khi timestamp thuộc một trong các khoảng `[start, end)` đã gán nhãn. Chỉ số temporal dùng tIoU với khoảng đúng. Truy vấn vắng mặt bị từ chối khi cosine top 1 **nhỏ hơn** ngưỡng `dev`.
- Môi trường: Python 3.12.12, PyTorch 2.5.1+cu121, Transformers 4.57.3, OpenCV 4.11.0; NVIDIA GeForce RTX 3060 Laptop GPU. Bản chạy lưu trong JSON dùng 21 index đã cache, tổng 4.476 frame, mất 52,3 giây cho nạp cache và truy vấn; đây không phải thời gian lập chỉ mục lạnh.

## Kết quả

| Chỉ số | Dev | Test |
|---|---:|---:|
| Keyframe Hit@1 | 24/64 (37,5%) | 2/20 (10,0%) |
| Keyframe Hit@5 | 43/64 (67,2%) | 5/20 (25,0%) |
| Keyframe MRR | 0,488 | 0,154 |
| Temporal R@1, tIoU ≥ 0,5 | 10/64 (15,6%) | 2/20 (10,0%) |
| Temporal R@5, tIoU ≥ 0,5 | 14/64 (21,9%) | 5/20 (25,0%) |
| Temporal mIoU top 1 | 0,242 | 0,100 |
| Phát hiện đúng truy vấn có mặt | 26/64 (40,6%) | 5/20 (25,0%) |
| Từ chối đúng truy vấn vắng mặt | 12/12 (100%) | 6/8 (75,0%) |
| Balanced accuracy có mặt/vắng mặt | 70,3% | 50,0% |

Ngưỡng cosine chọn trên `dev` là **0,1229826659**. Tất cả 84 khoảng đúng đều có ít nhất một frame được lấy mẫu và giữ trong index; sai số định vị ở đây không do khoảng đúng hoàn toàn thiếu frame. Trên `test`, khoảng tin cậy Wilson mức truy vấn cho Hit@1 là 2,8–30,1%; các truy vấn cùng video/người có tương quan, nên khoảng này không phản ánh đầy đủ bất định theo nhóm.

Ba nhóm `test` có kết quả keyframe Hit@1 khác nhau: `person_a` 1/12, `person_c` 0/4, `person_e` 1/4. Ở nhóm `dev`, `person_d` đạt 12/16, cao hơn nhiều nhóm khác; gộp các nhóm che khuất chênh lệch này. Trong 8 truy vấn hành động ở `test`, không truy vấn nào đạt Hit@1. Cỡ mẫu nhóm và loại truy vấn còn nhỏ, nên chỉ dùng phân tích này để định hướng khảo sát lỗi.

## Diễn giải và giới hạn

Kết quả hiện tại **chưa đủ mạnh để khẳng định sản phẩm định vị chính xác**. Ví dụ, truy vấn `v2-vi-013` gán nhãn tại 117–122 giây của `video_11`, nhưng keyframe top 1 ở 270,6 giây; `v2-vi-061` gán nhãn tại 31–48 giây của `video_43`, top 1 ở 8,64 giây. Các hành động và vật thể lặp lại trong cùng video có thể tạo đáp án hợp lý ngoài khoảng đang gán nhãn; cần kiểm tra toàn bộ clip và đánh dấu **mọi** khoảng đúng trước khi diễn giải các trường hợp đó như lỗi mô hình. Không sửa nhãn `test` theo dự đoán của phép đo này.

Nhãn hiện do cùng một tác nhân soạn và rà soát từ khung hình lấy mẫu, chưa được một người thứ hai xem toàn bộ 21 video. Truy vấn vắng mặt đặc biệt cần xác nhận trên toàn clip. Bộ video cùng phòng, chủ yếu cảnh ăn uống và làm việc, nên không suy rộng kết quả sang bối cảnh khác. Những câu hỏi giống nhau có thể khớp clip khác trong 62 video; vì chưa gán nhãn toàn kho cho từng câu, không tính độ chính xác tìm kiếm toàn kho. Không so trực tiếp số corpus v2 với T06 cũ vì dữ liệu và giao thức khác.

## Chạy lại

Đặt 62 MP4 theo [manifest](../data/annotations/corpus_v2_manifest.json), kiểm tra checksum và nhãn theo [hướng dẫn dữ liệu](corpus-v2-evaluation.md), rồi chạy:

```powershell
.venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --output artifacts\corpus-v2-benchmark\report.json
```

Tệp JSON đã công bố là [bản đo 2026-11-19](benchmark-results/corpus-v2.json). Để báo cáo đồ án có ground truth vững hơn, cần người khác rà soát liên tục 21 clip, bổ sung mọi khoảng đúng và xác nhận các truy vấn vắng mặt; sau đó khóa nhãn mới và đo lại như một phiên bản dữ liệu khác. Các cải tiến model tiếp theo chỉ chọn trên `dev`.
