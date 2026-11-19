# Rà soát nhãn `dev` và thử nghiệm cải thiện corpus v2

Ngày rà soát: 2026-11-19. Người rà soát: Codex. Đây là lượt kiểm tra nội bộ bằng video nguồn và khung hình gốc; không thay thế việc xem liên tục toàn bộ clip hoặc người gán nhãn độc lập. Giữ nguyên ba tệp nhãn/manifest và [baseline đã công bố](corpus-v2-baseline.md) để kết quả cũ tái lập được. Các quyết định chọn mô hình chỉ dùng split `dev`.

## Bằng chứng kiểm tra

Codex đối chiếu khung hình giữa cửa sổ đúng và khung hình top 1 của **64 truy vấn dương `dev`** trong bản baseline. Ảnh rà soát nằm cục bộ tại `artifacts/dataset-v2/dev_review/`; chúng không được đưa vào Git vì trích từ video của chủ sở hữu. Một khung hình không thể xác định chính xác toàn bộ thời điểm bắt đầu/kết thúc hành động. Vì vậy các trường hợp dưới đây là dấu hiệu cần bổ sung khoảng đúng, chưa phải bản nhãn thay thế.

| ID | Video | Cửa sổ hiện tại | Khung hình ngoài cửa sổ cần kiểm tra tiếp | Quan sát |
|---|---|---:|---:|---|
| `v2-vi-007` | `video_6` | 10–22 s | ~51,4 s | Cốc trắng vẫn ở cạnh hộp ngũ cốc. |
| `v2-vi-008` | `video_6` | 0–20 s | ~56,2 s | Người áo đỏ vẫn ngồi một mình ở bàn ăn. |
| `v2-vi-041` | `video_30` | 14–18 s | ~45,1 s | Có thêm cảnh đưa cốc xanh lá lên uống. |
| `v2-vi-057` | `video_40` | 20–24 s | ~41,3 s | Có thêm cảnh uống từ cốc xanh lá. |
| `v2-vi-075` | `video_53` | 0–20 s | ~20,6 s | Chai nắp cam còn ở bên phải tờ giấy, sát ranh giới. |
| `v2-vi-081` | `video_62` | 4–8 s | ~9,6 s | Người áo be còn viết trên giấy cạnh máy tính. |

Một nhãn vắng mặt `dev` được xác nhận sai: `v2-abs-012` ghi rằng cảnh “Người phụ nữ áo khoác xám sử dụng điện thoại ở bàn ăn” không có trong `video_28`, nhưng khung hình gốc tại khoảng **16 s** cho thấy người phụ nữ cầm và nhìn vào điện thoại đang sáng. Ảnh bằng chứng cục bộ: `artifacts/dataset-v2/video_28_phone_16s.jpg`. Không dùng hàng này để chọn mô hình/ngưỡng trên `dev`. JSON thử nghiệm ghi `excluded_query_ids: ["v2-abs-012"]`; không loại hàng nào ở `test`. Các nhãn vắng mặt còn lại chưa được xác nhận trên toàn bộ thời lượng video.

Các quan sát trên giải thích một phần vì sao dự đoán có thể trông đúng mà bị chấm sai. Chúng **không** đủ để điều chỉnh khoảng đúng theo vị trí dự đoán của mô hình. Cần xem các đoạn lân cận trên video nguồn, ghi mọi lần xuất hiện phù hợp và phiên bản hóa bộ nhãn mới trước khi đưa số liệu vào báo cáo như ground truth đã kiểm chứng.

## Giao thức thử nghiệm

Chạy cùng pipeline của web và cùng 64 truy vấn dương, 11 truy vấn vắng mặt hợp lệ còn lại trên `dev`. Mỗi model có index riêng. So Hit@1, Hit@5, MRR, temporal R@5/tIoU và phát hiện có mặt/vắng mặt. Ngưỡng từ chối chọn **chỉ trên `dev`**. Sau khi chọn cấu hình, đo `test` một lần, giữ nguyên nhãn đã khóa và công bố cả cải thiện lẫn thoái lui. Không diễn giải các điểm số này như độ chính xác toàn kho 62 video.

```powershell
.venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --split dev --exclude-query-id v2-abs-012 --output artifacts\corpus-v2-experiments\e1-dev.json
```

## Kết quả thử nghiệm và thay đổi đã chọn

Trên cùng tập `dev`, E2 (`google/siglip-base-patch16-256-multilingual`, revision `8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6`) phát hiện có mặt/vắng mặt tốt hơn, nhưng xếp hạng khung hình và định vị khoảng thời gian kém hơn E1. Vì định vị thời điểm là mục tiêu chính, giữ E1 làm model của web. Một phép thử kết hợp hai danh sách top 5 bằng RRF chỉ tăng Hit@1 thêm một truy vấn `dev` ở trọng số E1 1,5, không tăng Hit@5; không đưa thêm model vào sản phẩm.

| Cấu hình `dev` | Keyframe Hit@1 | Keyframe Hit@5 | Temporal R@5, tIoU ≥ 0,5 | Balanced accuracy có mặt/vắng mặt |
|---|---:|---:|---:|---:|
| E1 cũ, loại nhãn âm sai | 24/64 | 43/64 | 14/64 | 73,6% |
| E2, loại nhãn âm sai | 15/64 | 35/64 | 12/64 | 79,0% |
| E1, khung hình cách nhau ≥5 s, loại nhãn âm sai | 24/64 | **51/64** | 14/64 | 73,6% |

Phép chọn khung hình mới giữ top 1 theo cosine; chọn các hit tiếp theo từ cao xuống thấp sao cho cách các hit đã chọn ít nhất 5 giây trong cùng video. Nếu clip quá ngắn, lấp đủ top 5 bằng các hit còn lại theo điểm gốc. Vì thế giao diện có nhiều thời điểm khác nhau để mở xem. Đã thử làm mượt điểm theo lân cận 1–8 giây trên `dev`: Hit@5 đều giảm, nên không áp dụng. Khoảng cách 5 giây được chọn trên `dev`, rồi **đo `test` một lần**. Không dùng kết quả `test` để chỉnh tiếp khoảng cách.

| Keyframe trên `test` | E1 cũ | E1, cách nhau ≥5 s |
|---|---:|---:|
| Hit@1 | 2/20 (10,0%) | 2/20 (10,0%) |
| Hit@5 | 5/20 (25,0%) | **8/20 (40,0%)** |
| MRR | 0,154 | 0,187 |

Ba truy vấn `test` có thêm hit trong top 5: `v2-vi-004`, `v2-vi-013`, `v2-vi-063`; không có truy vấn nào mất hit. Các phép đo temporal và từ chối truy vấn vắng mặt không đổi: temporal R@5/tIoU ≥0,5 ở `test` vẫn 5/20, balanced accuracy có mặt/vắng mặt vẫn 50,0%. Thay đổi chỉ áp dụng cho **chế độ Khung hình** của web. Giao diện hiện mở mặc định ở chế độ tìm khoảng thời gian; không nên dùng số Hit@5 khung hình để mô tả chất lượng chế độ đó.

[JSON E2 trên `dev`](benchmark-results/corpus-v2-e2-dev.json) và [JSON E1 đã cải thiện trên cả hai split](benchmark-results/corpus-v2-diverse.json) lưu dự đoán từng câu, checksum nguồn và cấu hình. Bản E1 trên cả hai split vẫn chứa `v2-abs-012` ở phép hiệu chỉnh ngưỡng để giữ nguyên nhãn đầu vào đã khóa; so sánh chọn model ở bảng `dev` trên dùng bản thử nghiệm loại hàng sai này. Cả hai bảng vẫn chỉ là số đo trên nhãn nội bộ có giới hạn đã nêu, không phải độ chính xác được thẩm định độc lập.

```powershell
.venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --split dev --exclude-query-id v2-abs-012 --model-id google/siglip-base-patch16-256-multilingual --revision 8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6 --index-root artifacts\corpus-v2-experiments\e2-indexes --output artifacts\corpus-v2-experiments\e2-dev.json
.venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --split dev --exclude-query-id v2-abs-012 --keyframe-min-gap-s 5 --output artifacts\corpus-v2-experiments\e1-diverse-dev.json
.venv\Scripts\python.exe -m evaluation.corpus_v2_benchmark --device cuda --keyframe-min-gap-s 5 --output artifacts\corpus-v2-experiments\e1-diverse-full.json
```
