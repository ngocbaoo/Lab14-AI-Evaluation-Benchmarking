# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60
- **Tỉ lệ Pass/Fail:** 22 Pass / 38 Fail (Pass Rate: 0.3667)
- **Điểm LLM-Judge trung bình (V2):** 2.56 / 5.0
- **Cải thiện (Delta):** +0.33 so với bản V1.
- **Hit Rate trung bình:** 20.0%

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Retrieval Failure | 36 | Hit Rate bị báo cáo sai do lệch ID metadata |
| Hallucination | 8 | Model 0.5B tự ý đưa thêm nhận xét ngoài tài liệu |
| Language Mix | 5 | Trộn lẫn tiếng Việt - Anh dù đã có Prompt chặn |

## 3. Phân tích 5 Whys (Case tệ nhất: Hit Rate = 0)

### Case #1: Chỉ số Hit Rate luôn bằng 0.0%
1. **Symptom:** Hệ thống báo cáo không tìm thấy bất kỳ tài liệu đúng nào.
2. **Why 1:** Các ID trong `retrieved_ids` không khớp với `expected_retrieval_ids`.
3. **Why 2:** File bộ đề (`golden_set.jsonl`) bị lỗi Encoding và thiếu trường Mapping ID.
4. **Why 3:** Hệ thống sinh mã (SDG) chưa được đồng bộ hóa với hệ thống nạp dữ liệu (Ingestion).
5. **Why 4:** Ingestion pipeline lưu ID dưới dạng `data.md::chunk_XXX` trong khi Golden Set dùng định dạng khác.
6. **Root Cause:** Thiếu một Schema ID thống nhất giữa giai đoạn Thu thập (Data) và giai đoạn Đánh giá (Evaluation).

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Đã triển khai `final_fix.py` để đồng bộ ID và sửa lỗi Encoding UTF-8.
- [x] Đã cập nhật `MainAgent` V2 với Multi-Query để tăng xác suất tìm thấy chunk.
