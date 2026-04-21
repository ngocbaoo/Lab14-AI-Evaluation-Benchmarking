# BÁO CÁO CÁ NHÂN - LAB DAY 14 (AI EVALUATION)

**Họ và tên:** Lê Minh Hoàng
**MSSV:** 2A202600101

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

Trong dự án này, tôi chịu trách nhiệm chính trong **Giai đoạn 1 (45'): Thiết kế Golden Dataset & Script SDG** và thiết lập cơ chế đo lường hiệu suất truy xuất của Vector Database.

### A. Thiết kế Golden Dataset & Script SDG
*   **Synthetic Data Generation (SDG):** Xây dựng kịch bản (Script) tự động tạo dữ liệu đánh giá tổng hợp. 
*   **Golden Dataset:** Trực tiếp tạo và kiểm định ra **60 test cases chất lượng cao** (được phân loại theo các mức độ khó: Easy, Normal, Hard). Bộ dữ liệu này đóng vai trò làm chuẩn đo lường (Golden Standard) đáng tin cậy để đánh giá tính chính xác của LLM Judge và hệ thống RAG Agent.

### B. Đo lường hiệu suất Vector DB
*   **Tính toán Metrics truy xuất:** Thiết lập và lập trình cơ chế đo lường các chỉ số cốt lõi cho RAG Pipeline.
    *   **Hit Rate:** Tính toán tỷ lệ hệ thống truy xuất thành công tài liệu chứa câu trả lời đúng (Expected Chunk) trong Top K.
    *   **MRR (Mean Reciprocal Rank):** Tính toán thứ hạng của tài liệu đúng đầu tiên được trả về, giúp đánh giá chuyên sâu mức độ tối ưu của thuật toán tìm kiếm Vector.

---

## 2. Chiều sâu kỹ thuật (Technical Depth)

*   **Prompt Engineering trong SDG:** Áp dụng kỹ thuật Prompt tiên tiến để thiết kế các chỉ thị cho LLM (Generator), ép LLM phải sinh ra các bộ câu hỏi và câu trả lời đa dạng về ngữ nghĩa, không bị nhàm chán hoặc trùng lặp, bám sát các luồng dữ liệu (Context) khác nhau trong file.
*   **Data Generation Pipeline:** Viết mã tối ưu kết xuất dữ liệu sang định dạng chuẩn mực (JSONL) tự động, giúp đồng bộ hóa liền mạch với các giai đoạn đánh giá mô hình phía sau, giảm thiểu tỷ lệ lỗi (error rate) khi parse dữ liệu.
*   **Toán học trong Retrieval:** Nắm vững và cài đặt thủ công các công thức đo lường hiệu suất tìm kiếm (Hit Rate, MRR) theo đúng chuẩn bài toán RAG, giúp dự án lượng hóa được chất lượng của Vector DB và Embedding Model.

---

## 3. Giải quyết vấn đề (Problem Solving)

*   **Thách thức:** Sinh ra 60 test cases mà không bị thiên lệch (bias), đảm bảo phân hóa và bao phủ đủ các Use Cases phức tạp (Câu hỏi suy luận thay vì chỉ tìm keywords đơn giản).
    *   **Giải pháp:** Bổ sung các Role-playing và ràng buộc tạo độ khó đa dạng vào prompt sinh dữ liệu, sau đó thực hiện rà soát chéo (QC) để loại bỏ các câu hỏi quá mơ hồ, cải thiện chất lượng list Golden Set.
*   **Thách thức:** Tính MRR và Hit Rate cần xử lý việc đối chiếu ngặt nghèo giữa cụm `retrieved_ids` do agent trả về và `expected_ids` trong Golden Dataset.
    *   **Giải pháp:** Viết các hàm đánh giá luân phiên tối ưu, kết hợp xử lý các mảng dữ liệu (arrays) để tính thứ hạng một cách chính xác dựa vào Index. Cài đặt các cơ chế dự phòng (try-except) tránh crash toàn bộ luồng evaluation nếu một vài sample có format bất thường.

---
**Kết quả thực tế:** 
Nhờ có đủ 60 test cases (Golden Set) làm thước đo tiêu chuẩn cùng với các logic đo lường tính toán minh bạch, hệ thống đã ghi nhận Agent V2 thực tế đạt **Hit Rate: 20%** và chỉ số **MRR: 15.28%**. Kết hợp cùng các cải tiến của nhóm, bản đánh giá đã xác nhận phiên bản tối ưu Agent V2 đạt **điểm trung bình (Average Score) 2.56/5**, **Pass Rate 36.67%**, và thỏa mãn mọi tiêu chí phát hành (dán nhãn **APPROVE** bởi Judge).
