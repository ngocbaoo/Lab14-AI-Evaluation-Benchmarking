# BÁO CÁO CÁ NHÂN - LAB DAY 14 (AI EVALUATION)

**Họ và tên:** Tạ Bảo Ngọc
**MSSV:** 2A202600286

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

Trong dự án này, tôi tập trung vào hai trụ cột chính: **Xây dựng bộ dữ liệu cơ sở** và **Tối ưu hóa khả năng truy xuất của Agent**.

### A. Chiến lược Dữ liệu (Data Engineering)
*   **Curating & Cleaning:** Trực tiếp tìm kiếm và tinh chỉnh bộ dữ liệu kiến thức (định dạng Markdown). Xử lý triệt để lỗi Encoding (UTF-8) và chuẩn hóa cấu trúc dữ liệu thô để nạp vào Vector Database.
*   **ID Mapping Logic:** Thiết kế quy trình ánh xạ (Mapping) chính xác giữa các đoạn văn bản (Chunks) và bộ đề kiểm tra (Golden Set). Đây là bước sống còn để hệ thống có thể tính toán được chỉ số **Hit Rate** và **MRR** một cách tự động.

### B. Tối ưu hóa Agent (Advanced RAG)
*   **Multi-Query Retrieval:** Triển khai kỹ thuật mở rộng truy vấn. Khi người dùng đặt câu hỏi, Agent sẽ tự động sinh ra các câu hỏi tương đương để tăng vùng phủ (Coverage) tìm kiếm trong ChromaDB, giúp tăng Hit Rate lên đáng kể.
*   **Prompt Engineering cho SLM:** Tinh chỉnh System Prompt cho model nhỏ (Qwen-0.5B) bằng kỹ thuật **Few-shot Prompting**. Việc đưa vào các ví dụ mẫu giúp model vượt qua giới hạn về kích thước trung bình và trả lời chuyên sâu, đúng trọng tâm hơn.
*   **ChromaDB Persistence:** Cấu hình hệ thống lưu trữ vector có tính bền vững (Persistence), giúp Agent không phải lập chỉ mục lại nhiều lần, tối ưu hóa tốc độ khởi động hệ thống.

---

## 2. Chiều sâu kỹ thuật (Technical Depth)

Tôi đã áp dụng các kiến thức chuyên sâu về RAG trinh quy:
*   **Phân tích ma trận Retrieval:** Hiểu rõ mối liên hệ giữa Chunk size và tính chính xác của thông tin (Precision vs Recall).
*   **Xử lý Hallucination:** Áp dụng các ràng buộc (Constraints) nghiêm ngặt trong Prompt để ép Agent chỉ trả lời dựa trên context, giảm thiểu tối đa hiện tượng "nói sảng" của AI.
*   **Evaluation Driven Development:** Sử dụng kết quả đánh giá từ Judge để quay ngược lại tinh chỉnh mã nguồn Agent (Iteration), thay vì chỉ suy đoán cảm tính.

---

## 3. Giải quyết vấn đề (Problem Solving)

*   **Thách thức:** Hệ thống báo Hit Rate = 0% dù Agent có câu trả lời đúng.
    *   **Giải pháp:** Tôi đã phát hiện ra sự sai lệch giữa Metadata ID trong database và ID trong file test. Tôi đã viết script `final_fix.py` để đồng bộ lại toàn bộ, đưa chỉ số Hit Rate về giá trị thực tế (20%).
*   **Thách thức:** Model Qwen-0.5B thường bị lẫn tiếng Anh hoặc tiếng Trung.
    *   **Giải pháp:** Áp dụng XML Tagging và Strict Language Instruction trong Prompt, giúp model đạt độ ổn định ngôn ngữ cao hơn trong bản V2.

---
**Kết quả thực tế:** Bản V2 do tôi tối ưu đã đạt điểm **2.56**, tăng **+0.33** so với bản Baseline, và được hệ thống Judge chấp nhận (APPROVE) cho phát hành.
