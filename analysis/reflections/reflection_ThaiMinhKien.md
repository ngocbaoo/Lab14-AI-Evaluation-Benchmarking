# Báo cáo Cá nhân (Personal Reflection) - Lab Day 14
**Họ tên:** Thái Minh Kiên  
**Vai trò:** Lead Developer - Part 2: Build Evaluation Pipeline

---

## 1. Đóng góp Kỹ thuật (Engineering Contribution)

Trong dự án này, tôi chịu trách nhiệm chính trong việc xây dựng hệ thống lõi để đánh giá và đo lường hiệu năng của RAG Agent. Các đóng góp cụ thể bao gồm:

### Xây dựng Multi-Judge Consensus Engine (`engine/llm_judge.py`)
- **Tích hợp đa mô hình:** Triển khai module `MultiModelJudge` cho phép sử dụng song song cả **GPT-4o** và **Gemini 2.5 Flash** để đánh giá câu trả lời. Điều này giúp giảm thiểu bias của một mô hình đơn lẻ.
- **Logic đồng thuận & Xử lý xung đột:** Thiết kế thuật toán tính `final_score` dựa trên trung bình cộng và `agreement_rate`. Đặc biệt, tôi đã code logic tự động gắn cờ `needs_human_review` khi:
    - Sự chênh lệch điểm số giữa 2 judge > 1.0.
    - Kết quả đánh giá rơi vào vùng không chắc chắn (điểm gần ngưỡng 3.0).
    - Có lỗi từ một trong các provider.

### Tối ưu hiệu năng bằng Asynchronous Programming (`engine/runner.py`)
- **Parallel Execution:** Tận dụng `asyncio.gather` để gọi đồng thời các Judge, giúp giảm thời gian đánh giá xuống mức tối thiểu (hiệu năng thực tế đạt < 1 phút cho 50 test cases khi chạy batch size 5-10).
- **Batch Processing:** Xây dựng `BenchmarkRunner` hỗ trợ xử lý dữ liệu theo lô (batching) để tránh overload API rate limits nhưng vẫn duy trì tốc độ cao.

### Hệ thống Regression Testing & Release Gate (`main.py`)
- **Tự động hóa so sánh:** Viết module `_build_regression` để so sánh trực tiếp phiên bản V1 (Base) và V2 (Optimized).
- **Cổng kiểm soát chất lượng (Release Gate):** Thiết lập các ngưỡng (thresholds) để tự động đưa ra quyết định `APPROVE` hoặc `BLOCK_RELEASE` dựa trên sự sụt giảm của Score, Hit Rate hoặc sự gia tăng đột biến của tỉ lệ cần con người can thiệp (HITL rate).

---

## 2. Chiều sâu Kỹ thuật (Technical Depth)

Trong quá trình thực hiện, tôi đã nghiên cứu và áp dụng các khái niệm nâng cao:

- **Metrics Understanding:** 
    - **MRR (Mean Reciprocal Rank):** Mặc dù không trực tiếp viết module retrieval, tôi đã tích hợp MRR vào pipeline đánh giá để hiểu được vị trí của tài liệu đúng trong kết quả trả về ảnh hưởng thế nào đến chất lượng câu trả lời của LLM.
    - **Agreement Rate:** Áp dụng để đo lường độ tin cậy của hệ thống Judge tự động.
- **Trade-offs:** 
    - Tôi đã lựa chọn sử dụng **Gemini 2.5 Flash** kết hợp với **GPT-4o** để cân bằng giữa chi phí (cost) và độ chính xác (accuracy). Việc sử dụng toàn bộ GPT-4o cho 2 judge sẽ đội chi phí lên gấp 5-10 lần mà không cải thiện đáng kể độ đồng thuận.
- **Position Bias & Reasoning:** Để hạn chế việc LLM judge ưu tiên các câu trả lời dài hoặc có cấu trúc nhất định, tôi đã thiết kế prompt yêu cầu Judge phải đưa ra `reasoning` chi tiết trước khi chốt `score`.

---

## 3. Giải quyết vấn đề (Problem Solving)

- **Vấn đề:** Khi chạy song song nhiều request, thường xuyên gặp lỗi `rate_limit` hoặc API timeout làm gián đoạn pipeline.
- **Giải pháp:** Tôi đã triển khai cơ chế xử lý lỗi cục bộ trong `run_single_test`. Nếu một test case lỗi, hệ thống sẽ log lại lỗi đó và tiếp tục chạy các case khác thay vì dừng toàn bộ. Kết quả lỗi sẽ được đánh dấu `agent_error` hoặc `judge_error` để hậu kiểm.
- **Vấn đề:** Điểm số giữa các Judge thường xuyên lệch nhau do thang điểm 1-5 có tính cảm quan.
- **Giải pháp:** Tôi đã chuẩn hóa Rubric đánh giá ngay trong prompt (phân định rõ thế nào là 1, 2, 3, 4, 5 điểm) để ép các mô hình khác nhau về cùng một hệ quy chiếu tư duy.

---

## 4. Tự đánh giá

Dựa trên các tiêu chí trong `GRADING_RUBRIC.md`, tôi nghĩ rằng mình đã đạt được cơ bản các yêu cầu chuyên môn về Engineering, Technical Depth và Problem Solving trong phạm vi Part 2 của dự án.
