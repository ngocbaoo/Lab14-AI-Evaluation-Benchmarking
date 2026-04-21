Họ và tên: Nguyễn Xuân Hải  MSSV: (chưa cung cấp)

1. Đóng góp kỹ thuật (Engineering Contribution)
Trong dự án này, tôi phụ trách chính phần Retrieval Evaluation: mô tả pipeline ingestion Markdown vào Vector Database (Chroma) và xây dựng cơ chế đo lường Hit Rate/MRR để chứng minh retrieval stage hiệu quả trước khi đánh giá generation.

A. Pipeline ingestion Markdown (.md) → Vector DB (Chroma)
Chunking dữ liệu:  đọc file Markdown (mặc định `data.md`), chuẩn hoá text, tách theo đoạn và ghép thành các chunk theo `max_chars` (mặc định 1200). Mỗi chunk được gán ID quyết định theo format `{filename}::chunk_{00001}` (ví dụ `data.md::chunk_00002`) và ghi ra `chunks.jsonl`.
Embedding + Indexing/Storage: `ingest_chroma.py` đọc `chunks.jsonl`, tạo `chromadb.PersistentClient(path=./chroma_db)`, tạo collection `lab14` và upsert các chunk theo batch. Embedding function dùng `SentenceTransformerEmbeddingFunction` với model lấy từ biến môi trường `EMBEDDING_MODEL` (mặc định `all-MiniLM-L6-v2`). Kết quả được lưu persistent ở thư mục `chroma_db/`.

B. Triển khai Retrieval trong Agent và xuất `retrieved_ids`
Agent retrieval: `main_agent.py` dùng chính Chroma collection ở `./chroma_db`, thực hiện `collection.query(query_texts=[question], n_results=top_k, include=["documents","metadatas","distances"])`.
Output phục vụ evaluation: agent trả về `metadata.retrieved_ids` (list các chunk IDs) và `metadata.retrieval.distances` để trace chất lượng truy xuất.

C. Đo lường hiệu suất Vector DB: Hit Rate và MRR
Ground truth IDs: trong `golden_set.jsonl`, mỗi test case đã có `expected_retrieval_ids` (List[str]) để đối chiếu với `retrieved_ids`. Việc gắn expected IDs có script hỗ trợ match context vào `chunks.jsonl`.
Tính toán metrics: `retrieval_eval.py` cung cấp các hàm:
Hit Rate@k: = 1 nếu tồn tại ít nhất 1 `expected_id` nằm trong top-k của `retrieved_ids`, ngược lại 0.
MRR: tìm vị trí (1-index) của expected doc đầu tiên trong `retrieved_ids`, MRR = 1/rank; nếu không có thì 0.
Tích hợp vào benchmark: `runner.py` chạy agent cho từng case; `main.py` gọi `retrieval_eval.py` thông qua `ExpertEvaluator` để ghi vào `ragas["retrieval"]`, rồi tổng hợp và xuất `summary.json`, `benchmark_results.json`.

2. Chiều sâu kỹ thuật (Technical Depth)
Toán học Retrieval: hiểu và cài đặt đúng chuẩn bài toán RAG cho Hit Rate@k và MRR (xử lý top-k, rank 1-index, và trường hợp không có relevant doc).
Data flow & contract dữ liệu: đảm bảo “đường đi” của ID từ ingestion (chunk IDs deterministic trong `chunk_doc.py`) → storage (Chroma) → retrieval (agent trả `retrieved_ids`) → evaluation (so khớp với `expected_retrieval_ids` trong golden set).
Phân tích kết quả theo dataset: nhận diện các case “out-of-context” trong golden_set.jsonl` (expected_retrieval_ids rỗng) làm kéo giảm metric tổng, từ đó đề xuất đánh giá tách riêng in-context vs out-of-context để tránh hiểu sai chất lượng retriever.

3. Giải quyết vấn đề (Problem Solving)
Thách thức: Đảm bảo retrieval evaluation phản ánh đúng chất lượng retriever trước khi đánh giá generation, trong khi agent hiện đang trả lời dạng placeholder (generation không dùng context thật).
Giải pháp: Tập trung chứng minh retrieval qua Hit Rate/MRR bằng cách dựa trực tiếp trên `retrieved_ids` từ Chroma và `expected_retrieval_ids` trong golden set; coi đây là tiêu chí “đầu vào bắt buộc” trước khi đánh giá chất lượng câu trả lời.

Thách thức: Dataset có các case out-of-context (expected_retrieval_ids = []) làm metric tổng bị kéo thấp và khó diễn giải.
Giải pháp: Phân tích metric theo 2 nhóm: (1) tổng toàn bộ dataset và (2) chỉ các case có expected_retrieval_ids (in-context), để đánh giá retriever công bằng hơn.

Kết quả thực tế: Theo `summary.json`, benchmark gồm 60 cases ghi nhận Hit Rate = 0.10 và MRR = 0.0622. Khi xét riêng 46 case “in-context” (expected_retrieval_ids không rỗng), Hit Rate ≈ 0.1304 và MRR ≈ 0.0812. Điều này cho thấy retrieval stage có hoạt động nhưng hiệu quả còn thấp, đặc biệt nhóm `hard` trong dataset hiện tại có Hit Rate và MRR bằng 0.
