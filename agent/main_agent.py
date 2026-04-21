import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
import os
import re

class MainAgent:
    """
    Hệ thống RAG chuyên nghiệp sử dụng Qwen2.5 và FAISS cho dữ liệu Lịch sử (data.md)
    """
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                 embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Đang khởi tạo Agent (Device: {self.device}) ---")
        
        # 1. Load Tokenizer & Model LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ép sử dụng float16 để chạy nhanh hơn trên GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        print(f"--- Model đã tải lên: {self.model.device} ---")
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # 2. Load Embedding Model
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        
        # 3. Vector DB
        self.index = None
        self.documents = []

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Chia văn bản thành các đoạn nhỏ để máy dễ tìm kiếm"""
        # Chia theo đoạn văn trước
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Nếu có đoạn nào vẫn quá dài, chia cắt theo độ dài (brute force)
        final_chunks = []
        for c in chunks:
            if len(c) > chunk_size:
                start = 0
                while start < len(c):
                    final_chunks.append(c[start:start + chunk_size])
                    start += chunk_size - overlap
            else:
                final_chunks.append(c)
        
        return final_chunks

    def load_data(self, file_path: str):
        """Đọc và lập chỉ mục file dữ liệu"""
        if not os.path.exists(file_path):
            print(f"Lỗi: Không tìm thấy file {file_path}")
            return
        
        print(f"Đang đọc dữ liệu từ {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Tiền xử lý: Xóa bớt khoảng trắng dư thừa
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Cắt nhỏ văn bản
        self.documents = self.chunk_text(content)
        print(f"Đã chia thành {len(self.documents)} đoạn (chunks).")
        
        # Lập chỉ mục FAISS
        embeddings = self.embed_model.encode(self.documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("--- Đã hoàn tất lập chỉ mục Vector DB ---")

    async def query(self, question: str, top_k: int = 4) -> Dict:
        """Quy trình RAG: Retrieval -> Prompting -> Generation"""
        # 1. Retrieval
        context = "Không có thông tin liên quan."
        retrieved_docs = []
        if self.index is not None:
            question_emb = self.embed_model.encode([question])
            distances, indices = self.index.search(np.array(question_emb).astype('float32'), top_k)
            retrieved_docs = [self.documents[i] for i in indices[0] if i != -1]
            context = "\n\n---\n\n".join(retrieved_docs)

        # 2. Prompt Engineering (Sử dụng System Prompt và XML)
        system_prompt = """Bạn là một chuyên gia về Lịch sử Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên tài liệu được cung cấp trong thẻ <context>.
QUY TẮC:
1. CHỈ sử dụng thông tin trong thẻ <context>.
2. Nếu thông tin không có trong tài liệu, hãy đáp 'Tôi không biết'. Tuyệt đối không tự bịa ra thông tin.
3. Không sử dụng kiến thức bên ngoài."""

        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
<context>
{context}
</context>

<question>
{question}
</question><|im_end|>
<|im_start|>assistant
"""

        # 3. Generation (Tối ưu để tránh lặp và ảo tưởng)
        outputs = self.llm_pipeline(
            prompt, 
            max_new_tokens=350, 
            do_sample=False, 
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        full_text = outputs[0]['generated_text']
        answer = full_text.split("<|im_start|>assistant")[-1].strip()

        return {
            "answer": answer,
            "contexts": retrieved_docs,
            "metadata": {
                "llm_model": "Qwen2.5-0.5B",
                "chunks_retrieved": len(retrieved_docs)
            }
        }

# if __name__ == "__main__":
#     import asyncio
    
#     async def test():
#         agent = MainAgent()
        
#         # Tự động nạp file data.md
#         data_path = "data.md"
#         if os.path.exists(data_path):
#             agent.load_data(data_path)
#         else:
#             print("Vui lòng đảm bảo file data.md nằm ở thư mục gốc.")
        
#         # Danh sách câu hỏi kiểm thử thực tế từ dữ liệu
#         test_questions = [
#             "Hiệp định Paris được ký kết vào ngày tháng năm nào?",
#             "Chiến dịch Tây Nguyên diễn ra từ ngày nào đến ngày nào?",
#             "Nội dung chính của Hội nghị Trung ương Đảng lần thứ 21 là gì?",
#             "Trận 'Điện Biên Phủ trên không' có ý nghĩa gì?"
#         ]
        
#         print("\n" + "="*50)
#         print("BẮT ĐẦU CHẠY THỬ NGHIỆM RAG")
#         print("="*50)
        
#         for q in test_questions:
#             print(f"\n❓ Câu hỏi: {q}")
#             resp = await agent.query(q)
#             print(f"💡 Trả lời: {resp['answer']}")
#             print("\n📄 [DỮ LIỆU THẬT] Context mà Agent đã tìm được:")
#             for i, c in enumerate(resp['contexts']):
#                 print(f"   - Đoạn {i+1}: {c[:150]}...") # In 150 ký tự đầu của mỗi đoạn
#             print("-" * 50)

#     asyncio.run(test())
