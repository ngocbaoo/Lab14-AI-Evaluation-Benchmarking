import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
import os
import json

class MainAgent:
    """
    Hệ thống RAG chuyên nghiệp sử dụng Qwen2.5 và ChromaDB cho dữ liệu Lịch sử.
    Đã tối ưu hóa cho GPU và tích hợp tốt với hệ thống Benchmark.
    """
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", 
                 embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 chroma_db_path: str = "./chroma_db",
                 chroma_collection: str = "lab14",
                 top_k: int = 4):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.top_k = top_k
        print(f"--- Đang khởi tạo Agent (Device: {self.device}) ---")
        
        # 1. Load Tokenizer & Model LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        
        # 3. Setup ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=chroma_collection)

    def index_documents(self, chunks_file: str):
        """Nạp dữ liệu từ file chunks.jsonl vào ChromaDB"""
        if not os.path.exists(chunks_file):
            print(f"❌ Lỗi: Không tìm thấy file {chunks_file}")
            return

        print(f"Đang lập chỉ mục dữ liệu từ {chunks_file}...")
        
        ids = []
        documents = []
        metadatas = []
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                ids.append(data["id"])
                documents.append(data["text"])
                metadatas.append(data.get("metadata", {}))

        # Thêm vào ChromaDB theo từng batch để tránh lỗi bộ nhớ
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            # Tính toán embeddings
            embeddings = self.embed_model.encode(documents[i:end]).tolist()
            
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings,
                metadatas=metadatas[i:end]
            )
        
        print(f"--- Hoàn tất lập chỉ mục {len(ids)} đoạn vào ChromaDB ---")

    async def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """Quy trình RAG: Retrieval -> Prompting -> Generation"""
        k = top_k or self.top_k
        
        # 1. Retrieval
        question_emb = self.embed_model.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=question_emb,
            n_results=k
        )
        
        retrieved_docs = results['documents'][0]
        retrieved_ids = results['ids'][0]
        context = "\n\n---\n\n".join(retrieved_docs)

        # 2. Prompt Engineering (ChatML + XML)
        system_prompt = """Bạn là một nhà sử học chuyên về Lịch sử Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên tài liệu được cung cấp trong thẻ <context>.
        QUY TẮC:
        1. CHỈ sử dụng thông tin trong thẻ <context>.
        2. Nếu thông tin không có trong tài liệu, hãy đáp 'Tôi không biết'. Tuyệt đối không tự bịa ra thông tin.
        3. Trả lời chính xác, khách quan và ngắn gọn."""

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

        # 3. Generation
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
            "retrieved_ids": retrieved_ids, # Trả về IDs để evaluation tính Hit Rate
            "metadata": {
                "llm_model": "Qwen2.5-0.5B",
                "db_type": "ChromaDB"
            }
        }

if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Khởi tạo với cấu hình mặc định (tương thích với main.py)
        agent = MainAgent()
        
        # Lập chỉ mục từ file chunks.jsonl nếu có
        chunks_path = "data/chunks.jsonl"
        if os.path.exists(chunks_path):
            agent.index_documents(chunks_path)
        
        q = "Hiệp định Paris được ký kết vào ngày tháng năm nào?"
        print(f"\n❓ Câu hỏi test: {q}")
        resp = await agent.query(q)
        print(f"💡 Trả lời: {resp['answer']}")
        print(f"📌 IDs đã tìm thấy: {resp['retrieved_ids']}")

    asyncio.run(test())
