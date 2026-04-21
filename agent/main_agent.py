import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import List, Dict, Any, Optional

device = "cuda" if torch.cuda.is_available() else "cpu"

class MainAgent:
    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        chroma_collection: str = "lab14",
        top_k: int = 3,
        optimized: bool = False,
    ):
        self.top_k = top_k
        self.optimized = optimized
        
        # Load embedding model
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)
        
        # Load local LLM
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.llm_pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float16,
        )
        
        # Init ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=chroma_collection)
        print(f"--- Agent Initialized (Device: {device}, Optimized: {optimized}) ---")

    def index_documents(self, chunks_file: str):
        if not os.path.exists(chunks_file): return
        
        import json
        ids, documents, embeddings = [], [], []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                ids.append(data['id'])
                documents.append(data['text'])
                embeddings.append(self.embed_model.encode(data['text']).tolist())
        
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings)
        print(f"--- Đã lập chỉ mục {len(ids)} đoạn vào ChromaDB ---")

    async def query(self, question: str) -> Dict[str, Any]:
        # 1. Retrieval
        search_queries = [question]
        if self.optimized:
            # V2: Mở rộng truy vấn để tăng Hit Rate
            search_queries.append(f"Thông tin về {question} trong lịch sử Việt Nam")
        
        all_docs, all_ids = [], []
        for q in search_queries:
            results = self.collection.query(
                query_embeddings=[self.embed_model.encode(q).tolist()],
                n_results=self.top_k
            )
            if results["documents"]:
                all_docs.extend(results["documents"][0])
                all_ids.extend(results["ids"][0])

        # De-duplicate và lấy top_k
        unique_docs = list(dict.fromkeys(all_docs))[:self.top_k]
        unique_ids = list(dict.fromkeys(all_ids))[:self.top_k]
        context = "\n\n---\n\n".join(unique_docs)

        # 2. Prompt Engineering
        if self.optimized:
            # V2: Prompt chuyên sâu + Ví dụ
            system_prompt = """Bạn là chuyên gia sử học Việt Nam. Chỉ dùng tài liệu <context> để trả lời. 
Nếu không thấy thông tin, hãy đáp: 'Tôi không tìm thấy thông tin trong tài liệu'.
Ví dụ: 
Q: 'Nguyễn Huệ lên ngôi năm nào?' 
A: 'Dựa trên tài liệu, Nguyễn Huệ lên ngôi năm 1788.'"""
        else:
            # V1: Prompt cơ bản
            system_prompt = "Bạn là trợ lý ảo trả lời câu hỏi dựa trên tài liệu context được cung cấp."

        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
<context>
{context}
</context>
Câu hỏi: {question}<|im_end|>
<|im_start|>assistant
"""

        # 3. Generation
        outputs = self.llm_pipeline(
            prompt,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.1 if self.optimized else 0.7,
            top_p=0.9,
            pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
        )
        
        answer = outputs[0]["generated_text"].split("<|im_start|>assistant")[-1].strip()
        
        return {
            "answer": answer,
            "retrieved_ids": unique_ids,
            "contexts": unique_docs
        }
