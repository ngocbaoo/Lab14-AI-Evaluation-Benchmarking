import asyncio
import os
from typing import List, Dict

class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    """
    def __init__(self, *, chroma_db_path: str = "./chroma_db", chroma_collection: str = "lab14", top_k: int = 3):
        self.name = "SupportAgent-v1"
        self.top_k = top_k
        self._chroma_db_path = chroma_db_path
        self._chroma_collection = chroma_collection
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=self._chroma_db_path)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self._collection = client.get_or_create_collection(name=self._chroma_collection, embedding_function=embedding_fn)
        return self._collection

    async def query(self, question: str) -> Dict:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Tìm kiếm context liên quan.
        2. Generation: Gọi LLM để sinh câu trả lời.
        """
        collection = self._get_collection()
        result = collection.query(
            query_texts=[question],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_ids: List[str] = (result.get("ids") or [[]])[0]
        retrieved_docs: List[str] = (result.get("documents") or [[]])[0]
        retrieved_metas: List[Dict] = (result.get("metadatas") or [[]])[0]

        # (Tạm) Generation: giữ logic đơn giản để tập trung đánh giá Retrieval.
        return {
            "answer": f"Dựa trên tài liệu hệ thống, tôi xin trả lời câu hỏi '{question}' như sau: [Câu trả lời mẫu].",
            "contexts": retrieved_docs,
            "metadata": {
                "model": "gpt-4o-mini",
                "tokens_used": 150,
                "sources": [m.get("source") for m in retrieved_metas if isinstance(m, dict) and m.get("source")],
                "retrieved_ids": retrieved_ids,
                "retrieval": {
                    "top_k": self.top_k,
                    "distances": (result.get("distances") or [[]])[0],
                },
            }
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)
    asyncio.run(test())
