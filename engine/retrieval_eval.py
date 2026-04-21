from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int = 3) -> float:
        """
        Hit Rate@k:
        = 1 nếu ít nhất 1 expected_id nằm trong top_k của retrieved_ids, ngược lại 0.
        """
        if top_k <= 0:
            return 0.0
        expected_set = set(expected_ids or [])
        if not expected_set:
            return 0.0
        top_retrieved = list(retrieved_ids or [])[:top_k]
        hit = any(doc_id in expected_set for doc_id in top_retrieved)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
        """
        Mean Reciprocal Rank (MRR) cho 1 query:
        Tìm vị trí đầu tiên (1-indexed) của một expected_id trong retrieved_ids.
        MRR = 1 / position. Nếu không thấy thì 0.
        """
        expected_set = set(expected_ids or [])
        if not expected_set:
            return 0.0
        for i, doc_id in enumerate(retrieved_ids or []):
            if doc_id in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_query(
        self, expected_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int = 3
    ) -> Dict[str, float]:
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
        }

    def evaluate_from_case_and_response(
        self,
        test_case: Dict,
        response: Dict,
        *,
        top_k: int = 3,
        expected_field: str = "expected_retrieval_ids",
        retrieved_field: str = "retrieved_ids",
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        expected_ids = test_case.get(expected_field)
        if expected_ids is None and isinstance(test_case.get("metadata"), dict):
            expected_ids = test_case["metadata"].get(expected_field)

        retrieved_ids = response.get(retrieved_field)
        if retrieved_ids is None and isinstance(response.get("metadata"), dict):
            retrieved_ids = response["metadata"].get(retrieved_field)

        if expected_ids is None:
            return None, f"missing `{expected_field}` in test case"
        if retrieved_ids is None:
            return None, f"missing `{retrieved_field}` in agent response"
        if not isinstance(expected_ids, list) or not all(isinstance(x, str) for x in expected_ids):
            return None, f"`{expected_field}` must be List[str]"
        if not isinstance(retrieved_ids, list) or not all(isinstance(x, str) for x in retrieved_ids):
            return None, f"`{retrieved_field}` must be List[str]"

        return self.evaluate_query(expected_ids, retrieved_ids, top_k=top_k), None

    async def evaluate_batch(self, dataset: List[Dict], *, top_k: int = 3) -> Dict[str, float]:
        """
        Chạy eval cho toàn bộ bộ dữ liệu (khi mỗi item đã có cả expected + retrieved ids).
        Dataset cần có:
        - `expected_retrieval_ids`: List[str]
        - `retrieved_ids`: List[str]
        """
        total = 0
        hit_sum = 0.0
        mrr_sum = 0.0

        for item in dataset:
            expected_ids = item.get("expected_retrieval_ids")
            retrieved_ids = item.get("retrieved_ids")
            if not isinstance(expected_ids, list) or not isinstance(retrieved_ids, list):
                continue
            if not all(isinstance(x, str) for x in expected_ids) or not all(isinstance(x, str) for x in retrieved_ids):
                continue
            scores = self.evaluate_query(expected_ids, retrieved_ids, top_k=top_k)
            total += 1
            hit_sum += scores["hit_rate"]
            mrr_sum += scores["mrr"]

        if total == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}
        return {"avg_hit_rate": hit_sum / total, "avg_mrr": mrr_sum / total}
