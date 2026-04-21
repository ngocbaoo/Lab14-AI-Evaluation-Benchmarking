import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        question = test_case["question"]
        expected_answer = test_case.get("expected_answer", "")

        try:
            response = await self.agent.query(question)
        except Exception as exc:
            latency = time.perf_counter() - start_time
            return {
                "test_case": question,
                "expected_answer": expected_answer,
                "agent_response": "",
                "contexts": [],
                "retrieved_ids": [],
                "latency": latency,
                "ragas": {"retrieval": {"hit_rate": 0.0, "mrr": 0.0, "error": str(exc)}},
                "judge": {
                    "final_score": 0.0,
                    "agreement_rate": 0.0,
                    "reasoning": "Agent query failed. Human review required.",
                    "individual_results": [],
                    "used_models": [],
                    "failed_models": [],
                    "needs_human_review": True,
                    "cost": {"total_usd": 0.0, "by_model": {}},
                    "tokens": {"prompt": 0, "completion": 0},
                    "status": "agent_error",
                },
                "cost": {"agent_tokens": 0, "judge_total_usd": 0.0},
                "status": "fail",
            }

        latency = time.perf_counter() - start_time
        ragas_scores = await self.evaluator.score(test_case, response)
        retrieval_scores = ragas_scores.get("retrieval", {})

        metadata = response.get("metadata", {}) if isinstance(response.get("metadata"), dict) else {}
        contexts = response.get("contexts", []) if isinstance(response.get("contexts"), list) else []
        retrieved_ids = metadata.get("retrieved_ids", [])

        judge_result = await self.judge.evaluate_multi_judge(
            question,
            response.get("answer", ""),
            expected_answer,
            contexts=contexts,
            retrieved_ids=retrieved_ids if isinstance(retrieved_ids, list) else [],
            retrieval_metrics=retrieval_scores if isinstance(retrieval_scores, dict) else {},
        )

        final_score = float(judge_result.get("final_score", 0.0) or 0.0)
        return {
            "test_case": question,
            "expected_answer": expected_answer,
            "agent_response": response.get("answer", ""),
            "contexts": contexts,
            "retrieved_ids": retrieved_ids if isinstance(retrieved_ids, list) else [],
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "cost": {
                "agent_tokens": metadata.get("tokens_used", 0),
                "judge_total_usd": judge_result.get("cost", {}).get("total_usd", 0.0),
            },
            "status": "pass" if final_score >= 3 else "fail",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)
        return results
