import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from agent.main_agent import MainAgent
from engine.llm_judge import MultiModelJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


load_dotenv()


class ExpertEvaluator:
    def __init__(self, *, retrieval_top_k: int = 3):
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_evaluator = RetrievalEvaluator()

    async def score(self, case: Dict, resp: Dict) -> Dict:
        retrieval_scores, retrieval_error = self.retrieval_evaluator.evaluate_from_case_and_response(
            case,
            resp,
            top_k=self.retrieval_top_k,
        )
        if retrieval_scores is None:
            retrieval_scores = {"hit_rate": 0.0, "mrr": 0.0, "error": retrieval_error}

        contexts = resp.get("contexts", []) if isinstance(resp.get("contexts"), list) else []
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = resp.get("metadata", {}).get("retrieved_ids", []) if isinstance(resp.get("metadata"), dict) else []

        return {
            "faithfulness": 1.0 if retrieval_scores.get("hit_rate", 0.0) > 0 else 0.0,
            "relevancy": min(1.0, len(contexts) / max(1, self.retrieval_top_k)),
            "retrieval": retrieval_scores,
            "retrieval_debug": {
                "expected_retrieval_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }


def _load_dataset() -> Optional[List[Dict]]:
    if not os.path.exists("data/golden_set.jsonl"):
        print("Missing data/golden_set.jsonl. Run 'python data/synthetic_gen.py' first.")
        return None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("data/golden_set.jsonl is empty.")
        return None
    return dataset


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_run(agent_version: str, results: List[Dict]) -> Dict:
    total = len(results)
    retrieval_items = [r.get("ragas", {}).get("retrieval", {}) for r in results]
    valid_retrieval = [
        item
        for item in retrieval_items
        if isinstance(item.get("hit_rate"), (int, float)) and isinstance(item.get("mrr"), (int, float))
    ]

    avg_score = _avg([float(r.get("judge", {}).get("final_score", 0.0) or 0.0) for r in results])
    avg_hit_rate = _avg([float(item["hit_rate"]) for item in valid_retrieval])
    avg_mrr = _avg([float(item["mrr"]) for item in valid_retrieval])
    agreement_rate = _avg([float(r.get("judge", {}).get("agreement_rate", 0.0) or 0.0) for r in results])
    pass_rate = _avg([1.0 if r.get("status") == "pass" else 0.0 for r in results])
    human_review_rate = _avg(
        [1.0 if r.get("judge", {}).get("needs_human_review") else 0.0 for r in results]
    )
    avg_latency = _avg([float(r.get("latency", 0.0) or 0.0) for r in results])
    total_cost = round(
        sum(float(r.get("judge", {}).get("cost", {}).get("total_usd", 0.0) or 0.0) for r in results),
        8,
    )
    total_prompt_tokens = sum(int(r.get("judge", {}).get("tokens", {}).get("prompt", 0) or 0) for r in results)
    total_completion_tokens = sum(
        int(r.get("judge", {}).get("tokens", {}).get("completion", 0) or 0) for r in results
    )

    per_model_cost: Dict[str, float] = {}
    for result in results:
        by_model = result.get("judge", {}).get("cost", {}).get("by_model", {})
        for model, value in by_model.items():
            per_model_cost[model] = round(per_model_cost.get(model, 0.0) + float(value or 0.0), 8)

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "hit_rate": round(avg_hit_rate, 4),
            "mrr": round(avg_mrr, 4),
            "agreement_rate": round(agreement_rate, 4),
            "pass_rate": round(pass_rate, 4),
            "human_review_rate": round(human_review_rate, 4),
            "avg_latency_seconds": round(avg_latency, 4),
        },
        "cost": {
            "total_usd": total_cost,
            "by_model": per_model_cost,
        },
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        },
        "human_review": {
            "count": sum(1 for r in results if r.get("judge", {}).get("needs_human_review")),
            "rate": round(human_review_rate, 4),
        },
    }


def _build_regression(v1_summary: Dict, v2_summary: Dict) -> Dict:
    score_delta = round(v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"], 4)
    hit_rate_delta = round(v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"], 4)
    mrr_delta = round(v2_summary["metrics"]["mrr"] - v1_summary["metrics"]["mrr"], 4)
    agreement_delta = round(
        v2_summary["metrics"]["agreement_rate"] - v1_summary["metrics"]["agreement_rate"],
        4,
    )
    pass_rate_delta = round(v2_summary["metrics"]["pass_rate"] - v1_summary["metrics"]["pass_rate"], 4)
    hitl_delta = round(
        v2_summary["metrics"]["human_review_rate"] - v1_summary["metrics"]["human_review_rate"],
        4,
    )
    cost_delta = round(v2_summary["cost"]["total_usd"] - v1_summary["cost"]["total_usd"], 8)

    decision = "APPROVE"
    reasons = []
    if score_delta < 0:
        decision = "BLOCK_RELEASE"
        reasons.append("average score regressed")
    if hit_rate_delta < 0 or mrr_delta < 0:
        decision = "BLOCK_RELEASE"
        reasons.append("retrieval quality regressed")
    if pass_rate_delta < 0:
        decision = "BLOCK_RELEASE"
        reasons.append("pass rate regressed")
    if hitl_delta > 0.1:
        decision = "BLOCK_RELEASE"
        reasons.append("human review rate increased too much")

    return {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        "avg_score_delta": score_delta,
        "hit_rate_delta": hit_rate_delta,
        "mrr_delta": mrr_delta,
        "agreement_rate_delta": agreement_delta,
        "pass_rate_delta": pass_rate_delta,
        "human_review_rate_delta": hitl_delta,
        "cost_delta_usd": cost_delta,
        "decision": decision,
        "reasons": reasons or ["candidate meets release criteria"],
    }


async def run_benchmark_with_results(agent_version: str) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    print(f"Starting benchmark for {agent_version}...")
    dataset = _load_dataset()
    if dataset is None:
        return None, None

    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    batch_size = int(os.getenv("EVAL_BATCH_SIZE", "5"))
    agent = MainAgent(
        chroma_db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "lab14"),
        top_k=retrieval_top_k,
    )
    judge = MultiModelJudge(
        openai_model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o"),
        gemini_model=os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.0-flash"),
    )
    runner = BenchmarkRunner(
        agent,
        ExpertEvaluator(retrieval_top_k=retrieval_top_k),
        judge,
    )
    try:
        results = await runner.run_all(dataset, batch_size=batch_size)
        summary = _aggregate_run(agent_version, results)
    finally:
        await judge.aclose()
    return results, summary


async def run_benchmark(version: str) -> Optional[Dict]:
    _, summary = await run_benchmark_with_results(version)
    return summary


async def main() -> None:
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v1_summary or not v2_summary or v1_results is None or v2_results is None:
        print("Benchmark could not run. Check your dataset and API configuration.")
        return

    regression = _build_regression(v1_summary, v2_summary)
    v2_summary["regression"] = regression

    print("\n--- REGRESSION SUMMARY ---")
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.2f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Delta: {regression['avg_score_delta']:+.2f}")
    print(f"Decision: {regression['decision']}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": {
                    "Agent_V1_Base": v1_results,
                    "Agent_V2_Optimized": v2_results,
                },
                "summaries": {
                    "Agent_V1_Base": v1_summary,
                    "Agent_V2_Optimized": v2_summary,
                },
                "regression": regression,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if regression["decision"] == "APPROVE":
        print("Release gate: APPROVE")
    else:
        print("Release gate: BLOCK_RELEASE")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
