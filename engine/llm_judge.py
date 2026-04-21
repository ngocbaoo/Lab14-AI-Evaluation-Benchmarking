import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()


class MultiModelJudge:
    def __init__(
        self,
        *,
        openai_model: str = "gpt-4o",
        gemini_model: str = "gemini-2.0-flash",
        human_review_band: float = 0.5,
    ):
        self.openai_model = openai_model
        self.gemini_model = gemini_model
        self.human_review_band = human_review_band
        self._openai_client: Optional[AsyncOpenAI] = None
        self._init_clients()

    async def aclose(self) -> None:
        if self._openai_client is not None:
            await self._openai_client.close()

    def _init_clients(self) -> None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=openai_api_key)
        if google_api_key:
            genai.configure(api_key=google_api_key)

    def _build_prompt(
        self,
        *,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Optional[List[str]],
        retrieved_ids: Optional[List[str]],
        retrieval_metrics: Optional[Dict[str, Any]],
    ) -> str:
        contexts_text = "\n\n".join(f"[Context {idx + 1}]\n{ctx}" for idx, ctx in enumerate(contexts or []))
        metrics_text = json.dumps(retrieval_metrics or {}, ensure_ascii=False)
        ids_text = json.dumps(retrieved_ids or [], ensure_ascii=False)

        rubric = (
            "Score from 1 to 5 using one final score only.\n"
            "Use these criteria internally when deciding the final score:\n"
            "- Answer correctness compared with the expected answer.\n"
            "- Groundedness in the retrieved context.\n"
            "- Retrieval usefulness and whether the retrieved evidence supports the answer.\n"
            "- Hallucination avoidance.\n"
            "- Completeness relative to the user question.\n\n"
            "Interpretation:\n"
            "1 = wrong or hallucinated, unsupported by retrieval.\n"
            "2 = weak answer, major mistakes or weak grounding.\n"
            "3 = acceptable and mostly correct, enough grounding to pass.\n"
            "4 = strong, correct, grounded, and useful.\n"
            "5 = excellent, fully correct, grounded, and complete."
        )

        return (
            "You are an impartial evaluator for a RAG system.\n"
            "Judge both answer quality and retrieval quality.\n"
            "Return JSON only with this schema:\n"
            '{"score": <number 1-5>, "reasoning": "<short reason>"}\n\n'
            f"{rubric}\n\n"
            f"Question:\n{question}\n\n"
            f"Expected Answer:\n{ground_truth}\n\n"
            f"Agent Answer:\n{answer}\n\n"
            f"Retrieved IDs:\n{ids_text}\n\n"
            f"Retrieval Metrics:\n{metrics_text}\n\n"
            f"Retrieved Contexts:\n{contexts_text or '[No retrieved context provided]'}\n"
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            raise ValueError("empty judge response")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("judge response is not valid JSON")
            return json.loads(raw[start : end + 1])

    def _normalize_score(self, value: Any) -> float:
        score = float(value)
        return max(1.0, min(5.0, score))

    def _get_pricing(self, model: str) -> Dict[str, float]:
        defaults = {
            self.openai_model: {
                "input_per_1m": float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0")),
                "output_per_1m": float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0")),
            },
            self.gemini_model: {
                "input_per_1m": float(os.getenv("GOOGLE_INPUT_COST_PER_1M", "0")),
                "output_per_1m": float(os.getenv("GOOGLE_OUTPUT_COST_PER_1M", "0")),
            },
        }
        return defaults.get(model, {"input_per_1m": 0.0, "output_per_1m": 0.0})

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self._get_pricing(model)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input_per_1m"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output_per_1m"]
        return round(input_cost + output_cost, 8)

    def _judge_needs_human_review(self, *, score: float, status: str, reasoning: str) -> bool:
        if status != "ok":
            return True
        if not reasoning.strip():
            return True
        return abs(score - 3.0) <= self.human_review_band

    async def _judge_with_openai(self, prompt: str) -> Dict[str, Any]:
        if self._openai_client is None:
            return {
                "model": self.openai_model,
                "status": "error",
                "error": "missing OPENAI_API_KEY",
                "score": None,
                "reasoning": "",
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "latency": 0.0,
                "estimated_cost": 0.0,
            }

        start_time = time.perf_counter()
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.openai_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict evaluation judge. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            latency = time.perf_counter() - start_time
            content = response.choices[0].message.content or "{}"
            payload = self._extract_json(content)
            prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
            score = self._normalize_score(payload.get("score", 1))
            reasoning = str(payload.get("reasoning", "")).strip()
            return {
                "model": self.openai_model,
                "status": "ok",
                "score": score,
                "reasoning": reasoning,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "latency": latency,
                "estimated_cost": self._estimate_cost(self.openai_model, prompt_tokens, completion_tokens),
            }
        except Exception as exc:
            return {
                "model": self.openai_model,
                "status": "error",
                "error": str(exc),
                "score": None,
                "reasoning": "",
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "latency": time.perf_counter() - start_time,
                "estimated_cost": 0.0,
            }

    def _judge_with_gemini_sync(self, prompt: str) -> Dict[str, Any]:
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            return {
                "model": self.gemini_model,
                "status": "error",
                "error": "missing GOOGLE_API_KEY or GEMINI_API_KEY",
                "score": None,
                "reasoning": "",
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "latency": 0.0,
                "estimated_cost": 0.0,
            }

        start_time = time.perf_counter()
        try:
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0, "response_mime_type": "application/json"},
            )
            latency = time.perf_counter() - start_time
            payload = self._extract_json(getattr(response, "text", "") or "{}")
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            score = self._normalize_score(payload.get("score", 1))
            reasoning = str(payload.get("reasoning", "")).strip()
            return {
                "model": self.gemini_model,
                "status": "ok",
                "score": score,
                "reasoning": reasoning,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "latency": latency,
                "estimated_cost": self._estimate_cost(self.gemini_model, prompt_tokens, completion_tokens),
            }
        except Exception as exc:
            return {
                "model": self.gemini_model,
                "status": "error",
                "error": str(exc),
                "score": None,
                "reasoning": "",
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "latency": time.perf_counter() - start_time,
                "estimated_cost": 0.0,
            }

    async def _judge_with_gemini(self, prompt: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._judge_with_gemini_sync, prompt)

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        *,
        contexts: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        retrieval_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
            retrieved_ids=retrieved_ids,
            retrieval_metrics=retrieval_metrics,
        )

        openai_result, gemini_result = await asyncio.gather(
            self._judge_with_openai(prompt),
            self._judge_with_gemini(prompt),
        )
        judge_results = [openai_result, gemini_result]
        successful = [result for result in judge_results if result["status"] == "ok" and result["score"] is not None]
        failed_models = [result["model"] for result in judge_results if result["status"] != "ok"]

        if not successful:
            return {
                "final_score": 0.0,
                "agreement_rate": 0.0,
                "reasoning": "Both judges failed. Human review required.",
                "individual_results": judge_results,
                "used_models": [],
                "failed_models": failed_models,
                "needs_human_review": True,
                "cost": {"total_usd": 0.0, "by_model": {result["model"]: result["estimated_cost"] for result in judge_results}},
                "tokens": {
                    "prompt": sum(result["tokens_prompt"] for result in judge_results),
                    "completion": sum(result["tokens_completion"] for result in judge_results),
                },
                "status": "judge_error",
            }

        final_score = sum(result["score"] for result in successful) / len(successful)
        agreement_rate = 0.0
        if len(successful) == 2:
            agreement_rate = 1.0 if round(successful[0]["score"]) == round(successful[1]["score"]) else 0.0

        needs_human_review = len(successful) == 1
        needs_human_review = needs_human_review or any(
            self._judge_needs_human_review(
                score=result["score"],
                status=result["status"],
                reasoning=result["reasoning"],
            )
            for result in successful
        )
        if len(successful) == 2 and abs(successful[0]["score"] - successful[1]["score"]) > 1.0:
            needs_human_review = True

        reasoning = " | ".join(
            f'{result["model"]}: {result["reasoning"]}'
            for result in successful
            if result["reasoning"]
        ).strip()

        total_cost = round(sum(result["estimated_cost"] for result in judge_results), 8)
        return {
            "final_score": round(final_score, 4),
            "agreement_rate": agreement_rate,
            "reasoning": reasoning or "No judge reasoning returned.",
            "individual_results": judge_results,
            "used_models": [result["model"] for result in successful],
            "failed_models": failed_models,
            "needs_human_review": needs_human_review,
            "cost": {
                "total_usd": total_cost,
                "by_model": {result["model"]: result["estimated_cost"] for result in judge_results},
            },
            "tokens": {
                "prompt": sum(result["tokens_prompt"] for result in judge_results),
                "completion": sum(result["tokens_completion"] for result in judge_results),
            },
            "status": "ok",
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, str]:
        return {
            "status": "not_implemented",
            "message": "Position bias checking is out of scope for this lab implementation.",
        }
