import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import google.generativeai as genai

load_dotenv()

class MultiModelJudge:
    def __init__(
        self,
        openai_model: str = "gpt-4o",
        secondary_model: str = "gpt-4o-mini",
        human_review_band: float = 0.5,
    ):
        self.openai_model = openai_model
        self.secondary_model = secondary_model
        self.human_review_band = human_review_band
        self._openai_client = None
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self._openai_client = AsyncOpenAI(api_key=api_key)
        
        # Init Gemini if available (fallback)
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_api_key:
            genai.configure(api_key=google_api_key)
        self.gemini_model = "gemini-1.5-flash"

    async def aclose(self) -> None:
        if self._openai_client is not None:
            await self._openai_client.close()

    def _build_prompt(self, question: str, context: str, answer: str, expected: str, **kwargs) -> str:
        # Nếu context rỗng, thử lấy từ kwargs['contexts']
        if not context and "contexts" in kwargs:
            context = "\n\n".join(kwargs["contexts"])
            
        return f"""You are an impartial evaluator. 
Judge the answer based on correctness and context grounding.
Return JSON: {{"score": <1-5>, "reasoning": "<short text>"}}

Question: {question}
Expected Answer: {expected}
Agent Answer: {answer}
Context: {context}
"""

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str = "", context: str = "", **kwargs) -> Dict[str, Any]:
        # Xử lý việc tên tham số có thể khác nhau (expected vs ground_truth)
        expected = ground_truth or kwargs.get("expected", "")
        
        prompt = self._build_prompt(question, context, answer, expected, **kwargs)
        
        tasks = []
        # Judge 1: Primary OpenAI
        tasks.append(self._judge_with_openai(prompt, self.openai_model))
        
        # Judge 2: Secondary Model (OpenAI or Gemini)
        if "gpt" in self.secondary_model:
            tasks.append(self._judge_with_openai(prompt, self.secondary_model))
        else:
            tasks.append(self._judge_with_gemini(prompt))

        results = await asyncio.gather(*tasks)
        return self._consensus_logic(results)

    async def _judge_with_openai(self, prompt: str, model_name: str) -> Dict[str, Any]:
        if not self._openai_client:
            return {"model": model_name, "status": "error", "error": "No API Key"}
        
        start_time = time.perf_counter()
        try:
            response = await self._openai_client.chat.completions.create(
                model=model_name,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )
            payload = json.loads(response.choices[0].message.content)
            return {
                "model": model_name,
                "status": "ok",
                "score": float(payload.get("score", 1)),
                "reasoning": payload.get("reasoning", ""),
                "latency": time.perf_counter() - start_time,
                "estimated_cost": 0.0, # Simple for now
                "tokens_prompt": response.usage.prompt_tokens,
                "tokens_completion": response.usage.completion_tokens
            }
        except Exception as e:
            return {"model": model_name, "status": "error", "error": str(e), "score": None, "reasoning": ""}

    async def _judge_with_gemini(self, prompt: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        try:
            model = genai.GenerativeModel(self.gemini_model)
            response = await asyncio.to_thread(model.generate_content, prompt, 
                                               generation_config={"response_mime_type": "application/json"})
            payload = json.loads(response.text)
            return {
                "model": self.gemini_model,
                "status": "ok",
                "score": float(payload.get("score", 1)),
                "reasoning": payload.get("reasoning", ""),
                "latency": time.perf_counter() - start_time,
                "estimated_cost": 0.0,
                "tokens_prompt": 0,
                "tokens_completion": 0
            }
        except Exception as e:
            return {"model": self.gemini_model, "status": "error", "error": str(e), "score": None, "reasoning": ""}

    def _consensus_logic(self, results: List[Dict]) -> Dict[str, Any]:
        successful = [r for r in results if r["status"] == "ok" and r["score"] is not None]
        if not successful:
            return {"final_score": 0.0, "status": "error", "reasoning": "All judges failed"}
        
        final_score = sum(r["score"] for r in successful) / len(successful)
        agreement_rate = 0.0
        if len(successful) >= 2:
            agreement_rate = 1.0 if round(successful[0]["score"]) == round(successful[1]["score"]) else 0.0
            
        reasoning = " | ".join([f"{r['model']}: {r['reasoning']}" for r in successful])
        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement_rate,
            "reasoning": reasoning,
            "individual_results": results,
            "used_models": [r["model"] for r in successful],
            "failed_models": [r["model"] for r in results if r["status"] != "ok"],
            "needs_human_review": len(successful) < 2 or abs(successful[0]["score"] - successful[1]["score"]) > 1.5,
            "status": "ok"
        }
