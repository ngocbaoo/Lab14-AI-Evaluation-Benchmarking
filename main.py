import asyncio
import json
import os
import time
from typing import Dict, List
import google.generativeai as genai
from dotenv import load_dotenv

from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator

load_dotenv()

# Cấu hình Gemini làm Judge
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    judge_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("⚠️ Cảnh báo: Không tìm thấy GEMINI_API_KEY. Judge sẽ chạy ở chế độ MOCK.")
    judge_model = None

class ExpertEvaluator:
    """
    Sử dụng LLM để chấm điểm Faithfulness (Độ trung thực) và Relevancy (Độ liên quan)
    """
    def __init__(self, *, retrieval_top_k: int = 3):
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_evaluator = RetrievalEvaluator()

    async def score(self, case, resp):
        # 1. Evaluate Retrieval (Hit Rate, MRR)
        retrieval_scores, retrieval_error = self.retrieval_evaluator.evaluate_from_case_and_response(
            case, resp, top_k=self.retrieval_top_k
        )
        if retrieval_scores is None:
            retrieval_scores = {"hit_rate": 0.0, "mrr": 0.0, "error": retrieval_error}

        # 2. Evaluate Faithfulness & Relevancy using Gemini
        faithfulness = 0.5 # Default
        relevancy = 0.5
        
        if judge_model:
            try:
                context = "\n".join(resp.get("contexts", []))
                answer = resp.get("answer", "")
                question = case.get("question", "")
                
                prompt = f"""Hãy đánh giá câu trả lời của AI dựa trên đoạn văn bản (context) và câu hỏi (question).
                Context: {context}
                Question: {question}
                Answer: {answer}

                Yêu cầu:
                1. Faithfulness (0-1): Câu trả lời có trung thành với context không? (Không lấy kiến thức ngoài).
                2. Relevancy (0-1): Câu trả lời có đúng trọng tâm câu hỏi không?

                Trả lời dưới dạng JSON: {{"faithfulness": score, "relevancy": score}}"""
                
                eval_resp = await judge_model.generate_content_async(prompt)
                score_data = json.loads(eval_resp.text.replace("```json", "").replace("```", "").strip())
                faithfulness = score_data.get("faithfulness", 0.5)
                relevancy = score_data.get("relevancy", 0.5)
            except Exception as e:
                print(f"Lỗi khi đánh giá LLM: {e}")

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "retrieval": retrieval_scores
        }

class MultiModelJudge:
    """
    Giả lập Multi-Judge bằng cách yêu cầu Gemini đóng vai 2 chuyên gia khác nhau để đồng thuận
    """
    async def evaluate_multi_judge(self, q, a, gt):
        if not judge_model:
            return {"final_score": 3.0, "agreement_rate": 0.5, "reasoning": "Mock mode: No API key"}

        prompt = f"""Đóng vai 2 chuyên gia lịch sử độc lập chấm điểm câu trả lời của AI.
        Câu hỏi: {q}
        Câu trả lời của AI: {a}
        Đáp án chuẩn (Ground Truth): {gt}

        Hãy chấm điểm trên thang điểm 5.
        Trả lời dạng JSON: {{"judge_1_score": score, "judge_2_score": score, "reasoning": "giải thích"}}"""
        
        try:
            resp = await judge_model.generate_content_async(prompt)
            data = json.loads(resp.text.replace("```json", "").replace("```", "").strip())
            s1 = data.get("judge_1_score", 0)
            s2 = data.get("judge_2_score", 0)
            
            final_score = (s1 + s2) / 2
            agreement = 1.0 if abs(s1 - s2) <= 1 else 0.5 # Nếu điểm lệch không quá 1 thì coi là đồng thuận

            return {
                "final_score": final_score,
                "agreement_rate": agreement,
                "reasoning": data.get("reasoning", "")
            }
        except:
            return {"final_score": 0.0, "agreement_rate": 0.0, "reasoning": "Error in Judge API"}

async def run_benchmark_with_results(agent_version: str):
    print(f"\n🚀 BẮT ĐẦU BENCHMARK: {agent_version}")
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # Khởi tạo Agent thực tế (Dùng FAISS/Chroma từ main_agent.py)
    agent = MainAgent(chroma_db_path="./chroma_db", chroma_collection="lab14", top_k=retrieval_top_k)
    
    # Nạp dữ liệu vào DB trước khi chạy nếu cần
    if os.path.exists("data/chunks.jsonl"):
        agent.index_documents("data/chunks.jsonl")

    runner = BenchmarkRunner(
        agent,
        ExpertEvaluator(retrieval_top_k=retrieval_top_k),
        MultiModelJudge(),
    )
    
    results = await runner.run_all(dataset)

    total = len(results)
    if total == 0: return None, None

    # Tính toán Metrics trung bình
    sum_score = sum(r["judge"]["final_score"] for r in results)
    sum_hit = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results)
    sum_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results)
    sum_agree = sum(r["judge"]["agreement_rate"] for r in results)

    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum_score / total,
            "hit_rate": sum_hit / total,
            "mrr": sum_mrr / total,
            "agreement_rate": sum_agree / total
        }
    }
    return results, summary

async def main():
    # Chạy Benchmark cho phiên bản hiện tại
    results, summary = await run_benchmark_with_results("Qwen2.5-0.5B-RAG-V1")
    
    if not summary:
        print("❌ Lỗi benchmark.")
        return

    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ CUỐI CÙNG ({summary['metadata']['version']})")
    print(f"Total cases: {summary['metadata']['total']}")
    print(f"Avg Score: {summary['metrics']['avg_score']:.2f}/5.0")
    print(f"Hit Rate: {summary['metrics']['hit_rate']:.2f}")
    print(f"MRR: {summary['metrics']['mrr']:.2f}")
    print(f"Agreement Rate: {summary['metrics']['agreement_rate']:.2f}")
    print("="*50)

    # Lưu báo cáo
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Đã lưu kết quả vào thư mục reports/")

if __name__ == "__main__":
    asyncio.run(main())
