import json
import os
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def map_ids():
    chunks_path = "data/chunks.jsonl"
    golden_path = "data/golden_set.jsonl"
    output_path = "data/golden_set_mapped.jsonl"

    if not os.path.exists(chunks_path) or not os.path.exists(golden_path):
        print("❌ Thiếu file chunks.jsonl hoặc golden_set.jsonl")
        return

    print("--- Đang tải dữ liệu chunks ---")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print("--- Đang xử lý mapping bộ đề ---")
    mapped_count = 0
    with open(golden_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            case = json.loads(line)
            context_text = case.get("context", "").strip()
            
            if not context_text:
                # Trường hợp Out-of-context
                case["ground_truth_id"] = "N/A"
            else:
                # Tìm chunk khớp nhất
                best_match_id = None
                max_score = 0
                
                for chunk in chunks:
                    # So khớp chuỗi (đơn giản nhưng hiệu quả cho văn bản trích dẫn)
                    score = similar(context_text[:100], chunk["text"][:100])
                    if score > max_score:
                        max_score = score
                        best_match_id = chunk["id"]
                    
                    if score > 0.95: # Khớp gần như tuyệt đối
                        break
                
                case["expected_retrieval_ids"] = [best_match_id] if best_match_id else []
                mapped_count += 1
            
            f_out.write(json.dumps(case, ensure_ascii=False) + "\n")

    # Ghi đè lại file gốc
    os.replace(output_path, golden_path)
    print(f"✅ Đã gắn thành công ID cho {mapped_count} câu hỏi trong golden_set.jsonl")

if __name__ == "__main__":
    map_ids()
