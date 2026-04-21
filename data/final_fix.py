import json
import os

def final_fix():
    chunks_path = "data/chunks.jsonl"
    golden_path = "data/golden_set.jsonl"
    output_path = "data/golden_set_final.jsonl"

    # Tải chunks để tìm ID
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    fixed_count = 0
    with open(golden_path, "r", encoding="utf-8", errors='ignore') as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            try:
                # Sửa lỗi ký tự lạ bằng cách replace thủ công các mẫu phổ biến hoặc dùng encode/decode
                clean_line = line.encode('utf-8', 'ignore').decode('utf-8')
                case = json.loads(clean_line)
                
                # Tìm ID dựa trên text (Exact match hoặc Substring)
                ctx = case.get("context", "").strip()
                case["expected_retrieval_ids"] = []
                
                for ck in chunks:
                    # Nếu context trong bộ đề nằm trong chunk hoặc ngược lại
                    if ctx[:30] in ck["text"] or ck["text"][:30] in ctx:
                        case["expected_retrieval_ids"] = [ck["id"]]
                        break
                
                # Sửa các lỗi font phổ biến thủ công cho các câu hỏi quan trọng
                case["question"] = case["question"].replace("Chin tranh th gi>i", "Chiến tranh thế giới")
                
                f_out.write(json.dumps(case, ensure_ascii=False) + "\n")
                fixed_count += 1
            except Exception as e:
                print(f"Error at line: {e}")
                continue

    os.replace(output_path, golden_path)
    print(f"✅ Đã FIX triệt để và gán ID cho {fixed_count} câu hỏi.")

if __name__ == "__main__":
    final_fix()
