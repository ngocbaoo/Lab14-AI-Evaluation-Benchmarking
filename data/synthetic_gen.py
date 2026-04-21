import json
import asyncio
import os
import google.generativeai as genai
from typing import List, Dict
from dotenv import load_dotenv 


load_dotenv()

# 2. Lấy API Key từ biến môi trường
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Không tìm thấy GOOGLE_API_KEY hoặc GEMINI_API_KEY trong file .env!")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

async def generate_qa_batch(text: str, num_pairs: int) -> List[Dict]:
    prompt = f"""
        Dựa trên văn bản CHÍNH SÁCH HOÀN TIỀN dưới đây, hãy tạo ra {num_pairs} cặp câu hỏi và câu trả lời.
        Yêu cầu:
        1. Đa dạng độ khó: Dễ (tra cứu trực tiếp), Trung bình (cần suy luận), Khó (kết hợp nhiều điều khoản).
        2. Phải có ít nhất 2 câu hỏi 'lừa' (adversarial): ví dụ khách hàng cố tình hiểu sai ngày tháng hoặc ngoại lệ.
        3. Trả về định dạng JSON list, mỗi object có: question, expected_answer, context, metadata (difficulty, type).
        4. Chỉ sử dụng thông tin có trong văn bản.

        VĂN BẢN:
        {text}
        """
        
    response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
    return json.loads(response.text)

async def main():
    # Nội dung chính sách hoàn tiền Hải đã gửi
    raw_text = """
    CHÍNH SÁCH HOÀN TIỀN - PHIÊN BẢN 4. Áp dụng từ 01/02/2026.
    Điều 2: Hoàn tiền khi sản phẩm lỗi NSX, yêu cầu trong 7 ngày làm việc, chưa mở seal.
    Điều 3: Ngoại lệ không hoàn tiền cho hàng kỹ thuật số (license key, subscription), Flash Sale, đã kích hoạt tài khoản.
    Điều 5: Hoàn tiền 100% qua gốc hoặc 110% store credit.
    """
    
    total_needed = 50
    batch_size = 10 # Chia nhỏ để đảm bảo chất lượng mỗi câu hỏi
    all_qa_pairs = []
    
    print(f"Đang bắt đầu tạo {total_needed} test cases...")
    
    for i in range(0, total_needed, batch_size):
        print(f"Đang tạo đợt {i//batch_size + 1}...")
        batch = await generate_qa_batch(raw_text, batch_size)
        all_qa_pairs.extend(batch)
        # Nghỉ 1 chút để tránh hit rate limit của bản Free
        await asyncio.sleep(2) 

    # Lưu kết quả
    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in all_qa_pairs:
            # Gắn thêm context gốc vào từng cặp để làm input cho RAG sau này
            pair["context"] = raw_text 
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print(f"Xong! Đã tạo và lưu {len(all_qa_pairs)} test cases vào data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
