import os
import requests
import base64
import json
import re
from PIL import Image
import io

# ⚠️ 請替換為您的 Ollama/VLM 實際 API 端點
API_URL = "http://localhost:11434/api/generate" 

def encode_full_frame_to_base64(image_path):
    """
    只執行一次：讀取圖片檔案，並將其內容 Base64 編碼。
    此 Base64 字串將在當前 Frame 的所有 VLM 呼叫中重複使用。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 確保圖片被正確讀取和編碼
    img = Image.open(image_path)
    buffer = io.BytesIO()
    # 儲存為 JPEG 格式，以優化 VLM 傳輸速度
    img.save(buffer, format=img.format or "JPEG", quality=95) 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def select_target_position_filter(
    prompt,
    tracker_id,
    x1, y1, x2, y2,
    position_desc, # 從 Notebook 計算並傳入的 **位置描述**
    base64_image_string, # 接收已編碼的圖片字串
    quiet=False
):
    """
    執行 VLM 篩選：單圖輸入，純座標+位置邏輯判斷。
    VLM 必須嚴格遵循程式計算出的 position_desc 進行邏輯判斷。
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    coord_text = f"[{x1}, {y1}, {x2}, {y2}]"
    
    # 這是您原 LLM.py 的核心約束邏輯，確保 VLM 忽略視覺細節
    system_instruction = (
        "You are performing a strict binary classification for a tracking system.\n"
        "The full image and the bounding box coordinates are provided for context.\n"
        "Your primary task is a **LOGICAL CHECK** based on the object's calculated horizontal position.\n\n"
        "Contextual Information:\n"
        f"- Object Location Description: **{position_desc}**\n"
        f"- Bounding Box Coordinates (x1, y1, x2, y2): {coord_text}\n\n"
        "Rules:\n"
        "1. **STRICTLY IGNORE** the object's appearance, color, class, blurriness, or identity in the image.\n"
        "2. **ONLY** determine if the object's **Location Description** ('{position_desc}') is relevant to the **Question**.\n"
        "3. You MUST answer EXACTLY one of the following: 'yes' or 'no'.\n"
        "4. Do NOT output any other text or explanation.\n"
    )

    full_prompt = (
        f"{system_instruction}"
        f"Question: {prompt}\n\n"
        "Answer format:\nyes or no"
    )

    payload = {
        "model": "qwen2.5vl", # ⚠️ 請替換為您的 VLM 模型名稱
        "prompt": full_prompt,
        "images": [base64_image_string], # 重複使用單次編碼的圖片
        "stream": False,
    }

    if not quiet:
        print(f"[INFO] VLM Check for ID {tracker_id} at {position_desc}...")

    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        result_text = data.get("response", "").strip().lower()
        
        # 判斷是否包含 'yes'，且不包含 'no'
        if "yes" in result_text and "no" not in result_text:
            return True
        else:
            return False

    except Exception as e:
        if not quiet:
            print(f"[ERROR] LLM request failed for ID {tracker_id}: {e}")
        return False