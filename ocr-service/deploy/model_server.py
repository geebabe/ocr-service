import os
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

app = FastAPI()

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
print(f"Loading {MODEL_ID} using HuggingFace Transformers...")

# Load model and processor with optimizations
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2", # Force Flash Attention 2
    trust_remote_code=True
)

# Limit visual tokens to speed up inference significantly.
# Default can be very high, leading to long processing times.
# 1280*28*28 is approx 1MP, which is usually perfect for OCR.
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)
print(f"Model {MODEL_ID} loaded successfully with Flash Attention 2!")

@app.get("/health")
async def health():
    return {"status": "healthy"}

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Any # Can be str or List[MessageContent]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 1024
    temperature: float = 0.0

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        # Format messages for Qwen processor
        formatted_messages = []
        for msg in req.messages:
            if isinstance(msg.content, str):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            else:
                # Handle list of dicts (image + text)
                content_list = []
                for item in msg.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_list.append({"type": "text", "text": item.get("text")})
                        elif item.get("type") == "image_url":
                            # Qwen processor expects image as a URL or base64 data URI
                            image_url = item["image_url"]["url"]
                            content_list.append({"type": "image", "image": image_url})
                formatted_messages.append({"role": msg.role, "content": content_list})

        # Apply chat template
        text = processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(formatted_messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        import time
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=(req.temperature > 0),
                temperature=req.temperature if req.temperature > 0 else None
            )

        end_time = time.time()
        inference_duration = end_time - start_time
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"Inference completed in {inference_duration:.2f}s. "
              f"Input tokens: {inputs.input_ids.shape[1]}, "
              f"Output tokens: {len(generated_ids_trimmed[0])}")

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    }
                }
            ]
        }
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
