import os
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

app = FastAPI()

MODEL_ID = os.getenv("MODEL_ID", "erax-ai/EraX-VL-7B-V1.0")
print(f"Loading {MODEL_ID} using HuggingFace Transformers...")

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager", # replace with "flash_attention_2" if your GPU is Ampere architecture
    device_map="auto"
)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)
print("Model loaded successfully!")

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
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.06

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

        # Generation configs
        generation_config = model.generation_config
        generation_config.do_sample = True
        generation_config.temperature = req.temperature
        generation_config.top_k = 1
        generation_config.top_p = req.top_p
        generation_config.min_p = 0.1
        # generation_config.best_of = 5 # best_of usually handled by pipeline or specific generate calls
        generation_config.max_new_tokens = req.max_tokens
        generation_config.repetition_penalty = req.repetition_penalty

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                generation_config=generation_config
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

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
