import os
import json
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

# Load model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Model loaded successfully!")

# --- Outlines structured generation setup ---
_outlines_model = None
_logits_processors_cache: Dict[str, Any] = {}

def _get_outlines_model():
    """Lazily wrap the HF model with outlines for structured generation."""
    global _outlines_model
    if _outlines_model is None:
        try:
            import outlines.models.transformers
            _outlines_model = outlines.models.transformers(model, processor)
            print("Outlines model wrapper created successfully!")
        except Exception as e:
            print(f"Warning: outlines not available, structured output disabled: {e}")
    return _outlines_model

def _get_logits_processor(json_schema: dict):
    """Create or retrieve cached logits processor for a JSON schema."""
    schema_str = json.dumps(json_schema, sort_keys=True)
    if schema_str not in _logits_processors_cache:
        try:
            from outlines.processors import JSONLogitsProcessor
            tokenizer = processor.tokenizer
            lp = JSONLogitsProcessor(json_schema, tokenizer)
            _logits_processors_cache[schema_str] = lp
            print(f"Created logits processor for schema")
        except Exception as e:
            print(f"Warning: Could not create logits processor: {e}")
            return None
    return _logits_processors_cache[schema_str]


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
    guided_json: Optional[dict] = None  # JSON schema for structured output
    extra_body: Optional[dict] = None   # Alternative way to pass guided_json

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

        # Check for guided_json schema (from top-level or extra_body)
        json_schema = req.guided_json
        if json_schema is None and req.extra_body and "guided_json" in req.extra_body:
            json_schema = req.extra_body["guided_json"]

        # Build generate kwargs
        generate_kwargs = {
            **inputs,
            "max_new_tokens": req.max_tokens,
            "do_sample": (req.temperature > 0),
            "temperature": req.temperature if req.temperature > 0 else None,
        }

        # Add structured output constraint if schema provided
        if json_schema is not None:
            logits_processor = _get_logits_processor(json_schema)
            if logits_processor is not None:
                generate_kwargs["logits_processor"] = [logits_processor]
                print("Using structured output with logits processor")

        with torch.no_grad():
            generated_ids = model.generate(**generate_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
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
