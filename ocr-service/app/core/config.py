from pydantic_settings import BaseSettings
from typing import Optional
import paddle

if paddle.is_compiled_with_cuda():
    n = paddle.device.cuda.device_count()
    print(f"GPU count   : {n}")
    DEVICE = "gpu:0"
else:
    DEVICE = "cpu"

class Settings(BaseSettings):
    PROJECT_NAME: str = "OCR Inference Service"
    API_V1_STR: str = "/api/v1"
    
    # vLLM Server Settings
    VLLM_URL: str = "http://qwen-model:8000/v1/chat/completions"
    VLLM_MODEL: str = "Qwen/Qwen3-VL-2B-Instruct"  # overridden by VLLM_MODEL env var
    VLLM_MAX_TOKENS: int = 8192
    VLLM_TEMPERATURE: float = 0.0
    
    # PaddleOCR Settings
    PADDLE_DEVICE: str = DEVICE
    
    # Upload Settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".jpg", ".jpeg", ".png"}

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
