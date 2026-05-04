from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "OCR Inference Service"
    API_V1_STR: str = "/api/v1"
    
    # Default vLLM Server Settings (Qwen)
    VLLM_URL: str = "http://qwen-vlm:8000/v1/chat/completions"
    VLLM_MODEL: str = "Qwen/Qwen3-VL-2B-Instruct"
    VLLM_MAX_TOKENS: int = 1024
    VLLM_TEMPERATURE: float = 0.0
    
    # Other VLM Endpoints (Optional)
    VINTERN_URL: Optional[str] = "http://vintern-vlm:8000/v1/chat/completions"
    HUNYUAN_URL: Optional[str] = "http://hunyuan-ocr:8000/v1/chat/completions"
    ERAX_URL: Optional[str] = "http://erax-vlm:8000/v1/chat/completions"
    
    DEFAULT_MODEL: str = "qwen3vl"
    
    # Upload Settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".jpg", ".jpeg", ".png"}

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
