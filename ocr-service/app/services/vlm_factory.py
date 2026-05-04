from app.core.config import settings
from app.services.vllm_client import call_vlm_inference
from app.core.logger import logger

class VLMFactory:
    """
    Factory to manage and route requests to different VLM models.
    """
    
    MODELS = {
        "qwen3vl": {
            "url": settings.VLLM_URL,
            "name": settings.VLLM_MODEL
        },
        "vintern": {
            "url": settings.VINTERN_URL,
            "name": "Vintern-1B-v2"  # Example model name
        },
        "hunyuan": {
            "url": settings.HUNYUAN_URL,
            "name": "Hunyuan-OCR"
        },
        "erax": {
            "url": settings.ERAX_URL,
            "name": "EraX-V1"
        }
    }

    @classmethod
    async def get_inference(cls, model_id: str, image_base64: str) -> str:
        model_config = cls.MODELS.get(model_id.lower())
        
        if not model_config:
            logger.warning(f"Model {model_id} not found in registry. Falling back to default: {settings.DEFAULT_MODEL}")
            model_config = cls.MODELS.get(settings.DEFAULT_MODEL)
            model_id = settings.DEFAULT_MODEL

        url = model_config["url"]
        name = model_config["name"]
        
        logger.info(f"Routing OCR request to model: {model_id} ({name}) at {url}")
        
        return await call_vlm_inference(image_base64, url, name)

vlm_factory = VLMFactory()
