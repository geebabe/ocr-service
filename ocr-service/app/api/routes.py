from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import time
from app.schemas.response import APIResponse
from app.services.document_service import process_document
from app.services.vllm_client import call_vllm_inference
from app.services.parser_service import parse_vllm_output
from app.api.dependencies import get_token_header
from app.core.logger import logger

router = APIRouter()

@router.post("/ocr", response_model=APIResponse, dependencies=[Depends(get_token_header)])
async def perform_ocr(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received OCR request for file: {file.filename}")
    
    try:
        # 1. Process document (convert PDF to image if necessary, resize, encode to base64)
        image_base64, width, height = await process_document(file)
        
        # 2. Call vLLM for inference
        raw_output = await call_vllm_inference(image_base64)
        
        # 3. Clean up Markdown output
        markdown_text = parse_vllm_output(raw_output)
        
        latency = round(time.time() - start_time, 2)
        
        return APIResponse(
            success=True,
            data=markdown_text,
            metadata={"latency_seconds": latency, "image_size": f"{width}x{height}"}
        )
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=str(e)
        )
