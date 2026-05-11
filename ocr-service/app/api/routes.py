from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import time
from app.schemas.response import APIResponse, InvoiceExtraction
from app.services.document_service import process_document
from app.services.paddle_client import call_paddleocr
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
        
        # 2. Extract texts via PaddleOCR microservice
        ocr_context = await call_paddleocr(image_base64, width, height)
        
        # 3. Call vLLM for inference, injecting OCR text and bounding boxes
        raw_output = await call_vllm_inference(image_base64, ocr_context=ocr_context)
        
        # 4. Parse output and map to Pydantic models
        parsed_data = parse_vllm_output(raw_output, width, height)
        
        latency = round(time.time() - start_time, 2)
        
        return APIResponse(
            success=True,
            data=parsed_data,
            metadata={"latency_seconds": latency, "image_size": f"{width}x{height}"}
        )
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=str(e)
        )
