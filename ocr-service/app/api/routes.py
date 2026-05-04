from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import time
from app.schemas.response import APIResponse, InvoiceExtraction
from app.services.document_service import process_document
from app.services.vllm_client import call_vllm_inference
from app.services.parser_service import parse_vllm_output
from app.api.dependencies import get_token_header
from app.core.logger import logger
from app.services.paddle_service import run_paddle_ocr
from app.services.matcher_service import merge_ocr_results
import asyncio

router = APIRouter()

@router.post("/ocr", response_model=APIResponse, dependencies=[Depends(get_token_header)])
async def perform_ocr(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received OCR request for file: {file.filename}")
    
    try:
        # 1. Process document (convert PDF to image if necessary, resize, encode to base64)
        image_base64, width, height = await process_document(file)
        
        # 2. Call vLLM and PaddleOCR concurrently
        raw_output_task = call_vllm_inference(image_base64)
        # paddle_service is synchronous right now, but we can run it in a threadpool to not block the event loop
        ocr_tokens_task = asyncio.to_thread(run_paddle_ocr, image_base64)
        
        raw_output, ocr_tokens = await asyncio.gather(raw_output_task, ocr_tokens_task)
        
        # 3. Parse VLLM output and merge with PaddleOCR tokens
        parsed_dict = parse_vllm_output(raw_output)
        final_extraction = merge_ocr_results(parsed_dict, ocr_tokens)
        
        latency = round(time.time() - start_time, 2)
        
        return APIResponse(
            success=True,
            data=final_extraction,
            metadata={"latency_seconds": latency, "image_size": f"{width}x{height}"}
        )
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=str(e)
        )
