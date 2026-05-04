from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import time
from app.schemas.response import APIResponse, InvoiceExtraction
from app.services.vlm_factory import vlm_factory
from app.services.document_service import process_document
from app.services.parser_service import parse_vllm_output
from app.services.paddle_service import run_paddle_ocr
from app.services.matcher_service import merge_ocr_results
from app.api.dependencies import get_token_header
from app.core.logger import logger
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query

router = APIRouter()

@router.post("/ocr", response_model=APIResponse, dependencies=[Depends(get_token_header)])
async def perform_ocr(
    file: UploadFile = File(...),
    model: str = Query("qwen3vl", description="Model to use for OCR (qwen3vl, vintern, hunyuan, erax)")
):
    start_time = time.time()
    logger.info(f"Received OCR request for file: {file.filename}, model: {model}")
    
    try:
        # 1. Process document (convert PDF to image if necessary, resize, encode to base64)
        image_base64, width, height = await process_document(file)
        
        # 2. Call selected VLM and PaddleOCR concurrently
        raw_output_task = vlm_factory.get_inference(model, image_base64)
        # paddle_service is synchronous right now, but we can run it in a threadpool to not block the event loop
        ocr_tokens_task = asyncio.to_thread(run_paddle_ocr, image_base64)
        
        raw_output, ocr_tokens = await asyncio.gather(raw_output_task, ocr_tokens_task)
        
        # 3. Parse VLM output and merge with PaddleOCR tokens
        parsed_dict = parse_vllm_output(raw_output)
        final_extraction = merge_ocr_results(parsed_dict, ocr_tokens)
        
        latency = round(time.time() - start_time, 2)
        
        return APIResponse(
            success=True,
            data=final_extraction,
            metadata={
                "latency_seconds": latency, 
                "image_size": f"{width}x{height}",
                "model": model
            }
        )
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
        return APIResponse(
            success=False,
            error=str(e)
        )
