import base64
import fitz  # PyMuPDF
from fastapi import UploadFile, HTTPException
from PIL import Image
import io
from app.core.logger import logger
from app.utils.image_utils import optimize_image

async def process_document(file: UploadFile) -> tuple[str, int, int]:
    """
    Process the uploaded file. 
    If PDF, converts the first page to an image.
    If Image, reads it.
    Returns: base64_encoded_string, width, height
    """
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        if filename.endswith(".pdf"):
            logger.info("Processing PDF document")
            pdf_document = fitz.open(stream=content, filetype="pdf")
            if len(pdf_document) == 0:
                raise ValueError("PDF is empty")
            
            # Extract first page as image
            page = pdf_document[0]
            pix = page.get_pixmap(dpi=200) # High enough DPI for OCR
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            logger.info("Processing image document")
            img = Image.open(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Optimize image (resize, correct orientation)
        img = optimize_image(img)
        
        width, height = img.size
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str, width, height
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid document: {str(e)}")
