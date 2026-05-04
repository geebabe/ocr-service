from PIL import Image

def optimize_image(img: Image.Image, max_size: int = 1500) -> Image.Image:
    """
    Optimize image for OCR by resizing if too large, maintaining aspect ratio.
    """
    width, height = img.size
    
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
            
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    return img
