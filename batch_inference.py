#!/usr/bin/env python3
"""
Batch inference script for OCR invoice extraction.
Sends images in parallel to leverage vLLM's continuous batching.
"""

import os
import json
import glob
import argparse
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Cập nhật URL nếu cần
API_URL = "https://8000-01kqvkhktq54k0sn3vdvakqzfk.cloudspaces.litng.ai/api/v1/ocr"
IMAGE_DIR = "./data"
OUTPUT_DIR = "result_erax_v1"
MAX_WORKERS = 8  # Tăng lên để tận dụng Batching của vLLM
MAX_RETRIES = 2   # Thử lại nếu timeout


def process_image(image_path: str, api_url: str, output_dir: str) -> dict:
    """Send a single image to the OCR API and save the result with retry logic."""
    image_name = Path(image_path).stem
    output_path = Path(output_dir) / f"{image_name}.json"

    # Skip if already processed
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"file": image_path, "status": "skipped", "output": str(output_path)}

    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    api_url,
                    headers={"accept": "application/json"},
                    files={"file": (Path(image_path).name, f, "multipart/form-data")},
                    timeout=180, # Tăng timeout vì batching có thể đợi lâu hơn một chút
                )

            if response.status_code == 200:
                result = response.json()
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(result, out_f, ensure_ascii=False, indent=2)
                return {"file": image_path, "status": "success", "output": str(output_path)}
            
            # Nếu server overload (503/504) hoặc lỗi khác
            if response.status_code in [502, 503, 504]:
                attempt += 1
                time.sleep(2 * attempt)
                continue
                
            return {"file": image_path, "status": "error", "error": f"HTTP {response.status_code}: {response.text[:100]}"}

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            attempt += 1
            if attempt <= MAX_RETRIES:
                time.sleep(5)
                continue
            return {"file": image_path, "status": "error", "error": "Timeout/Connection Error after retries"}
        except Exception as e:
            return {"file": image_path, "status": "error", "error": str(e)}
    
    return {"file": image_path, "status": "error", "error": "Max retries exceeded"}


def main():
    parser = argparse.ArgumentParser(description="Batch OCR inference utilizing vLLM batching")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel requests to vLLM")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    # Find images
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(os.path.join(args.image_dir, pattern)))
    all_images = sorted(set(all_images))

    if not all_images:
        print(f"No images found in {args.image_dir}")
        return

    # Filter
    images_to_process = []
    skipped_count = 0
    for img in all_images:
        out_p = Path(args.output_dir) / f"{Path(img).stem}.json"
        if not args.no_skip and out_p.exists() and out_p.stat().st_size > 0:
            skipped_count += 1
        else:
            images_to_process.append(img)

    if args.limit > 0:
        images_to_process = images_to_process[:args.limit]

    if not images_to_process:
        print(f"Nothing to process. Skipped: {skipped_count}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Starting batch inference with {args.workers} workers (vLLM Batching mode)")
    print(f"Files to process: {len(images_to_process)} (Skipped: {skipped_count})")
    print("-" * 60)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, img, args.api_url, args.output_dir): img for img in images_to_process}
        
        with tqdm(total=len(futures), desc="Inference", unit="img") as pbar:
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                if res["status"] == "error":
                    tqdm.write(f"  ✗ ERROR {Path(res['file']).name}: {res['error']}")
                pbar.update(1)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    print("-" * 60)
    print(f"✅ Done! Success: {success}, Failed: {failed}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
