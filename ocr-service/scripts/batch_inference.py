import os
import httpx
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional
import time

# Constants
DEFAULT_API_URL = "http://localhost:8000/api/v1/ocr"
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

async def process_single_file(
    client: httpx.AsyncClient, 
    file_path: Path, 
    output_dir: Path, 
    api_url: str, 
    api_key: Optional[str] = None
) -> bool:
    """Processes a single file and saves the JSON result."""
    output_file = output_dir / f"{file_path.stem}.json"
    
    # Skip if already processed
    if output_file.exists():
        return True

    headers = {}
    if api_key:
        headers["x-token"] = api_key # Matching the get_token_header logic

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            response = await client.post(api_url, files=files, headers=headers, timeout=120.0)
            
        if response.status_code == 200:
            result = response.json()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return True
        else:
            print(f"Error processing {file_path.name}: HTTP {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception processing {file_path.name}: {str(e)}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Batch process invoices for OCR.")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images/PDFs")
    parser.add_argument("--output", "-o", required=True, help="Output directory for JSON results")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--key", help="API Key (x-token header)")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="Number of concurrent requests")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect files
    files_to_process = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    
    if not files_to_process:
        print(f"No valid files found in {input_dir}")
        return

    print(f"Found {len(files_to_process)} files. Starting batch process (concurrency={args.concurrency})...")
    
    start_time = time.time()
    success_count = 0
    
    # Use a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def wrapped_process(client, file_path):
        async with semaphore:
            return await process_single_file(client, file_path, output_dir, args.url, args.key)

    async with httpx.AsyncClient() as client:
        tasks = [wrapped_process(client, f) for f in files_to_process]
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)

    duration = time.time() - start_time
    print(f"\nFinished!")
    print(f"Total files: {len(files_to_process)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(files_to_process) - success_count}")
    print(f"Total time: {duration:.2f}s (Average: {duration/len(files_to_process):.2f}s per file)")

    # Export to CSV summary
    if success_count > 0:
        import csv
        csv_path = output_dir / "summary.csv"
        print(f"\nGenerating CSV summary: {csv_path}")
        
        summary_data = []
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("success") and data.get("data"):
                        invoice = data["data"]
                        # Flatten some key fields for the CSV
                        summary_data.append({
                            "filename": json_file.stem,
                            "invoice_number": invoice.get("invoice_number", {}).get("value"),
                            "invoice_date": invoice.get("invoice_date", {}).get("value"),
                            "vendor_name": invoice.get("vendor", {}).get("name", {}).get("value"),
                            "total_amount": invoice.get("total_amount", {}).get("value"),
                            "currency": invoice.get("currency", {}).get("value")
                        })
            except Exception as e:
                print(f"Error reading {json_file.name} for CSV: {e}")

        if summary_data:
            keys = summary_data[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(summary_data)
            print(f"Successfully exported {len(summary_data)} rows to CSV.")

if __name__ == "__main__":
    asyncio.run(main())
