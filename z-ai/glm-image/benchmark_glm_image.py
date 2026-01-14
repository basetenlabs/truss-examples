import argparse
import base64
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

BASE_PROMPT = "Adriana Chechik"
API_URL = "https://model-4w7e5m6q.api.baseten.co/environments/production/predict"
API_KEY = os.environ.get("BASETEN_API_KEY")


def generate_prompt():
    random_char = random.choice(string.ascii_letters + string.digits)
    return f"{random_char} {BASE_PROMPT}"


def save_b64_image(b64_data, output_dir, request_id):
    image_data = base64.b64decode(b64_data)
    output_path = output_dir / f"image_{request_id}.png"
    with open(output_path, "wb") as f:
        f.write(image_data)
    return output_path


def make_request(request_id, output_dir):
    client = requests.Session()
    prompt = generate_prompt()

    start_time = time.time()
    resp = client.post(
        API_URL,
        headers={"Authorization": f"Api-Key {API_KEY}"},
        json={
            "n": 1,
            "size": "1024x1024",
            "guidance_scale": 2.5,
            "prompt": prompt,
            "response_format": "b64_json",
            "num_inference_steps": 8,
        },
    )
    elapsed = time.time() - start_time

    if resp.status_code == 200:
        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            b64_data = data["data"][0]["b64_json"]
            output_path = save_b64_image(b64_data, output_dir, request_id)
            return {
                "request_id": request_id,
                "success": True,
                "elapsed": elapsed,
                "output_path": str(output_path),
            }

    return {
        "request_id": request_id,
        "success": False,
        "elapsed": elapsed,
        "status_code": resp.status_code,
        "error": resp.text,
    }


def run_benchmark(concurrency, max_runtime, max_requests, output_base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"run_{timestamp}_concurrency_{concurrency}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting benchmark with concurrency={concurrency}")
    print(f"Output directory: {output_dir}")

    start_time = time.time()
    request_count = 0
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []

        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_runtime:
                print(f"Max runtime ({max_runtime}s) reached")
                break

            if request_count >= max_requests:
                print(f"Max requests ({max_requests}) reached")
                break

            if len(futures) < concurrency:
                future = executor.submit(make_request, request_count, output_dir)
                futures.append(future)
                request_count += 1

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                futures.remove(future)
                break

    for future in as_completed(futures):
        result = future.result()
        results.append(result)

    total_time = time.time() - start_time
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nResults for concurrency={concurrency}:")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total time: {total_time:.2f}s")

    if successful:
        avg_time = sum(r["elapsed"] for r in successful) / len(successful)
        print(f"  Avg request time: {avg_time:.2f}s")
        print(f"  Requests/second: {len(successful) / total_time:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flux API")
    parser.add_argument(
        "--concurrency",
        action="append",
        type=int,
        required=True,
        help="Concurrency levels to test (can be specified multiple times)",
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=120,
        help="Maximum runtime in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=128,
        help="Maximum number of requests (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Base output directory (default: benchmark_results)",
    )

    args = parser.parse_args()

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for concurrency in args.concurrency:
        results = run_benchmark(
            concurrency, args.max_runtime, args.max_requests, output_base_dir
        )
        all_results[concurrency] = results

    print("\n" + "=" * 50)
    print("Benchmark complete!")
    print(f"Results saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
