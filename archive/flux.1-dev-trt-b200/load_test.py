#!/usr/bin/env python3
"""
Test script for the Flux model using Truss API endpoint.

Usage:
    python test_truss.py
    python test_truss.py --save-image
    python test_truss.py --prompt "your custom prompt here"
"""

import argparse
import base64
import os
import time
import requests
import concurrent.futures
from typing import List, Dict, Any


def test_truss_api_call(
    prompt="a beautiful photograph of Mt. Fuji during cherry blossom, photorealistic, high quality",
    height=1024,
    width=1024,
    steps=30,
    guidance_scale=3.5,
    seed=42,
):
    """Test Truss API endpoint prediction."""
    print(f"Testing Truss API endpoint with prompt: '{prompt}'")

    # Prepare input - same structure as test_model.py
    model_input = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted",
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "batch_size": 1,
        "batch_count": 1,
    }

    # API endpoint configuration
    api_url = "https://app.baseten.co/models/{MODEL_ID}/predict"
    api_key = os.getenv(
        "BASETEN_API_KEY", "YOUR_API_KEY"
    )  # Get from environment variable

    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }

    try:
        start_time = time.time()
        response = requests.post(api_url, headers=headers, json=model_input)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()

            if result.get("status") == "success":
                print("✅ Truss API prediction successful!")
                print(f"   Time: {end_time - start_time:.2f} seconds")
                print(f"   Prompt: {result.get('prompt', prompt)}")
                print(
                    f"   Dimensions: {result.get('width', width)}x{result.get('height', height)}"
                )
                print(f"   Steps: {result.get('num_inference_steps', steps)}")
                print(
                    f"   Guidance Scale: {result.get('guidance_scale', guidance_scale)}"
                )
                print(f"   Seed: {result.get('seed', seed)}")

                return {
                    "result": result,
                    "time": end_time - start_time,
                    "success": True,
                }
            else:
                print(
                    f"❌ Truss API prediction failed: {result.get('error', 'Unknown error')}"
                )
                return {
                    "result": None,
                    "time": end_time - start_time,
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                }
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return {
                "result": None,
                "time": end_time - start_time,
                "success": False,
                "error": f"HTTP {response.status_code}",
            }

    except Exception as e:
        print(f"❌ Truss API prediction failed with exception: {e}")
        return {"result": None, "time": 0, "success": False, "error": str(e)}


def test_concurrent_requests(
    prompts: List[str],
    max_workers: int = 5,
    height=1024,
    width=1024,
    steps=30,
    guidance_scale=3.5,
    seed=42,
    sizes: List[tuple] = None,
):
    """Test Truss API endpoint with concurrent requests."""
    print(
        f"🚀 Starting concurrent load test with {len(prompts)} requests, max {max_workers} workers"
    )
    if sizes:
        print(f"📏 Using variable image sizes: {len(sizes)} different dimensions")
    print("=" * 60)

    results = []
    start_time = time.time()

    def make_request(prompt, request_id):
        """Make a single API request."""
        # Use variable size if provided, otherwise use default
        if sizes and request_id < len(sizes):
            h, w = sizes[request_id]
            print(
                f"📤 Sending request {request_id + 1}/{len(prompts)} ({h}x{w}): '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
            )
            return test_truss_api_call(prompt, h, w, steps, guidance_scale, seed)
        else:
            print(
                f"📤 Sending request {request_id + 1}/{len(prompts)}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
            )
            return test_truss_api_call(
                prompt, height, width, steps, guidance_scale, seed
            )

    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests
        future_to_prompt = {
            executor.submit(make_request, prompt, i): (prompt, i)
            for i, prompt in enumerate(prompts)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt, request_id = future_to_prompt[future]
            try:
                result = future.result()
                results.append(result)
                print(f"📥 Completed request {request_id + 1}/{len(prompts)}")
            except Exception as e:
                print(f"❌ Request {request_id + 1} failed with exception: {e}")
                results.append(
                    {"result": None, "time": 0, "success": False, "error": str(e)}
                )

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]

    if successful_requests:
        avg_time = sum(r["time"] for r in successful_requests) / len(
            successful_requests
        )
        min_time = min(r["time"] for r in successful_requests)
        max_time = max(r["time"] for r in successful_requests)
    else:
        avg_time = min_time = max_time = 0

    # Print summary
    print("\n" + "=" * 60)
    print("📊 LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total requests: {len(prompts)}")
    print(f"Successful: {len(successful_requests)}")
    print(f"Failed: {len(failed_requests)}")
    print(f"Success rate: {len(successful_requests) / len(prompts) * 100:.1f}%")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average request time: {avg_time:.2f} seconds")
    print(f"Min request time: {min_time:.2f} seconds")
    print(f"Max request time: {max_time:.2f} seconds")
    print(f"Throughput: {len(successful_requests) / total_time:.2f} requests/second")

    if failed_requests:
        print("\n❌ Failed requests:")
        for i, result in enumerate(failed_requests):
            print(f"   Request {i + 1}: {result.get('error', 'Unknown error')}")

    return results


def get_varied_prompts(num_prompts: int) -> List[str]:
    """Generate a list of varied prompts for load testing."""
    base_prompts = [
        "a beautiful photograph of Mt. Fuji during cherry blossom, photorealistic, high quality",
        "a majestic dragon soaring through a mystical forest, digital art, detailed",
        "a cozy coffee shop interior with warm lighting, people working on laptops, photorealistic",
        "a futuristic cityscape at sunset with flying cars and neon lights, cinematic",
        "a serene mountain lake reflecting snow-capped peaks, nature photography, high resolution",
        "a steampunk airship floating above Victorian-era buildings, detailed illustration",
        "a magical library with floating books and glowing crystals, fantasy art",
        "a peaceful Japanese garden with koi pond and cherry blossoms, traditional art style",
        "a cyberpunk street scene with neon signs and rain, Blade Runner style",
        "a whimsical treehouse village connected by rope bridges, children's book illustration",
        "a dramatic storm over the ocean with lightning and waves, nature photography",
        "a cozy cabin in the woods with smoke from chimney, winter scene, photorealistic",
        "a space station orbiting Earth with stars and nebula in background, sci-fi art",
        "a bustling medieval marketplace with merchants and colorful stalls, fantasy",
        "a tranquil zen meditation room with candles and incense, minimalist design",
        "a roaring waterfall in a tropical jungle with exotic birds, nature photography",
        "a steampunk robot butler serving tea in a Victorian parlor, detailed illustration",
        "a magical crystal cave with glowing formations and underground lake, fantasy",
        "a peaceful farm at golden hour with rolling hills and grazing animals, pastoral",
        "a futuristic robot city with advanced technology and clean architecture, sci-fi",
    ]

    # If we need more prompts than we have, cycle through them
    if num_prompts <= len(base_prompts):
        return base_prompts[:num_prompts]
    else:
        # Cycle through prompts and add variations
        prompts = []
        for i in range(num_prompts):
            base_prompt = base_prompts[i % len(base_prompts)]
            if i >= len(base_prompts):
                # Add variation number for additional prompts
                variation_num = (i // len(base_prompts)) + 1
                prompts.append(f"{base_prompt} (variation {variation_num})")
            else:
                prompts.append(base_prompt)
        return prompts


def get_variable_sizes(num_sizes: int, size_range: str = "512-1024") -> List[tuple]:
    """Generate a list of variable image sizes for load testing."""
    try:
        min_size, max_size = map(int, size_range.split("-"))
        # Ensure sizes are multiples of 8 (model requirement)
        min_size = (min_size // 8) * 8
        max_size = (max_size // 8) * 8

        # Common sizes that work well with the model
        common_sizes = [
            (512, 512),
            (512, 768),
            (768, 512),
            (768, 768),
            (768, 1024),
            (1024, 768),
            (1024, 1024),
            (1024, 1280),
            (1280, 1024),
            (1280, 1280),
            (1280, 1536),
            (1536, 1280),
            (1536, 1536),
            (1536, 1792),
            (1792, 1536),
            (1792, 1792),
            (1792, 2048),
            (2048, 1792),
            (2048, 2048),
        ]

        # Filter sizes within the specified range
        valid_sizes = [
            (h, w)
            for h, w in common_sizes
            if min_size <= h <= max_size and min_size <= w <= max_size
        ]

        if not valid_sizes:
            # Fallback to simple multiples of 8 within range
            valid_sizes = []
            for size in range(min_size, max_size + 1, 64):  # Step by 64 for variety
                if size % 8 == 0:
                    valid_sizes.append((size, size))

        # If we need more sizes than available, cycle through them
        if num_sizes <= len(valid_sizes):
            return valid_sizes[:num_sizes]
        else:
            # Cycle through sizes and add variations
            sizes = []
            for i in range(num_sizes):
                base_size = valid_sizes[i % len(valid_sizes)]
                if i >= len(valid_sizes):
                    # Add variation by slightly modifying dimensions
                    variation_num = (i // len(valid_sizes)) + 1
                    h, w = base_size
                    # Add small variations while keeping multiples of 8
                    h_var = h + (variation_num * 8) % 64
                    w_var = w + (variation_num * 8) % 64
                    sizes.append((h_var, w_var))
                else:
                    sizes.append(base_size)
            return sizes

    except Exception as e:
        print(f"❌ Error parsing size range '{size_range}': {e}")
        print("   Using default size 1024x1024")
        return [(1024, 1024)] * num_sizes


def save_image(result, filename="test_truss_output.jpg", output_dir="./output"):
    """Save the generated image to a file using the same decoding logic as test_model.py."""
    # Handle both old and new result formats
    if isinstance(result, dict) and "result" in result:
        # New format from concurrent testing
        actual_result = result["result"]
        if not actual_result or actual_result.get("status") != "success":
            print("❌ No valid result to save")
            return
    else:
        # Old format from single request
        actual_result = result
        if not actual_result or actual_result.get("status") != "success":
            print("❌ No valid result to save")
            return

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use output directory for filename
        output_path = os.path.join(output_dir, filename)

        # Decode base64 image - same logic as test_model.py
        image_data = base64.b64decode(actual_result["data"])

        # Save to file
        with open(output_path, "wb") as f:
            f.write(image_data)

        print(f"✅ Image saved as '{output_path}'")

        # Try to open the image (macOS)
        try:
            os.system(f"open {output_path}")
            print("✅ Image opened in default viewer")
        except:
            print(f"📁 Image saved at: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"❌ Failed to save image: {e}")


def save_image_bulk(result, filename="test_truss_output.jpg", output_dir="./output"):
    """Save the generated image to a file without opening it (for bulk operations)."""
    # Handle both old and new result formats
    if isinstance(result, dict) and "result" in result:
        # New format from concurrent testing
        actual_result = result["result"]
        if not actual_result or actual_result.get("status") != "success":
            return False
    else:
        # Old format from single request
        actual_result = result
        if not actual_result or actual_result.get("status") != "success":
            return False

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use output directory for filename
        output_path = os.path.join(output_dir, filename)

        # Decode base64 image - same logic as test_model.py
        image_data = base64.b64decode(actual_result["data"])

        # Save to file
        with open(output_path, "wb") as f:
            f.write(image_data)

        return True

    except Exception as e:
        print(f"❌ Failed to save image: {e}")
        return False


def save_all_images(
    results: List[Dict[str, Any]],
    prompts: List[str],
    output_dir="./output",
    sizes: List[tuple] = None,
):
    """Save all successful images from concurrent load test results."""
    if not results:
        print("❌ No results to save")
        return

    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        print("❌ No successful results to save")
        return

    print(f"\n💾 Saving {len(successful_results)} successful images...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    for i, result in enumerate(successful_results):
        try:
            # Get the corresponding prompt for this result
            # Note: This assumes results are in the same order as prompts
            # In a real scenario, you might want to track which prompt corresponds to which result
            prompt = prompts[i] if i < len(prompts) else f"prompt_{i + 1}"

            # Get size information if available
            size_info = ""
            if sizes and i < len(sizes):
                h, w = sizes[i]
                size_info = f"_{h}x{w}"
            elif (
                result.get("result")
                and "width" in result["result"]
                and "height" in result["result"]
            ):
                h = result["result"]["height"]
                w = result["result"]["width"]
                size_info = f"_{h}x{w}"

            # Create a safe filename from the prompt
            safe_filename = "".join(
                c for c in prompt[:50] if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            safe_filename = safe_filename.replace(" ", "_")
            filename = f"load_test_{i + 1:03d}_{safe_filename}{size_info}.jpg"

            # Save the image without opening it (for bulk saves)
            save_image_bulk(result, filename=filename, output_dir=output_dir)
            saved_count += 1

        except Exception as e:
            print(f"❌ Failed to save image {i + 1}: {e}")

    print(
        f"✅ Successfully saved {saved_count}/{len(successful_results)} images to '{output_dir}'"
    )

    # Try to open the output directory
    try:
        os.system(f"open {output_dir}")
        print("📁 Opened output directory")
    except:
        print(f"📁 Images saved in: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the Flux model using Truss API endpoint"
    )
    parser.add_argument(
        "--save-image", action="store_true", help="Save the generated image to a file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful photograph of Mt. Fuji during cherry blossom, photorealistic, high quality",
        help="Custom prompt for testing",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument(
        "--api-key",
        type=str,
        help="Baseten API key (or set BASETEN_API_KEY environment variable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for saved images (default: ./output)",
    )

    # Load testing arguments
    parser.add_argument(
        "--concurrent", action="store_true", help="Run concurrent load test"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent workers (default: 5)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="File containing prompts (one per line) for load testing",
    )
    parser.add_argument(
        "--use-varied-prompts",
        action="store_true",
        help="Use different preset prompts for load testing",
    )
    parser.add_argument(
        "--save-all-images",
        action="store_true",
        help="Save all successful images from concurrent load test (automatically enables image saving)",
    )
    parser.add_argument(
        "--variable-sizes",
        action="store_true",
        help="Use variable image sizes during load testing",
    )
    parser.add_argument(
        "--size-range",
        type=str,
        default="512-1024",
        help="Range of sizes for variable testing (format: min-max, e.g., 512-1024)",
    )

    args = parser.parse_args()

    print("🚀 Starting Flux Truss API test...")
    print("=" * 50)

    # Set API key if provided
    if args.api_key:
        os.environ["BASETEN_API_KEY"] = args.api_key

    if args.concurrent:
        # Load testing mode
        prompts = []

        if args.prompt_file:
            # Load prompts from file
            try:
                with open(args.prompt_file, "r") as f:
                    prompts = [line.strip() for line in f if line.strip()]
                print(f"📄 Loaded {len(prompts)} prompts from {args.prompt_file}")
            except FileNotFoundError:
                print(f"❌ Prompt file not found: {args.prompt_file}")
                return
        else:
            # Generate prompts based on num_requests
            if args.use_varied_prompts:
                # Use varied preset prompts
                prompts = get_varied_prompts(args.num_requests)
                print(f"🎨 Using {len(prompts)} varied prompts for load testing")
            else:
                # Use base prompt with variations
                base_prompt = args.prompt
                for i in range(args.num_requests):
                    if i == 0:
                        prompts.append(base_prompt)
                    else:
                        # Create variations of the base prompt
                        prompts.append(f"{base_prompt} (variation {i + 1})")

        # Generate variable sizes if requested
        sizes = None
        if args.variable_sizes:
            sizes = get_variable_sizes(len(prompts), args.size_range)
            print(f"📏 Generated {len(sizes)} variable sizes:")
            for i, (h, w) in enumerate(sizes[:5]):  # Show first 5
                print(f"   {i + 1}. {h}x{w}")
            if len(sizes) > 5:
                print(f"   ... and {len(sizes) - 5} more")

        # Show prompts being used (first few)
        if len(prompts) <= 5:
            print("📝 Prompts to be tested:")
            for i, prompt in enumerate(prompts):
                size_info = (
                    f" ({sizes[i][0]}x{sizes[i][1]})"
                    if sizes and i < len(sizes)
                    else ""
                )
                print(f"   {i + 1}. {prompt}{size_info}")
        else:
            print("📝 First 3 prompts to be tested:")
            for i, prompt in enumerate(prompts[:3]):
                size_info = (
                    f" ({sizes[i][0]}x{sizes[i][1]})"
                    if sizes and i < len(sizes)
                    else ""
                )
                print(f"   {i + 1}. {prompt}{size_info}")
            print(f"   ... and {len(prompts) - 3} more")

        # Run concurrent load test
        results = test_concurrent_requests(
            prompts=prompts,
            max_workers=args.max_workers,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=3.5,
            seed=42,
            sizes=sizes,
        )

        # Save images if requested
        if args.save_image or args.save_all_images:
            if args.save_all_images:
                # Save all successful images
                print("\n" + "=" * 50)
                save_all_images(results, prompts, args.output_dir, sizes)
            else:
                # Save only first successful result
                successful_results = [r for r in results if r["success"]]
                if successful_results:
                    print("\n" + "=" * 50)
                    save_image(successful_results[0], output_dir=args.output_dir)

    else:
        # Single request mode
        result = test_truss_api_call(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=3.5,
            seed=42,
        )

        if result["success"] and args.save_image:
            print("\n" + "=" * 50)
            save_image(result, output_dir=args.output_dir)

    print("\n" + "=" * 50)
    print("🎉 Truss API test completed!")


if __name__ == "__main__":
    main()
