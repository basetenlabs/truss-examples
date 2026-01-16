#!/usr/bin/env python3
"""
Image resizing proxy server for SGLang.
Intercepts requests, resizes base64 images, and forwards to SGLang.
"""

import asyncio
import base64
import os
import subprocess
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI()

# SGLang server runs on port 8001 internally
SGLANG_URL = "http://localhost:8001"

# Reusable HTTP client for better connection pooling and lower latency
_http_client: Optional[httpx.AsyncClient] = None
_sglang_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup():
    """Initialize shared HTTP clients."""
    global _http_client, _sglang_client
    _http_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=100))
    _sglang_client = httpx.AsyncClient(timeout=300.0)


@app.on_event("shutdown")
async def shutdown():
    """Close shared HTTP clients."""
    global _http_client, _sglang_client
    if _http_client:
        await _http_client.aclose()
    if _sglang_client:
        await _sglang_client.aclose()

def is_url(s: str) -> bool:
    """Check if string is an HTTP/HTTPS URL."""
    return s.startswith("http://") or s.startswith("https://")

def is_base64_data_url(s: str) -> bool:
    """Check if string is a base64 data URL."""
    return s.startswith("data:") and "base64," in s

async def fetch_image_from_url(url: str) -> Image.Image:
    """Download image from HTTP/HTTPS URL."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=100))
    response = await _http_client.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64 data URL to PIL Image."""
    # Strip "data:image/...;base64," prefix
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_data))

def encode_image_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 data URL (PNG format)."""
    buffer = BytesIO()
    # Convert to RGB if necessary (handles RGBA)
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    
    img.save(buffer, format="JPEG", quality=85)  # JPEG is smaller than PNG
    new_b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{new_b64}"

def _process_image_cpu(img: Image.Image, scale: float) -> str:
    """CPU-bound image processing (runs in thread pool)."""
    # Compute new size
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    
    # Resize using high-quality filter
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Return as base64 data URL
    return encode_image_to_base64(img_resized)


async def resize_image(image_url: str, scale: float) -> str:
    """
    Resize an image from any supported input type.
    Uses thread pool for CPU-bound operations to avoid blocking event loop.
    
    Args:
        image_url: Can be a base64 data URL or HTTP/HTTPS URL
        scale: Scale factor (e.g., 0.5 = half size)
    
    Returns:
        Base64 data URL of resized image
    """
    # Load image based on input type
    if is_base64_data_url(image_url):
        # For base64, decode and process in thread pool
        loop = asyncio.get_event_loop()
        def decode_and_process():
            img = decode_base64_image(image_url)
            return _process_image_cpu(img, scale)
        return await loop.run_in_executor(None, decode_and_process)
    elif is_url(image_url):
        # For HTTP/HTTPS URLs, fetch asynchronously first, then process in thread pool
        img = await fetch_image_from_url(image_url)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process_image_cpu, img, scale)
    else:
        # Unknown format - pass through unchanged
        return image_url


async def process_messages(messages: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    """
    Process messages and resize any images found in image_url fields.
    Handles both base64 data URLs and HTTP/HTTPS URLs.
    Processes multiple images in parallel for better latency.
    """
    processed_messages = []
    
    # Collect all image URLs that need processing
    image_tasks = []  # List of (message_idx, content_idx, url, item/message_ref)
    
    for msg_idx, message in enumerate(messages):
        if isinstance(message.get("content"), list):
            for content_idx, item in enumerate(message["content"]):
                if isinstance(item, dict) and item.get("type") == "image_url":
                    original_url = item.get("image_url", {}).get("url", "")
                    if original_url:
                        image_tasks.append((msg_idx, content_idx, original_url, item, "list"))
        elif isinstance(message.get("content"), dict):
            if message["content"].get("type") == "image_url":
                original_url = message["content"].get("image_url", {}).get("url", "")
                if original_url:
                    image_tasks.append((msg_idx, None, original_url, message, "dict"))
    
    # Process all images in parallel
    if image_tasks:
        resize_tasks = [resize_image(url, scale) for _, _, url, _, _ in image_tasks]
        try:
            resized_urls = await asyncio.gather(*resize_tasks, return_exceptions=True)
        except Exception as e:
            print(f"Warning: Error during parallel image processing: {e}")
            resized_urls = [None] * len(image_tasks)
    else:
        resized_urls = []
    
    # Reconstruct messages with resized images
    for msg_idx, message in enumerate(messages):
        processed_message = message.copy()
        
        if isinstance(message.get("content"), list):
            processed_content = []
            for content_idx, item in enumerate(message["content"]):
                if isinstance(item, dict) and item.get("type") == "image_url":
                    # Find corresponding resized URL
                    task_idx = next(
                        (i for i, (m_idx, c_idx, _, _, _) in enumerate(image_tasks)
                         if m_idx == msg_idx and c_idx == content_idx and image_tasks[i][4] == "list"),
                        None
                    )
                    if task_idx is not None:
                        resized_url = resized_urls[task_idx]
                        if isinstance(resized_url, Exception):
                            print(f"Warning: Failed to resize image: {resized_url}")
                            processed_content.append(item)
                        elif resized_url is not None:
                            processed_item = item.copy()
                            processed_item["image_url"] = {"url": resized_url}
                            processed_content.append(processed_item)
                        else:
                            processed_content.append(item)
                    else:
                        processed_content.append(item)
                else:
                    processed_content.append(item)
            processed_message["content"] = processed_content
        
        elif isinstance(message.get("content"), dict):
            if message["content"].get("type") == "image_url":
                # Find corresponding resized URL
                task_idx = next(
                    (i for i, (m_idx, _, _, _, ctx_type) in enumerate(image_tasks)
                     if m_idx == msg_idx and ctx_type == "dict"),
                    None
                )
                if task_idx is not None:
                    resized_url = resized_urls[task_idx]
                    if isinstance(resized_url, Exception):
                        print(f"Warning: Failed to resize image: {resized_url}")
                    elif resized_url is not None:
                        processed_message["content"] = {
                            "type": "image_url",
                            "image_url": {"url": resized_url}
                        }
        
        processed_messages.append(processed_message)
    
    return processed_messages


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy chat completions endpoint with image resizing.
    Extracts 'scale' parameter from request body (defaults to 0.5).
    """
    body = await request.json()
    
    # Extract scale parameter (default to 0.5)
    scale = body.pop("scale", 0.5)
    
    # Process messages to resize images
    if "messages" in body:
        body["messages"] = await process_messages(body["messages"], scale)
    
    # Forward request to SGLang using shared client
    global _sglang_client
    if _sglang_client is None:
        _sglang_client = httpx.AsyncClient(timeout=300.0)
    
    if body.get("stream", False):
        # Handle streaming response
        async with _sglang_client.stream(
            "POST",
            f"{SGLANG_URL}/v1/chat/completions",
            json=body,
        ) as response:
            async def generate():
                async for chunk in response.aiter_bytes():
                    yield chunk
            
            return StreamingResponse(
                generate(),
                media_type=response.headers.get("content-type", "text/event-stream"),
                status_code=response.status_code,
            )
    else:
        # Handle non-streaming response
        response = await _sglang_client.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=body,
        )
        return response.json()


@app.get("/health_generate")
async def health_generate():
    """Proxy health check endpoint."""
    global _sglang_client
    if _sglang_client is None:
        _sglang_client = httpx.AsyncClient(timeout=300.0)
    response = await _sglang_client.get(f"{SGLANG_URL}/health_generate")
    return response.json()


@app.get("/health")
async def health():
    """Proxy health check endpoint."""
    global _sglang_client
    if _sglang_client is None:
        _sglang_client = httpx.AsyncClient(timeout=300.0)
    response = await _sglang_client.get(f"{SGLANG_URL}/health")
    return response.json()


def wait_for_sglang():
    """Wait for SGLang server to be ready (synchronous version)."""
    print("Waiting for SGLang server to start...")
    import requests
    for _ in range(60):  # Wait up to 5 minutes
        try:
            response = requests.get(f"{SGLANG_URL}/health_generate", timeout=5.0)
            if response.status_code == 200:
                print("SGLang server is ready!")
                return
        except:
            time.sleep(5)
    
    raise RuntimeError("SGLang server failed to start")


def start_sglang_server():
    """Start SGLang server in the background."""
    gpu_count = int(
        subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode().count("\n")
    )
    
    hf_token_path = "/secrets/hf_access_token"
    hf_token = None
    if os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            hf_token = f.read().strip()
    
    # Download model if needed
    model_path = "/app/Qwen3-VL-32B-Instruct-FP8"
    if not os.path.exists(model_path):
        if hf_token:
            subprocess.run(
                [
                    "hf", "download",
                    "Qwen/Qwen3-VL-32B-Instruct-FP8",
                    "--local-dir", model_path,
                    "--token", hf_token
                ],
                check=True
            )
    
    # Start SGLang server
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--served-model-name", "Qwen/Qwen3-VL-32B-Instruct-FP8",
        "--tool-call-parser", "qwen",
        "--tp-size", str(gpu_count),
        "--ep-size", "1",
        "--context-length", "24000",
        "--max-running-requests", "8",
        "--mem-fraction-static", "0.8",
        "--host", "0.0.0.0",
        "--port", "8001",  # Internal port
    ]
    
    subprocess.Popen(cmd)


def startup_sglang():
    """Start SGLang server in a background thread."""
    start_sglang_server()
    wait_for_sglang()


if __name__ == "__main__":
    import uvicorn
    
    # Start SGLang in background thread
    sglang_thread = threading.Thread(target=startup_sglang, daemon=True)
    sglang_thread.start()
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

