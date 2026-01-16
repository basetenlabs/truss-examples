import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from PIL import Image

logger = logging.getLogger(__name__)

# Request counter for tracking
_request_counter = 0
_request_start_times = {}
_status_logger_task = None
_model_instance = None  # Global reference to model instance for status logging

# Memory tracking helper
def get_memory_usage_mb():
    """Get current memory usage in MB."""
    try:
        try:
            import psutil
        except ImportError:
            return None
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception as e:
        logger.debug(f"Failed to get memory usage: {e}")
        return None

async def periodic_status_logger():
    """Periodically log system status including active requests and memory."""
    while True:
        try:
            await asyncio.sleep(30)  # Log every 30 seconds
            
            # Get active requests
            active_requests = {}
            current_time = time.time()
            for req_id, start_time in _request_start_times.items():
                elapsed = current_time - start_time
                active_requests[req_id] = elapsed
            
            mem_usage = get_memory_usage_mb()
            
            # Check SGLang process health
            sglang_health = "unknown"
            sglang_process_status = "unknown"
            try:
                global _model_instance
                if _model_instance and _model_instance.sglang_process:
                    poll_result = _model_instance.sglang_process.poll()
                    if poll_result is None:
                        sglang_process_status = "running"
                    else:
                        sglang_process_status = f"exited_with_code_{poll_result}"
                        logger.error(f"[STATUS] SGLang process has exited with code {poll_result}!")
                    
                    # Try health endpoint
                    try:
                        health_response = httpx.get(
                            f"http://localhost:{_model_instance.sglang_port}/health_generate",
                            timeout=2.0
                        )
                        if health_response.status_code == 200:
                            sglang_health = "healthy"
                        else:
                            sglang_health = f"unhealthy_status_{health_response.status_code}"
                    except Exception as e:
                        sglang_health = f"health_check_failed_{type(e).__name__}"
            except Exception as e:
                sglang_health = f"check_error_{type(e).__name__}"
            
            num_active = len(active_requests)
            if num_active > 10:
                logger.warning(f"[STATUS] WARNING: {num_active} active requests (high load, possible queue backup)")
            
            if active_requests:
                logger.info(f"[STATUS] Active requests: {num_active}, mem={mem_usage:.1f}MB if available, sglang_process={sglang_process_status}, sglang_health={sglang_health}")
                for req_id, elapsed in sorted(active_requests.items(), key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"[STATUS]   Request #{req_id}: running for {elapsed:.2f}s")
                    if elapsed > 60:
                        logger.warning(f"[STATUS]   WARNING: Request #{req_id} has been running for {elapsed:.2f}s (>60s)")
                    if elapsed > 300:
                        logger.error(f"[STATUS]   ERROR: Request #{req_id} has been running for {elapsed:.2f}s (>5min)")
            else:
                logger.debug(f"[STATUS] No active requests, mem={mem_usage:.1f}MB if available, sglang_process={sglang_process_status}, sglang_health={sglang_health}")
        except Exception as e:
            logger.error(f"[STATUS] Error in periodic status logger: {e}\n{traceback.format_exc()}")

# Reusable HTTP client for fetching images from URLs
_http_client: Optional[httpx.AsyncClient] = None
_sglang_client: Optional[httpx.AsyncClient] = None


def is_url(s: str) -> bool:
    """Check if string is an HTTP/HTTPS URL."""
    return s.startswith("http://") or s.startswith("https://")


def is_base64_data_url(s: str) -> bool:
    """Check if string is a base64 data URL."""
    return s.startswith("data:") and "base64," in s


async def fetch_image_from_url(url: str) -> Image.Image:
    """Download image from HTTP/HTTPS URL."""
    start_time = time.time()
    logger.info(f"[fetch_image_from_url] Starting fetch from URL: {url[:100]}...")
    global _http_client
    if _http_client is None:
        logger.info("[fetch_image_from_url] Creating new HTTP client for image fetching")
        _http_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=100))
    else:
        logger.debug(f"[fetch_image_from_url] Using existing HTTP client. Pool stats: {getattr(_http_client, '_transport', {}).get('_pool', 'N/A')}")
    
    try:
        response = await _http_client.get(url)
        response.raise_for_status()
        img_size = len(response.content)
        logger.info(f"[fetch_image_from_url] Fetched {img_size / 1024:.2f} KB in {time.time() - start_time:.2f}s")
        img = Image.open(BytesIO(response.content))
        logger.debug(f"[fetch_image_from_url] Image opened: {img.size}, mode={img.mode}")
        return img
    except Exception as e:
        logger.error(f"[fetch_image_from_url] Error fetching image: {e}\n{traceback.format_exc()}")
        raise


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64 data URL to PIL Image."""
    # Strip "data:image/...;base64," prefix
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_data))


def encode_image_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 data URL (JPEG format)."""
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
    start_time = time.time()
    original_size = img.size
    logger.debug(f"[_process_image_cpu] Processing image: {original_size}, scale={scale}")
    
    # Compute new size
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    
    # Resize using high-quality filter
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    logger.debug(f"[_process_image_cpu] Resized to: {img_resized.size}")
    
    # Return as base64 data URL
    result = encode_image_to_base64(img_resized)
    elapsed = time.time() - start_time
    result_size = len(result)
    logger.info(f"[_process_image_cpu] Processed image in {elapsed:.2f}s, result size: {result_size / 1024:.2f} KB")
    
    # Explicit cleanup
    img_resized.close()
    img.close()
    
    return result


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
    start_time = time.time()
    url_type = "base64" if is_base64_data_url(image_url) else ("url" if is_url(image_url) else "unknown")
    logger.debug(f"[resize_image] Starting resize, type={url_type}, scale={scale}")
    
    try:
        # Load image based on input type
        if is_base64_data_url(image_url):
            # For base64, decode and process in thread pool
            loop = asyncio.get_event_loop()
            def decode_and_process():
                img = decode_base64_image(image_url)
                return _process_image_cpu(img, scale)
            result = await loop.run_in_executor(None, decode_and_process)
            logger.debug(f"[resize_image] Base64 resize completed in {time.time() - start_time:.2f}s")
            return result
        elif is_url(image_url):
            # For HTTP/HTTPS URLs, fetch asynchronously first, then process in thread pool
            img = await fetch_image_from_url(image_url)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _process_image_cpu, img, scale)
            logger.debug(f"[resize_image] URL resize completed in {time.time() - start_time:.2f}s")
            return result
        else:
            # Unknown format - pass through unchanged
            logger.debug(f"[resize_image] Unknown format, passing through unchanged")
            return image_url
    except Exception as e:
        logger.error(f"[resize_image] Error during resize: {e}\n{traceback.format_exc()}")
        raise


async def process_messages(messages: List[Dict[str, Any]], scale: float = 1.0) -> List[Dict[str, Any]]:
    """
    Process messages and resize any images found in image_url fields.
    Handles both base64 data URLs and HTTP/HTTPS URLs.
    Processes multiple images in parallel for better latency.
    
    If scale == 1.0, skips resizing and returns messages as-is for performance.
    """
    start_time = time.time()
    mem_before = get_memory_usage_mb()
    logger.info(f"[process_messages] Starting message processing, scale={scale}, num_messages={len(messages)}, mem={mem_before:.1f}MB")
    
    # Early exit: if scale is 1.0, no resizing needed - return messages unchanged
    if scale == 1.0:
        logger.debug("[process_messages] Scale=1.0, skipping image processing")
        return messages
    
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
    
    logger.info(f"[process_messages] Found {len(image_tasks)} images to process")
    
    # Process all images in parallel
    if image_tasks:
        resize_tasks = [resize_image(url, scale) for _, _, url, _, _ in image_tasks]
        try:
            resize_start = time.time()
            resized_urls = await asyncio.gather(*resize_tasks, return_exceptions=True)
            resize_elapsed = time.time() - resize_start
            logger.info(f"[process_messages] Parallel image processing completed in {resize_elapsed:.2f}s")
            
            # Count successes and failures
            successes = sum(1 for r in resized_urls if not isinstance(r, Exception) and r is not None)
            failures = sum(1 for r in resized_urls if isinstance(r, Exception))
            logger.info(f"[process_messages] Image processing results: {successes} success, {failures} failures")
        except Exception as e:
            logger.error(f"[process_messages] Error during parallel image processing: {e}\n{traceback.format_exc()}")
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
                            logger.warning(f"Failed to resize image: {resized_url}")
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
                        logger.warning(f"Failed to resize image: {resized_url}")
                    elif resized_url is not None:
                        processed_message["content"] = {
                            "type": "image_url",
                            "image_url": {"url": resized_url}
                        }
        
        processed_messages.append(processed_message)
    
    elapsed = time.time() - start_time
    mem_after = get_memory_usage_mb()
    mem_delta = mem_after - mem_before if mem_before and mem_after else None
    logger.info(f"[process_messages] Message processing completed in {elapsed:.2f}s, mem={mem_after:.1f}MB (delta={mem_delta:+.1f}MB if available)")
    
    return processed_messages


class Model:
    def __init__(self, **kwargs):
        global _model_instance
        _model_instance = self  # Store global reference for status logging
        
        self._config = kwargs.get("config", {})
        self._secrets = kwargs.get("secrets", {})
        self.llm_engine = None
        self.model_path = None
        self.sglang_port = 8001  # Internal port for SGLang server
        self.sglang_process = None
        logger.info("[Model.__init__] Model instance created")
        
    def get_gpu_count(self) -> int:
        """Get the number of GPUs available."""
        try:
            result = subprocess.check_output(["nvidia-smi", "--list-gpus"], stderr=subprocess.DEVNULL)
            return len(result.decode().strip().split("\n"))
        except:
            return 1
    
    def load(self):
        """Load the SGLang runtime server."""
        load_start = time.time()
        mem_before = get_memory_usage_mb()
        logger.info(f"[load] Starting model load - mem={mem_before:.1f}MB")
        
        # Set up HuggingFace token if available
        hf_token = self._secrets.get("hf_access_token")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            logger.debug("[load] HuggingFace token set from secrets")
        else:
            logger.debug("[load] No HuggingFace token found in secrets")
        
        # Determine model path
        model_path = "/app/Qwen3-VL-32B-Instruct-FP8"
        if not os.path.exists(model_path):
            logger.info(f"[load] Model path {model_path} does not exist")
            # Try to download if we have a token
            if hf_token:
                logger.info("[load] Downloading model...")
                download_start = time.time()
                subprocess.run(
                    [
                        "hf", "download",
                        "Qwen/Qwen3-VL-32B-Instruct-FP8",
                        "--local-dir", model_path,
                        "--token", hf_token
                    ],
                    check=True
                )
                download_elapsed = time.time() - download_start
                logger.info(f"[load] Model download completed in {download_elapsed:.2f}s")
            else:
                # Use HuggingFace model ID directly
                model_path = "Qwen/Qwen3-VL-32B-Instruct-FP8"
                logger.info(f"[load] Using HuggingFace model ID: {model_path}")
        else:
            logger.info(f"[load] Using existing model path: {model_path}")
        
        self.model_path = model_path
        
        # Get GPU count for tensor parallelism
        gpu_count = self.get_gpu_count()
        logger.info(f"[load] Loading SGLang runtime with {gpu_count} GPUs...")
        
        # Start SGLang server using subprocess (like the original config)
        # This matches the approach used in the working baseline config
        # Use 127.0.0.1 for internal server binding (avoids address assignment issues)
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
            "--host", "127.0.0.1",
            "--port", str(self.sglang_port),
        ]

        # self.llm = sgl.Engine(model_path=model_path, tp_size=gpu_count, ep_size=1, context_length=24000, max_running_requests=8, mem_fraction_static=0.8, host="127.0.0.1", port=self.sglang_port)
        

        logger.info(f"[load] Starting SGLang server with command: {' '.join(cmd)}")
        server_start = time.time()
        self.sglang_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        logger.info(f"[load] SGLang subprocess started, PID={self.sglang_process.pid}")
        
        # Wait for 10 seconds and check if command fails (like vllm pattern)
        logger.debug("[load] Waiting 10 seconds for SGLang process to initialize...")
        time.sleep(10)
        
        if self.sglang_process.poll() is None:
            logger.info("[load] SGLang process is running (poll returned None)")
        else:
            stdout, stderr = self.sglang_process.communicate()
            if self.sglang_process.returncode != 0:
                logger.error(f"[load] SGLang process failed with return code {self.sglang_process.returncode}")
                logger.error(f"[load] STDOUT: {stdout[:1000] if stdout else 'None'}")
                logger.error(f"[load] STDERR: {stderr[:1000] if stderr else 'None'}")
                raise RuntimeError(
                    f"Command failed with code {self.sglang_process.returncode}: {stderr}"
                )
        
        # Wait for server to be ready (polling health endpoint)
        logger.info("[load] Waiting for SGLang server to be ready...")
        wait_start = time.time()
        self._wait_for_sglang()
        wait_elapsed = time.time() - wait_start
        logger.info(f"[load] SGLang server ready after {wait_elapsed:.2f}s")
        
        # Initialize HTTP client for SGLang
        # Use longer timeout for generation (30 min) and connection pool limits
        global _sglang_client
        logger.info("[load] Initializing HTTP client for SGLang...")
        _sglang_client = httpx.AsyncClient(
            timeout=httpx.Timeout(1800.0, connect=30.0),  # 30 min read, 30s connect
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        logger.info("[load] HTTP client initialized")
        
        # Start periodic status logger
        global _status_logger_task
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _status_logger_task = asyncio.create_task(periodic_status_logger())
                logger.info("[load] Started periodic status logger")
            else:
                logger.warning("[load] Event loop not running, cannot start status logger")
        except Exception as e:
            logger.warning(f"[load] Failed to start status logger: {e}")
        
        total_load_time = time.time() - load_start
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before if mem_before and mem_after else None
        logger.info(f"[load] SGLang runtime loaded successfully in {total_load_time:.2f}s! mem={mem_after:.1f}MB (delta={mem_delta:+.1f}MB if available)")
    
    def _wait_for_sglang(self):
        """Wait for SGLang server to be ready (following vllm/ultravox pattern)."""
        MAX_FAILED_SECONDS = 600  # 10 minutes
        sglang_url = f"http://localhost:{self.sglang_port}"
        logger.info("[_wait_for_sglang] Waiting for SGLang server to start...")
        
        server_up = False
        start_time = time.time()
        attempt_count = 0
        while time.time() - start_time < MAX_FAILED_SECONDS:
            attempt_count += 1
            try:
                health_start = time.time()
                response = httpx.get(f"{sglang_url}/health_generate", timeout=5.0)
                health_elapsed = time.time() - health_start
                if response.status_code == 200:
                    server_up = True
                    total_wait = time.time() - start_time
                    logger.info(f"[_wait_for_sglang] SGLang server is ready! (attempt {attempt_count}, total wait: {total_wait:.2f}s, last health check: {health_elapsed:.3f}s)")
                    break
                else:
                    logger.debug(f"[_wait_for_sglang] Health check returned {response.status_code} (attempt {attempt_count})")
            except httpx.RequestError as e:
                seconds_passed = int(time.time() - start_time)
                if seconds_passed % 10 == 0:
                    logger.info(f"[_wait_for_sglang] Server is starting for {seconds_passed} seconds (attempt {attempt_count}): {e}")
                time.sleep(1)  # Wait for 1 second before retrying
            except Exception as e:
                seconds_passed = int(time.time() - start_time)
                logger.debug(f"[_wait_for_sglang] Unexpected error during health check (attempt {attempt_count}): {e}")
                time.sleep(1)
        
        if not server_up:
            total_wait = time.time() - start_time
            logger.error(f"[_wait_for_sglang] SGLang server failed to start after {total_wait:.2f}s ({attempt_count} attempts)")
            raise RuntimeError(
                "SGLang server failed to start within the maximum allowed time."
            )
    
    async def chat_completions(self, request: Request) -> Response:
        """
        OpenAI-compatible chat completions endpoint.
        Receives request, sends it to predict, gets response and streams it back.
        """
        global _request_counter
        _request_counter += 1
        request_id = _request_counter
        start_time = time.time()
        mem_before = get_memory_usage_mb()
        
        logger.info(f"[chat_completions] REQUEST #{request_id} STARTED - mem={mem_before:.1f}MB")
        _request_start_times[request_id] = start_time
        
        # Parse request body
        try:
            parse_start = time.time()
            body = await request.json()
            parse_elapsed = time.time() - parse_start
            logger.debug(f"[chat_completions] REQUEST #{request_id} - Parsed JSON in {parse_elapsed:.3f}s")
        except Exception as e:
            logger.error(f"[chat_completions] REQUEST #{request_id} - Error parsing request body: {e}\n{traceback.format_exc()}")
            return Response(
                content=json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400,
                media_type="application/json"
            )
        
        # Call predict with the request body
        try:
            predict_start = time.time()
            result = await self.predict(body)
            predict_elapsed = time.time() - predict_start
            logger.info(f"[chat_completions] REQUEST #{request_id} - predict() completed in {predict_elapsed:.2f}s")
            
            # Check if result is a generator (streaming response)
            # When stream=True, predict() returns an async generator
            if hasattr(result, '__aiter__'):
                # Streaming response - iterate over the generator and yield chunks
                logger.info(f"[chat_completions] REQUEST #{request_id} - Starting streaming response")
                chunk_count = 0
                stream_start = time.time()
                
                async def generate():
                    nonlocal chunk_count
                    try:
                        async for chunk in result:
                            chunk_count += 1
                            if chunk_count % 10 == 0:
                                logger.debug(f"[chat_completions] REQUEST #{request_id} - Streamed {chunk_count} chunks")
                            yield chunk
                        stream_elapsed = time.time() - stream_start
                        logger.info(f"[chat_completions] REQUEST #{request_id} - Streaming completed: {chunk_count} chunks in {stream_elapsed:.2f}s")
                    except Exception as e:
                        logger.error(f"[chat_completions] REQUEST #{request_id} - Error during streaming: {e}\n{traceback.format_exc()}")
                        raise
                    finally:
                        total_elapsed = time.time() - start_time
                        mem_after = get_memory_usage_mb()
                        mem_delta = mem_after - mem_before if mem_before and mem_after else None
                        logger.info(f"[chat_completions] REQUEST #{request_id} COMPLETED - Total: {total_elapsed:.2f}s, mem={mem_after:.1f}MB (delta={mem_delta:+.1f}MB if available)")
                        _request_start_times.pop(request_id, None)
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response - return JSON
                total_elapsed = time.time() - start_time
                mem_after = get_memory_usage_mb()
                mem_delta = mem_after - mem_before if mem_before and mem_after else None
                logger.info(f"[chat_completions] REQUEST #{request_id} COMPLETED - Total: {total_elapsed:.2f}s, mem={mem_after:.1f}MB (delta={mem_delta:+.1f}MB if available)")
                _request_start_times.pop(request_id, None)
                
                return Response(
                    content=json.dumps(result),
                    media_type="application/json"
                )
        except Exception as e:
            total_elapsed = time.time() - start_time
            mem_after = get_memory_usage_mb()
            logger.error(f"[chat_completions] REQUEST #{request_id} FAILED after {total_elapsed:.2f}s - Error: {e}\n{traceback.format_exc()}\nmem={mem_after:.1f}MB")
            _request_start_times.pop(request_id, None)
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )

    async def predict(self, model_input: Dict[str, Any]) -> Union[Dict[str, Any], AsyncIterator[bytes]]:
        """
        Predict function that processes images and calls SGLang.
        
        Expected input format (OpenAI-compatible):
        {
            "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
            "messages": [...],
            "scale": 1.0,  # Optional, defaults to 1.0 (no resizing)
            "max_tokens": 4096,
            "temperature": 0.6,
            "stream": False,
            ...
        }
        """
        predict_start = time.time()
        mem_before = get_memory_usage_mb()
        logger.info(f"[predict] Starting prediction - mem={mem_before:.1f}MB")
        
        # Extract scale parameter (default to 1.0 - no resizing)
        scale = model_input.pop("scale", 1.0)
        logger.debug(f"[predict] Scale parameter: {scale}")
        
        # Process messages to resize images
        messages = model_input.get("messages", [])
        num_messages = len(messages)
        if messages:
            msg_process_start = time.time()
            messages = await process_messages(messages, scale)
            msg_process_elapsed = time.time() - msg_process_start
            logger.info(f"[predict] Processed {num_messages} messages in {msg_process_elapsed:.2f}s")
            model_input["messages"] = messages
        
        # Prepare request body for SGLang (OpenAI-compatible format)
        request_body = {
            "model": model_input.get("model", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
            "messages": messages,
            "max_tokens": model_input.get("max_tokens", 4096),
            "temperature": model_input.get("temperature", 0.6),
        }
        
        # Add optional parameters
        if "top_p" in model_input:
            request_body["top_p"] = model_input["top_p"]
        if "top_k" in model_input:
            request_body["top_k"] = model_input["top_k"]
        if "stop" in model_input:
            request_body["stop"] = model_input["stop"]
        
        # Add tool-related parameters (required for tool calling)
        if "tools" in model_input:
            request_body["tools"] = model_input["tools"]
        if "tool_choice" in model_input:
            request_body["tool_choice"] = model_input["tool_choice"]
        # Default to false to limit to max 1 tool call per response
        request_body["parallel_tool_calls"] = model_input.get("parallel_tool_calls", False)
        
        stream = model_input.get("stream", False)
        request_body["stream"] = stream
        
        logger.info(f"[predict] Request body prepared: max_tokens={request_body['max_tokens']}, stream={stream}, num_messages={len(messages)}")
        
        # Call SGLang server via HTTP
        global _sglang_client
        if _sglang_client is None:
            logger.warning("[predict] SGLang client is None, creating new client")
            _sglang_client = httpx.AsyncClient(
                timeout=httpx.Timeout(1800.0, connect=30.0),  # 30 min read, 30s connect
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
        else:
            # Log connection pool status if available
            try:
                transport = getattr(_sglang_client, '_transport', None)
                if transport:
                    pool = getattr(transport, '_pool', None)
                    if pool:
                        logger.debug(f"[predict] SGLang client pool: {pool}")
            except:
                pass
        
        sglang_url = f"http://localhost:{self.sglang_port}"
        
        # Check SGLang server health before making request
        try:
            health_start = time.time()
            health_response = httpx.get(f"{sglang_url}/health_generate", timeout=5.0)
            health_elapsed = time.time() - health_start
            if health_response.status_code == 200:
                logger.debug(f"[predict] SGLang health check OK in {health_elapsed:.3f}s")
            else:
                logger.warning(f"[predict] SGLang health check returned {health_response.status_code} in {health_elapsed:.3f}s")
        except Exception as e:
            logger.warning(f"[predict] SGLang health check failed: {e}")
        
        try:
            if stream:
                # Handle streaming response
                logger.info("[predict] Starting streaming request to SGLang")
                sglang_request_start = time.time()
                
                # Create a generator that manages its own context to prevent connection leaks
                async def generate():
                    stream_ctx = None
                    try:
                        # Enter the context manager and keep it alive while generating
                        stream_ctx = _sglang_client.stream(
                            "POST",
                            f"{sglang_url}/v1/chat/completions",
                            json=request_body,
                        )
                        logger.debug("[predict] Stream context created, entering...")
                        response = await stream_ctx.__aenter__()
                        logger.debug(f"[predict] Stream response received, status={response.status_code}")
                        
                        chunk_count = 0
                        first_chunk_time = None
                        try:
                            async for chunk in response.aiter_bytes():
                                if first_chunk_time is None:
                                    first_chunk_time = time.time()
                                    time_to_first_chunk = first_chunk_time - sglang_request_start
                                    logger.info(f"[predict] First chunk received in {time_to_first_chunk:.2f}s")
                                chunk_count += 1
                                if chunk_count % 50 == 0:
                                    logger.debug(f"[predict] Streamed {chunk_count} chunks so far")
                                yield chunk
                            
                            total_stream_time = time.time() - sglang_request_start
                            logger.info(f"[predict] Streaming completed: {chunk_count} chunks in {total_stream_time:.2f}s")
                        finally:
                            # Ensure context is properly exited when generator is exhausted
                            if stream_ctx:
                                logger.debug("[predict] Exiting stream context...")
                                await stream_ctx.__aexit__(None, None, None)
                                logger.debug("[predict] Stream context exited")
                    except Exception as e:
                        logger.error(f"[predict] Error in streaming generator: {e}\n{traceback.format_exc()}")
                        if stream_ctx:
                            try:
                                await stream_ctx.__aexit__(type(e), e, e.__traceback__)
                            except:
                                pass
                        raise
                
                return generate()
            else:
                # Handle non-streaming response
                logger.info("[predict] Starting non-streaming request to SGLang")
                sglang_request_start = time.time()
                response = await _sglang_client.post(
                    f"{sglang_url}/v1/chat/completions",
                    json=request_body,
                )
                sglang_request_elapsed = time.time() - sglang_request_start
                logger.info(f"[predict] SGLang non-streaming response received in {sglang_request_elapsed:.2f}s, status={response.status_code}")
                
                result = response.json()
                total_predict_elapsed = time.time() - predict_start
                mem_after = get_memory_usage_mb()
                mem_delta = mem_after - mem_before if mem_before and mem_after else None
                logger.info(f"[predict] Prediction completed in {total_predict_elapsed:.2f}s, mem={mem_after:.1f}MB (delta={mem_delta:+.1f}MB if available)")
                return result
        except httpx.ReadTimeout as e:
            elapsed = time.time() - predict_start
            logger.error(f"[predict] ReadTimeout during generation after {elapsed:.2f}s (request took >30 min): {e}\n{traceback.format_exc()}")
            raise RuntimeError(
                "Request to SGLang server timed out. The generation may be taking too long. "
                "Consider reducing max_tokens or checking SGLang server status."
            ) from e
        except httpx.ConnectTimeout as e:
            elapsed = time.time() - predict_start
            logger.error(f"[predict] ConnectTimeout to SGLang server after {elapsed:.2f}s: {e}\n{traceback.format_exc()}")
            raise RuntimeError(
                "Failed to connect to SGLang server. The server may be overloaded or unavailable."
            ) from e
        except httpx.HTTPError as e:
            elapsed = time.time() - predict_start
            logger.error(f"[predict] HTTP error during generation after {elapsed:.2f}s: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"HTTP error communicating with SGLang server: {e}") from e
        except Exception as e:
            elapsed = time.time() - predict_start
            logger.error(f"[predict] Error during generation after {elapsed:.2f}s: {e}\n{traceback.format_exc()}")
            raise

