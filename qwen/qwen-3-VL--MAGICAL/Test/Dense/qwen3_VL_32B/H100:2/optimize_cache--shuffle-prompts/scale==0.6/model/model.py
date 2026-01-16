import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import time
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from PIL import Image

logger = logging.getLogger(__name__)

# Pattern to match "Screenshot size: NxN pixels" text
_SCREENSHOT_SIZE_PATTERN = re.compile(r"^Screenshot size:\s*\d+x\d+\s*pixels$", re.IGNORECASE)

# Pattern to extract "Context for Today:" section from end of system prompt
_CONTEXT_FOR_TODAY_PATTERN = re.compile(r"\n\nContext for Today:\n.*$", re.DOTALL)

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


def _remove_screenshot_size_text(content: List[Dict[str, Any]]) -> None:
    """Remove 'Screenshot size: NxN pixels' text item from content list in-place."""
    for i, item in enumerate(content):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if _SCREENSHOT_SIZE_PATTERN.match(text.strip()):
                del content[i]
                return  # Only one occurrence


def _optimize_for_prefix_cache(messages: List[Dict[str, Any]]) -> None:
    """
    Optimize messages for prefix caching by:
    1. Removing system messages containing "<!-- NONCE:" (useless)
    2. Extracting "Context for Today:" from system prompt and moving it to the 
       content list with the image (so system prompt becomes static/cacheable)
    """
    extracted_context = None
    
    # First pass: remove NONCE messages and extract context from system prompts
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                # Remove NONCE system messages
                if "<!-- NONCE:" in content:
                    del messages[i]
                    continue
                # Extract and remove "Context for Today:" section
                match = _CONTEXT_FOR_TODAY_PATTERN.search(content)
                if match:
                    extracted_context = match.group(0).lstrip("\n")  # Keep "Context for Today:..." 
                    msg["content"] = content[:match.start()]
        i += 1
    
    # Second pass: insert extracted context into the content list with the image
    if extracted_context:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    # Find if this content has an image_url
                    has_image = any(
                        isinstance(item, dict) and item.get("type") == "image_url"
                        for item in content
                    )
                    if has_image:
                        # Insert context at the beginning (before image)
                        content.insert(0, {"type": "text", "text": extracted_context})
                        break


async def process_messages(messages: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    """
    Process messages and resize any images found in image_url fields.
    Handles both base64 data URLs and HTTP/HTTPS URLs.
    Processes multiple images in parallel for better latency.
    Also filters out 'Screenshot size: NxN pixels' text items.
    
    If scale == 1.0, skips resizing but still filters screenshot size text items.
    """
    # Optimize for prefix caching: remove NONCE, move dynamic context to end
    _optimize_for_prefix_cache(messages)
    
    # Remove screenshot size text from messages with list content
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            _remove_screenshot_size_text(content)
    
    # Early exit: if scale is 1.0, no resizing needed
    if scale == 1.0:
        return messages
    
    processed_messages = []
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
            logger.warning(f"Error during parallel image processing: {e}")
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
    
    return processed_messages


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs.get("config", {})
        self._secrets = kwargs.get("secrets", {})
        self.llm_engine = None
        self.model_path = None
        self.sglang_port = 8001  # Internal port for SGLang server
        self.sglang_process = None
        # Default scale is 1.0 (no resizing). Can be overridden per-request.
        self.default_scale = 0.6
        
    def get_gpu_count(self) -> int:
        """Get the number of GPUs available."""
        try:
            result = subprocess.check_output(["nvidia-smi", "--list-gpus"], stderr=subprocess.DEVNULL)
            return len(result.decode().strip().split("\n"))
        except:
            return 1
    
    def load(self):
        """Load the SGLang runtime server."""
        # Set up HuggingFace token if available
        hf_token = self._secrets.get("hf_access_token")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # Determine model path
        model_path = "/app/Qwen3-VL-32B-Instruct-FP8"
        if not os.path.exists(model_path):
            # Try to download if we have a token
            if hf_token:
                logger.info("Downloading model...")
                subprocess.run(
                    [
                        "hf", "download",
                        "Qwen/Qwen3-VL-32B-Instruct-FP8",
                        "--local-dir", model_path,
                        "--token", hf_token
                    ],
                    check=True
                )
            else:
                # Use HuggingFace model ID directly
                model_path = "Qwen/Qwen3-VL-32B-Instruct-FP8"
        
        self.model_path = model_path
        
        # Get GPU count for tensor parallelism
        gpu_count = self.get_gpu_count()
        logger.info(f"Loading SGLang runtime with {gpu_count} GPUs...")
        
        # Start SGLang server using subprocess (like the original config)
        # This matches the approach used in the working baseline config
        # Use 127.0.0.1 for internal server binding (avoids address assignment issues)
        os.environ["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"
        os.environ["SGLANG_VLM_CACHE_SIZE_MB"] = "0"

        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--served-model-name", "Qwen/Qwen3-VL-32B-Instruct-FP8",
            "--tool-call-parser", "qwen",
            "--tp-size", str(gpu_count),
            "--ep-size", "1",
            "--context-length", "24000",
            "--max-running-requests", "8",
            "--mem-fraction-static", "0.65",
            "--keep-mm-feature-on-device",
            "--host", "127.0.0.1",
            "--port", str(self.sglang_port),
        ]

        # self.llm = sgl.Engine(model_path=model_path, tp_size=gpu_count, ep_size=1, context_length=24000, max_running_requests=8, mem_fraction_static=0.8, host="127.0.0.1", port=self.sglang_port)
        

        logger.info(f"Starting SGLang server with command: {' '.join(cmd)}")
        self.sglang_process = subprocess.Popen(
            cmd, stdout=None, stderr=None, text=True
        )
        
        # Wait for 10 seconds and check if command fails (like vllm pattern)
        time.sleep(10)
        
        if self.sglang_process.poll() is None:
            logger.info("Command to start SGLang server ran successfully")
        else:
            stdout, stderr = self.sglang_process.communicate()
            if self.sglang_process.returncode != 0:
                logger.error(f"Command failed with error: {stderr}")
                raise RuntimeError(
                    f"Command failed with code {self.sglang_process.returncode}: {stderr}"
                )
        
        # Wait for server to be ready (polling health endpoint)
        self._wait_for_sglang()
        
        # Initialize HTTP client for SGLang
        # Use longer timeout for generation (30 min) and connection pool limits
        global _sglang_client
        _sglang_client = httpx.AsyncClient(
            timeout=httpx.Timeout(1800.0, connect=30.0),  # 30 min read, 30s connect
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        
        logger.info("SGLang runtime loaded successfully!")
    
    def _wait_for_sglang(self):
        """Wait for SGLang server to be ready (following vllm/ultravox pattern)."""
        MAX_FAILED_SECONDS = 600  # 10 minutes
        sglang_url = f"http://localhost:{self.sglang_port}"
        logger.info("Waiting for SGLang server to start...")
        
        server_up = False
        start_time = time.time()
        while time.time() - start_time < MAX_FAILED_SECONDS:
            try:
                response = httpx.get(f"{sglang_url}/health_generate", timeout=5.0)
                if response.status_code == 200:
                    server_up = True
                    logger.info("SGLang server is ready!")
                    break
            except httpx.RequestError as e:
                seconds_passed = int(time.time() - start_time)
                if seconds_passed % 10 == 0:
                    logger.info(f"Server is starting for {seconds_passed} seconds: {e}")
                time.sleep(1)  # Wait for 1 second before retrying
        
        if not server_up:
            raise RuntimeError(
                "SGLang server failed to start within the maximum allowed time."
            )
    
    async def chat_completions(self, request: Request) -> Response:
        """
        OpenAI-compatible chat completions endpoint.
        Receives request, sends it to predict, gets response and streams it back.
        """
        request_start = time.perf_counter()
        
        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Error parsing request body: {e}")
            return Response(
                content=json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400,
                media_type="application/json"
            )
        
        # Call predict with the request body
        try:
            result = await self.predict(body)
            
            # Check if result is a generator (streaming response)
            # When stream=True, predict() returns an async generator
            if hasattr(result, '__aiter__'):
                # Streaming response - iterate over the generator and yield chunks
                async def generate():
                    async for chunk in result:
                        yield chunk
                    total_e2e = time.perf_counter() - request_start
                    logger.info(f"Total request (streaming) took {total_e2e*1000:.2f}ms")
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response - return JSON
                total_e2e = time.perf_counter() - request_start
                logger.info(f"Total request took {total_e2e*1000:.2f}ms")
                return Response(
                    content=json.dumps(result),
                    media_type="application/json"
                )
        except Exception as e:
            logger.error(f"Error in predict: {e}")
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
            "scale": 1.0,  # Optional, defaults to config value (or 1.0 if not in config)
            "max_tokens": 4096,
            "temperature": 0.6,
            "stream": False,
            ...
        }
        """
        # === PREPROCESSING ===
        preprocess_start = time.perf_counter()
        
        # Extract scale parameter (defaults to config value, or 1.0 if not in config)
        scale = model_input.pop("scale", self.default_scale)
        
        # Process messages to resize images
        messages = model_input.get("messages", [])
        if messages:
            messages = await process_messages(messages, scale)
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
        
        preprocess_time = time.perf_counter() - preprocess_start
        logger.info(f"Preprocessing took {preprocess_time*1000:.2f}ms")
        
        # === MODEL INFERENCE ===
        global _sglang_client
        if _sglang_client is None:
            _sglang_client = httpx.AsyncClient(
                timeout=httpx.Timeout(1800.0, connect=30.0),  # 30 min read, 30s connect
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
        
        sglang_url = f"http://localhost:{self.sglang_port}"
        
        try:
            if stream:
                # Handle streaming response
                # Create a generator that manages its own context to prevent connection leaks
                async def generate():
                    inference_start = time.perf_counter()
                    first_token_time = None
                    
                    # Enter the context manager and keep it alive while generating
                    stream_ctx = _sglang_client.stream(
                        "POST",
                        f"{sglang_url}/v1/chat/completions",
                        json=request_body,
                    )
                    response = await stream_ctx.__aenter__()
                    try:
                        async for chunk in response.aiter_bytes():
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                ttft = first_token_time - inference_start
                                logger.info(f"Time to first token: {ttft*1000:.2f}ms")
                            yield chunk
                    finally:
                        # Ensure context is properly exited when generator is exhausted
                        await stream_ctx.__aexit__(None, None, None)
                        total_inference = time.perf_counter() - inference_start
                        logger.info(f"Total inference (streaming) took {total_inference*1000:.2f}ms")
                
                return generate()
            else:
                # Handle non-streaming response
                inference_start = time.perf_counter()
                response = await _sglang_client.post(
                    f"{sglang_url}/v1/chat/completions",
                    json=request_body,
                )
                inference_time = time.perf_counter() - inference_start
                logger.info(f"Model inference took {inference_time*1000:.2f}ms")
                
                return response.json()
        except httpx.ReadTimeout as e:
            logger.error(f"ReadTimeout during generation (request took >30 min): {e}")
            raise RuntimeError(
                "Request to SGLang server timed out. The generation may be taking too long. "
                "Consider reducing max_tokens or checking SGLang server status."
            ) from e
        except httpx.ConnectTimeout as e:
            logger.error(f"ConnectTimeout to SGLang server: {e}")
            raise RuntimeError(
                "Failed to connect to SGLang server. The server may be overloaded or unavailable."
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during generation: {e}")
            raise RuntimeError(f"HTTP error communicating with SGLang server: {e}") from e
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

