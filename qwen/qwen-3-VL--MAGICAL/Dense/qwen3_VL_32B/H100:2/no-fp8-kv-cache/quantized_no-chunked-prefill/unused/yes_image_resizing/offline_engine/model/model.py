import asyncio
import base64
import json
import logging
import os
import re
import signal
import subprocess
import time
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx
import sglang as sgl
from sglang.utils import async_stream_and_merge
from sglang.srt.parser.conversation import chat_templates
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Reusable HTTP client for fetching images from URLs
_http_client: Optional[httpx.AsyncClient] = None


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


async def process_messages(messages: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    """
    Process messages and resize any images found in image_url fields.
    Handles both base64 data URLs and HTTP/HTTPS URLs.
    Processes multiple images in parallel for better latency.
    
    If scale == 1.0, skips resizing and returns messages as-is for performance.
    """
    # Early exit: if scale is 1.0, no resizing needed - return messages unchanged
    if scale == 1.0:
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


def _convert_messages_to_prompt_and_extract_images(
    messages: List[Dict[str, Any]], 
    tokenizer: AutoTokenizer,
    chat_template_name: str = "qwen"
) -> Tuple[str, Optional[List[str]]]:
    """
    Convert OpenAI messages format to SGLang prompt format and extract images.
    For VLM models, images must be passed separately via image_data parameter.
    
    Returns:
        tuple: (prompt_string, image_data_list)
    """
    # Get image_token from chat template
    try:
        conv = chat_templates[chat_template_name].copy()
        image_token = conv.image_token
    except (KeyError, AttributeError):
        # Fallback: try to get from tokenizer's chat template
        # For Qwen3-VL, the image token is typically "<image>"
        image_token = "<image>"
        logger.warning(f"Could not get image_token from chat template '{chat_template_name}', using default: {image_token}")
    
    # Extract images and replace image_url with image_token placeholder
    processed_messages = []
    image_data_list = []
    
    for msg in messages:
        processed_msg = msg.copy()
        
        if isinstance(msg.get("content"), list):
            # Handle list content (multimodal)
            new_content = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    # Replace image_url with image_token placeholder
                    new_content.append(image_token)
                    # Extract image URL/data
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        image_data_list.append(image_url)
                else:
                    # Keep text and other content types as-is
                    new_content.append(item)
            processed_msg["content"] = new_content
        elif isinstance(msg.get("content"), dict) and msg["content"].get("type") == "image_url":
            # Handle single image_url in content dict
            processed_msg["content"] = image_token
            image_url = msg["content"].get("image_url", {}).get("url", "")
            if image_url:
                image_data_list.append(image_url)
        
        processed_messages.append(processed_msg)
    
    # Apply chat template to processed messages (with image_token placeholders)
    prompt = tokenizer.apply_chat_template(
        processed_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Return prompt and image data list
    return prompt, image_data_list if image_data_list else None


def _convert_to_openai_format(
    sglang_result: Any,
    model_input: Dict[str, Any],
    tools: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Convert SGLang generation result to OpenAI-compatible format.
    sgl.Engine.generate() returns list of results: [{"text": "...", "meta_info": {...}}]
    """
    # Extract text from result
    if isinstance(sglang_result, dict):
        text = sglang_result.get("text", "")
        meta_info = sglang_result.get("meta_info", {})
    else:
        text = str(sglang_result)
        meta_info = {}
    
    # Build OpenAI response format
    response = {
        "id": f"chatcmpl-{os.urandom(16).hex()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_input.get("model", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop"  # May need to detect actual reason from meta_info
        }],
        "usage": {
            "prompt_tokens": meta_info.get("prompt_tokens", 0),
            "completion_tokens": meta_info.get("completion_tokens", 0),
            "total_tokens": meta_info.get("total_tokens", 0)
        }
    }
    
    # Handle tool calls if present in result
    # SGLang with tool-call-parser="qwen" returns tool calls in meta_info
    tool_calls = None
    
    # First, check meta_info for parsed tool calls
    if meta_info and "tool_calls" in meta_info:
        sglang_tool_calls = meta_info.get("tool_calls", [])
        if sglang_tool_calls:
            # Parse tool calls from SGLang format to OpenAI format
            tool_calls = []
            for idx, tool_call in enumerate(sglang_tool_calls):
                # SGLang tool call format may vary, try common formats
                if isinstance(tool_call, dict):
                    # Expected format: {"function": {"name": "...", "arguments": "..."}, "id": "..."}
                    function = tool_call.get("function", {})
                    if isinstance(function, dict):
                        # Ensure arguments is a JSON string (OpenAI requirement)
                        arguments = function.get("arguments", "")
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments)
                        elif not isinstance(arguments, str):
                            arguments = str(arguments)
                        
                        tool_calls.append({
                            "id": tool_call.get("id", f"call_{os.urandom(8).hex()}"),
                            "type": "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": arguments  # Must be JSON string
                            },
                            "index": idx  # Index for ordering (OpenAI format)
                        })
                    else:
                        # Alternative format: {"name": "...", "arguments": "...", "id": "..."}
                        # Ensure arguments is a JSON string (OpenAI requirement)
                        arguments = tool_call.get("arguments", "")
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments)
                        elif not isinstance(arguments, str):
                            arguments = str(arguments)
                        
                        tool_calls.append({
                            "id": tool_call.get("id", f"call_{os.urandom(8).hex()}"),
                            "type": "function",
                            "function": {
                                "name": tool_call.get("name", ""),
                                "arguments": arguments  # Must be JSON string
                            },
                            "index": idx  # Index for ordering (OpenAI format)
                        })
    
    # If tools were provided but no tool calls found in meta_info,
    # check if tool calls might be embedded in the text (fallback)
    if not tool_calls and tools and text:
        # SGLang might return tool calls as JSON in text
        # Try to parse tool calls from text (Qwen format)
        try:
            # Look for tool call patterns in text
            # Qwen tool call format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
            tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
            matches = re.findall(tool_call_pattern, text, re.DOTALL)
            if matches:
                tool_calls = []
                for idx, match in enumerate(matches):
                    try:
                        tool_data = json.loads(match)
                        tool_calls.append({
                            "id": f"call_{os.urandom(8).hex()}",
                            "type": "function",
                            "function": {
                                "name": tool_data.get("name", ""),
                                "arguments": json.dumps(tool_data.get("arguments", {})) if isinstance(tool_data.get("arguments"), dict) else str(tool_data.get("arguments", ""))
                            },
                            "index": idx  # Index for ordering (OpenAI format)
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool call JSON: {match}")
                        continue
        except Exception as e:
            logger.debug(f"Error parsing tool calls from text: {e}")
    
    # If tool calls found, add them to message and update finish_reason
    if tool_calls:
        response["choices"][0]["message"]["tool_calls"] = tool_calls
        response["choices"][0]["finish_reason"] = "tool_calls"
        # Tool calls may have content as well (e.g., reasoning before tool call)
        # Keep the text content if present, but set to None if it's just tool call markers
        if text and not any(marker in text for marker in ["<tool_call>", "</tool_call>"]):
            # Text content is valid (not just tool call markers)
            pass
        else:
            # Text is just tool call markers, set content to None
            response["choices"][0]["message"]["content"] = None
    
    return response


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs.get("config", {})
        self._secrets = kwargs.get("secrets", {})
        self.llm_engine = None
        self.tokenizer = None
        self.model_path = None
        
    def get_gpu_count(self) -> int:
        """Get the number of GPUs available."""
        try:
            result = subprocess.check_output(["nvidia-smi", "--list-gpus"], stderr=subprocess.DEVNULL)
            return len(result.decode().strip().split("\n"))
        except:
            return 1
    
    def load(self):
        """Load the SGLang engine directly."""
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
        logger.info(f"Loading SGLang engine with {gpu_count} GPUs...")
        
        # Workaround: Monkey-patch signal.signal to handle non-main thread initialization
        # sgl.Engine tries to set signal handlers which only work in main thread
        # In Truss, load() is called from a worker thread, so we need to catch this gracefully
        original_signal = signal.signal
        
        def patched_signal(signalnum, handler):
            """Patched signal handler that gracefully handles non-main thread calls."""
            try:
                return original_signal(signalnum, handler)
            except ValueError as e:
                # Signal handler can only be set in main thread - this is expected in Truss
                # Log a debug message but don't fail - the engine will still work without signal handlers
                logger.debug(f"Could not set signal handler for {signalnum} (not in main thread): {e}")
                return None
        
        # Apply monkey patch
        signal.signal = patched_signal
        
        try:
            # Create sgl.Engine with same parameters as subprocess approach
            self.llm_engine = sgl.Engine(
                model_path=model_path,
                tp_size=gpu_count,
                ep_size=1,
                context_length=24000,
                max_running_requests=8,  # Max anticipated concurrent requests
                mem_fraction_static=0.8,
                trust_remote_code=True,
                tool_call_parser="qwen",  # Enable Qwen tool call parsing
            )
            
            # CRITICAL FIX: Don't call get_server_info() synchronously from load()
            # This can cause issues if the event loop isn't running or if called from wrong thread
            # Instead, just verify the engine object exists and has required attributes
            if not hasattr(self.llm_engine, 'tokenizer_manager'):
                raise RuntimeError("Engine tokenizer_manager not initialized")
            
            # Ensure event loop exists (but don't try to run it here)
            if hasattr(self.llm_engine, 'loop'):
                if self.llm_engine.loop.is_closed():
                    raise RuntimeError("Engine event loop is closed after initialization")
                logger.info("Engine event loop is active")
            
            # Keep a reference to prevent garbage collection
            # This ensures atexit handlers don't run prematurely
            self._engine_keepalive = self.llm_engine
            logger.info("Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SGLang engine: {e}", exc_info=True)
            # Clean up if initialization failed
            if hasattr(self, 'llm_engine') and self.llm_engine is not None:
                try:
                    self.llm_engine.shutdown()
                except:
                    pass
            raise
        finally:
            # Restore original signal.signal after initialization
            signal.signal = original_signal
        
        # Load tokenizer for chat template conversion
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        logger.info("SGLang engine loaded successfully!")
        
        # Final verification: Check GPU memory is still allocated
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                logger.info(f"GPU memory allocated after load: {gpu_memory:.2f} GB")
                if gpu_memory < 1.0:  # Less than 1GB suggests model isn't loaded
                    logger.error(f"WARNING: GPU memory is too low ({gpu_memory:.2f} GB). Model may not be loaded correctly.")
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
    
    async def chat_completions(self, request: Request) -> Response:
        """
        OpenAI-compatible chat completions endpoint.
        Receives request, sends it to predict, gets response and streams it back.
        """
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
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response - return JSON
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
            "scale": 0.5,  # Optional, defaults to 0.5
            "max_tokens": 4096,
            "temperature": 0.6,
            "stream": False,
            ...
        }
        """
        # Extract scale parameter (default to 0.5)
        scale = model_input.pop("scale", 0.5)
        
        # Process messages to resize images (same as HTTP server approach)
        messages = model_input.get("messages", [])
        if messages:
            messages = await process_messages(messages, scale)
            model_input["messages"] = messages
        
        # Convert OpenAI messages format to prompt and extract images
        # For VLM models, images must be passed separately via image_data parameter
        prompt, image_data = _convert_messages_to_prompt_and_extract_images(
            messages, 
            self.tokenizer,
            chat_template_name="qwen"  # Qwen3-VL uses "qwen" chat template
        )
        
        # Prepare sampling parameters for sgl.Engine (same structure as HTTP server)
        sampling_params = {
            "max_new_tokens": model_input.get("max_tokens", 4096),
            "temperature": model_input.get("temperature", 0.6),
        }
        
        # Add optional parameters
        if "top_p" in model_input:
            sampling_params["top_p"] = model_input["top_p"]
        if "top_k" in model_input:
            sampling_params["top_k"] = model_input["top_k"]
        if "stop" in model_input:
            sampling_params["stop"] = model_input["stop"]
        
        # Handle tool calling - pass tools to engine if provided
        tools = model_input.get("tools")
        tool_choice = model_input.get("tool_choice")
        # parallel_tool_calls = model_input.get("parallel_tool_calls", False)
        
        # Add tool parameters to sampling_params if tools are provided
        if tools:
            sampling_params["tools"] = tools
        if tool_choice:
            sampling_params["tool_choice"] = tool_choice
        # sampling_params["parallel_tool_calls"] = parallel_tool_calls
        
        stream = model_input.get("stream", False)
        
        # Generate using sgl.Engine async API
        # For VLM, pass image_data as separate parameter
        try:
            if stream:
                # Streaming asynchronous generation
                async def generate():
                    # Use async_stream_and_merge for streaming with overlap removal
                    # Note: async_stream_and_merge might need image_data parameter
                    # If it doesn't support it, we may need to use a different approach
                    try:
                        async for chunk in async_stream_and_merge(
                            self.llm_engine, 
                            prompt, 
                            sampling_params,
                            image_data=image_data if image_data else None
                        ):
                            # Format each chunk as Server-Sent Events
                            data = {
                                "id": f"chatcmpl-{os.urandom(16).hex()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_input.get("model", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(data)}\n\n".encode()
                    except TypeError:
                        # Fallback: async_stream_and_merge might not support image_data
                        # Try without image_data (may not work for VLM)
                        logger.warning("async_stream_and_merge doesn't support image_data, trying without")
                        async for chunk in async_stream_and_merge(
                            self.llm_engine, 
                            prompt, 
                            sampling_params
                        ):
                            data = {
                                "id": f"chatcmpl-{os.urandom(16).hex()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_input.get("model", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(data)}\n\n".encode()
                    
                    # Final done message
                    yield b"data: [DONE]\n\n"
                
                return generate()
            else:
                # Non-streaming asynchronous generation
                # Pass image_data as separate parameter (like the VLM example)
                if image_data:
                    # Handle single vs multiple images
                    if isinstance(image_data, list) and len(image_data) == 1:
                        image_data_param = image_data[0]
                    else:
                        image_data_param = image_data
                    
                    outputs = await self.llm_engine.async_generate(
                        prompt,  # String prompt with image_token placeholders
                        image_data=image_data_param,  # Pass images separately
                        sampling_params=sampling_params
                    )
                else:
                    # No images, regular text generation
                    outputs = await self.llm_engine.async_generate(
                        prompt,
                        sampling_params=sampling_params
                    )
                
                # Convert SGLang result to OpenAI format
                # outputs format: {"text": "...", "meta_info": {...}} or list
                if isinstance(outputs, list):
                    result = outputs[0]
                else:
                    result = outputs
                
                return _convert_to_openai_format(
                    result,
                    model_input,
                    tools=tools
                )
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise

