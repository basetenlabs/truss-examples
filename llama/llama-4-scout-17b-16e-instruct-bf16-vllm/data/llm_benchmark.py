import argparse
import asyncio
import csv
import json
import logging
import os
import random
import string
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp


from tqdm import tqdm


from transformers import AutoTokenizer


@dataclass
class RequestResult:
    success: bool
    latency: float
    ttft: float
    error: Optional[str] = None
    generated_text: Optional[str] = None
    prompt: Optional[str] = None


@dataclass
class RequestMetrics:
    success: bool
    latency: float
    ttft: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    tpot: float
    error: Optional[str] = None
    generated_text: Optional[str] = None
    prompt: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Benchmarking Tool")
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Backend to benchmark (e.g., 'generic')",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO)",
    )
    parser.add_argument("--api_url", type=str, required=True, help="API endpoint URL")
    parser.add_argument("--model", type=str, required=False, help="Model name or path")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration of the benchmark in seconds (optional)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        nargs="+",
        default=[100],
        help="Number of prompts to process. Can be multiple values to match concurrency levels.",
    )
    parser.add_argument(
        "--output_len", type=int, default=50, help="Maximum output length"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=None,
        help="Number of concurrent requests. Can be multiple values.",
    )
    parser.add_argument(
        "--request_rate", type=float, default=None, help="Number of requests per second"
    )
    parser.add_argument(
        "--random_input", type=int, default=30, help="Input length for random dataset"
    )
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--extra_request_body",
        type=str,
        help="Extra parameters for the request body (JSON format)",
    )
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument(
        "--input_file", type=str, help="Path to input file containing prompts"
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["random", "file", "stdin", "custom"],
        default="random",
        help="Type of input to use",
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file for results (CSV format)"
    )
    parser.add_argument(
        "--disable_warmup", action="store_true", help="Disable warmup requests"
    )
    parser.add_argument(
        "--warmup_requests",
        type=int,
        default=1,
        help="Number of warmup requests to perform",
    )
    parser.add_argument(
        "--tokenizer", type=str, help="Name of the AutoTokenizer to use (e.g., 'gpt2')"
    )
    parser.add_argument(
        "--api_key", type=str, required=True, help="API key for authentication"
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["prompt", "messages"],
        default="prompt",
        help="Style of prompt to use: 'prompt' for single string, 'messages' for chat-style",
    )
    parser.add_argument(
        "--prompt_multiplier",
        type=int,
        default=1,
        help="Number of times to repeat each prompt",
    )

    args = parser.parse_args()

    if args.duration is not None:
        if len(args.concurrency) != 1:
            parser.error(
                "When using --duration, specify exactly one --concurrency value."
            )

    # Ensure num_prompts and concurrency have matching lengths or handle appropriately
    if args.concurrency is not None:
        if len(args.num_prompts) == 1 and len(args.concurrency) > 1:
            # If only one num_prompts value is provided, duplicate it for each concurrency
            args.num_prompts = args.num_prompts * len(args.concurrency)
        elif len(args.num_prompts) != len(args.concurrency):
            parser.error(
                f"Number of --num_prompts values ({len(args.num_prompts)}) must match "
                f"number of --concurrency values ({len(args.concurrency)})."
            )

    # Parse extra_request_body if provided
    if args.extra_request_body:
        try:
            args.extra_request_body = json.loads(args.extra_request_body)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for --extra_request_body")
    else:
        args.extra_request_body = {}

    # Ensure input_file is provided if input_type is 'file'
    if args.input_type == "file" and not args.input_file:
        parser.error("--input_file is required when --input_type is 'file'")

    # Set default output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"benchmark_{timestamp}.csv"
    elif not args.output_file.endswith(".csv"):
        args.output_file += ".csv"

    if args.concurrency is None and args.request_rate is None:
        parser.error("At least one of --concurrency or --request_rate must be provided")

    if args.concurrency is None:
        args.concurrency = [
            float("inf")
        ]  # Use infinite concurrency for rate-limited benchmarks

    if args.request_rate is None:
        args.request_rate = float("inf")
    return args


args = parse_args()

logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class InputHandler:
    def __init__(self, args, prompt_count_index=0):
        self.args = args
        # Use the specified prompt count index or default to the first one
        self.prompt_count = (
            args.num_prompts[prompt_count_index]
            if isinstance(args.num_prompts, list)
            else args.num_prompts
        )

    def get_prompts(self):
        prompts = []
        if self.args.input_type == "random":
            prompts = self.generate_random_prompts()
        elif self.args.input_type == "file":
            prompts = self.read_prompts_from_file()
        elif self.args.input_type == "stdin":
            prompts = self.read_prompts_from_stdin()
        elif self.args.input_type == "custom":
            prompts = self.get_prompts_custom()
        else:
            raise ValueError(f"Invalid input type: {self.args.input_type}")

        return prompts * self.args.prompt_multiplier

    def get_timed_prompts(self):
        if self.args.input_type == "random":
            while True:
                yield self.generate_single_random_prompt()
        else:
            prompts = self.get_prompts()
            while True:
                yield from prompts

    def generate_single_random_prompt(self):
        return " ".join(
            "".join(random.choices(string.ascii_lowercase, k=5))
            for _ in range(self.args.random_input)
        )

    def get_prompts_custom(self):
        # Read the frank.txt file
        try:
            with open("frank.txt", "r", encoding="utf-8") as f:
                frank_text = f.read()
        except Exception as e:
            logger.error(f"Error reading frank.txt: {e}")
            raise

        # Initialize the tokenizer
        if self.args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)
        else:
            logger.error("Tokenizer is required for custom input type")
            raise ValueError("Tokenizer is required for custom input type")

        # Generate a list of random tokens to insert at the beginning of each prompt
        random_tokens = []
        for _ in range(self.prompt_count):
            # Generate a random token (a random word)
            random_token = "".join(random.choices(string.ascii_lowercase, k=5))
            random_tokens.append(random_token)

        # Create prompts by truncating the text to the specified length
        prompts = []
        for i in range(self.prompt_count):
            # Insert the random token at the beginning
            prompt_with_token = random_tokens[i] + " " + frank_text

            # Tokenize and truncate to random_input length
            tokenized_prompt = tokenizer(
                prompt_with_token,
                max_length=self.args.random_input,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            # Convert token IDs back to text
            truncated_prompt = tokenizer.decode(
                tokenized_prompt[0], skip_special_tokens=True
            )
            prompts.append(truncated_prompt)

        return prompts

    def generate_random_prompts(self):
        random.seed(42)
        # 5 letter words
        random_prompts = [
            " ".join(
                "".join(random.choices(string.ascii_lowercase, k=5))
                for _ in range(self.args.random_input)
            )
            for _ in range(self.prompt_count)
        ]
        # use tokenizer to return prompt for proper length
        logger.info(
            f"If a tokenizer was provided we'll truncate the prompts to {self.args.random_input}"
        )

        if self.args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)
            tokenized_prompts = [
                tokenizer(
                    prompt,
                    max_length=self.args.random_input,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                for prompt in random_prompts
            ]

            # Convert token IDs back to prompts for more readability
            truncated_prompts = [
                tokenizer.decode(prompt[0], skip_special_tokens=True)
                for prompt in tokenized_prompts
            ]

            return truncated_prompts
        else:
            return random_prompts

    def read_prompts_from_file(self):
        with open(self.args.input_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return self._handle_prompt_count(prompts)

    def read_prompts_from_stdin(self):
        prompts = []
        logger.info("\nEnter your prompts. Press Enter after each prompt.")
        logger.info(f"Enter up to {self.prompt_count} prompts.")
        logger.info(
            "When you're finished, press Ctrl+D (Unix) or Ctrl+Z (Windows) followed by Enter."
        )

        for i in range(1, self.prompt_count + 1):
            try:
                prompt = input(f"Prompt {i}: ").strip()
                if prompt:
                    prompts.append(prompt)
                else:
                    logger.info(
                        "Empty prompt ignored. Continue with the next or finish input."
                    )
            except EOFError:
                break

        if not prompts:
            logger.info("No prompts were entered. Exiting.")
            sys.exit(1)

        return self._handle_prompt_count(prompts)

    def _handle_prompt_count(self, prompts):
        if len(prompts) < self.prompt_count:
            logger.info(
                f"\nWarning: Only {len(prompts)} prompt(s) available. Using all available prompts."
            )
        elif len(prompts) > self.prompt_count:
            logger.info(
                f"\nWarning: {len(prompts)} prompts found. Using first {self.prompt_count} prompts."
            )
            prompts = prompts[: self.prompt_count]
        return prompts


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self, prompt: str, session: aiohttp.ClientSession
    ) -> RequestResult:
        pass


class StreamHandler:
    def __init__(self):
        self.buffer = ""
        self.generated_text = ""
        self.first_token_time = None
        self.response_format = None
        self.special_tokens = {
            "start_header": "<|start_header_id|>",
            "end_header": "<|end_header_id|>",
            "eot": "<|eot_id|>",
        }

    def process_chunk(self, chunk: str) -> str:
        if not self.response_format == "openai":
            self.detect_format(chunk)

        if self.response_format == "openai":
            return self.process_openai_chunk(chunk)
        elif self.response_format == "special_tokens":
            return self.process_special_tokens_chunk(chunk)
        else:
            return chunk

    def process_special_tokens_chunk(self, chunk: str) -> str:
        # Remove the header if present
        if (
            self.special_tokens["start_header"] in chunk
            and self.special_tokens["end_header"] in chunk
        ):
            _, chunk = chunk.split(self.special_tokens["end_header"], 1)
            chunk = chunk.lstrip()  # Remove leading whitespace
            if (
                not chunk
            ):  # If chunk is empty after removing header, return empty string
                return ""

        # Remove the EOT token if present
        if self.special_tokens["eot"] in chunk:
            chunk, _ = chunk.split(self.special_tokens["eot"], 1)

        return chunk

    def process_openai_chunk(self, chunk: str) -> str:
        chunk = chunk.strip()

        if chunk.startswith("data: "):
            chunk = chunk[6:]
        # Remove 'data: ' prefix
        if chunk == "[DONE]":
            return ""
        if chunk == "":
            return ""
        try:
            data = json.loads(chunk)
            return data["choices"][0]["delta"].get("content", "")
        except json.JSONDecodeError:
            return ""

    def detect_format(self, chunk: str):
        if chunk.startswith("data: "):
            self.response_format = "openai"
        elif any(token in chunk for token in self.special_tokens.values()):
            self.response_format = "special_tokens"
        else:
            self.response_format = "unknown"

    def update_metrics(self, content: str):
        if content and self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        if content is not None:
            self.generated_text += content

    def get_results(self) -> Dict[str, Any]:
        return {
            "generated_text": self.generated_text,
            "first_token_time": self.first_token_time,
        }


class GenericRestProvider(LLMProvider):
    def __init__(self, args):
        self.api_url = args.api_url
        self.api_key = args.api_key
        self.tokenizer = self._init_tokenizer(args.tokenizer)
        self.args = args
        self.prompt_style = args.prompt_style
        self.model_name = args.model

    def _init_tokenizer(self, tokenizer_name: Optional[str]):
        if tokenizer_name:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        return None

    def _tokenize(self, text: str) -> list:
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return text.split()  # Simple fallback tokenization

    async def generate(self, prompt: str, session) -> "RequestResult":
        payload = self._prepare_payload(prompt, self.args.stream)
        headers = self._prepare_headers()

        start_time = time.perf_counter()
        if self.args.stream:
            return await self._handle_streaming_response(
                payload=payload,
                headers=headers,
                start_time=start_time,
                prompt=prompt,
                session=session,
            )
        else:
            return await self._handle_non_streaming_response(
                payload=payload,
                headers=headers,
                start_time=start_time,
                prompt=prompt,
                session=session,
            )

    def _prepare_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        payload = {
            "stream": stream,
            "max_tokens": self.args.output_len,
            **self.args.extra_request_body,
        }
        if self.prompt_style == "messages":
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt

        if self.model_name:
            payload["model"] = self.model_name

        if self.args.stream:
            payload["stream"] = self.args.stream

        return {k: v for k, v in payload.items() if v is not None}

    def _prepare_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
        }

    async def _handle_non_streaming_response(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        start_time: float,
        prompt: str,
    ) -> "RequestResult":
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url, json=payload, headers=headers, timeout=None
            ) as response:
                end_time = time.perf_counter()
                response_text = await response.text()
                return RequestResult(
                    success=True,
                    generated_text=response_text,
                    latency=end_time - start_time,
                    ttft=0,
                    prompt=prompt,
                )

    async def _handle_streaming_response(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        start_time: float,
        prompt: str,
        session: aiohttp.ClientSession,
    ) -> "RequestResult":
        time.perf_counter()
        response = await session.post(
            self.api_url, json=payload, headers=headers, timeout=None
        )
        # logger.info(f"Streaming request sent at {request_sent_time - START_TIME:.6f} seconds after TEST start")

        async with response:
            if response.status != 200:
                return RequestResult(
                    success=False,
                    error=f"API Error: HTTP {response.status}",
                    latency=0,
                    ttft=0,
                    prompt=prompt,
                )

            stream_handler = StreamHandler()

            async for chunk in response.content:
                # print(chunk)
                chunk = chunk.decode("utf-8")
                content = stream_handler.process_chunk(chunk)
                stream_handler.update_metrics(content)

            end_time = time.perf_counter()
            results = stream_handler.get_results()
            latency = end_time - start_time
            ttft = (
                results["first_token_time"] - start_time
                if results["first_token_time"]
                else 0
            )

            return RequestResult(
                success=True,
                generated_text=results["generated_text"],
                latency=latency,
                ttft=ttft,
                prompt=prompt,
            )

    def _extract_generated_text(self, json_response: Dict[str, Any]) -> str:
        return json_response.get("generated_text", "")


class RequestHandler:
    def __init__(self, provider: LLMProvider, args):
        self.provider = provider
        self.concurrency = args.concurrency
        self.request_rate = args.request_rate
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
        )

        # set timeout to 20 minutes
        self.timeout = 1200
        self.stream = args.stream
        self.disable_tqdm = args.disable_tqdm
        self.warmup_results = []
        self.initial_delay = 0.01
        self.input_handler = InputHandler(args)

    async def warmup(self, prompts, num_warmup_requests: int = 5):
        logger.info("Performing warmup requests...")
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=32, keepalive_timeout=30),
            timeout=aiohttp.ClientTimeout(
                total=6 * 60 * 60, connect=60, sock_connect=60
            ),
        ) as session:
            warmup_tasks = [
                self.provider.generate(prompt, session)
                for prompt in prompts[:num_warmup_requests]
            ]
            self.warmup_results = await asyncio.gather(*warmup_tasks)
        logger.info("Warmup completed.")
        # This is too conservative, setting it to 0.01 manually yields better results
        # self._calculate_initial_delay()

    def _calculate_initial_delay(self):
        if self.warmup_results:
            delays = [
                r.ttft
                - ((r.latency - r.ttft) / len(self.tokenizer.encode(r.generated_text)))
                for r in self.warmup_results
                if r.success
            ]
            if delays:
                self.initial_delay = sum(delays) / len(delays)
        logger.info(f"Initial delay set to {self.initial_delay:.3f} seconds")

    async def make_request(
        self, prompt: str, session: aiohttp.ClientSession
    ) -> RequestResult:
        start_time = time.perf_counter()
        result = await asyncio.wait_for(
            self.provider.generate(prompt, session), timeout=self.timeout
        )
        end_time = time.perf_counter()
        result.latency = end_time - start_time
        return result

    async def run_benchmark(self, prompts: List[str]) -> List[RequestResult]:
        if self.request_rate < float("inf") and self.concurrency == float("inf"):
            return await self.run_rate_only_benchmark(prompts)
        elif self.request_rate < float("inf"):
            return await self.run_rate_limited_benchmark(prompts)
        else:
            return await self.run_max_concurrency_benchmark(prompts)

    async def run_timed_benchmark(self, duration: int) -> List[RequestResult]:
        results = []
        end_time = time.perf_counter() + duration
        prompts = self.input_handler.get_timed_prompts()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def controlled_request():
            async with semaphore:
                prompt = next(prompts)
                return await self.make_request(prompt)

        with tqdm(total=duration, disable=self.disable_tqdm, unit="s") as pbar:
            while time.perf_counter() < end_time:
                if self.request_rate < float("inf"):
                    await asyncio.sleep(1 / self.request_rate)

                task = asyncio.create_task(controlled_request())
                result = await task
                results.append(result)
                pbar.update(
                    min(
                        time.perf_counter() - pbar.last_print_t,
                        end_time - time.perf_counter(),
                    )
                )

                if not result.success:
                    logger.error(f"Error: {result.error}")

        return results

    async def run_rate_limited_benchmark(
        self, prompts: List[str]
    ) -> List[RequestResult]:
        logger.info(
            f"Running rate limited benchmark with concurrency {self.concurrency} and request rate {self.request_rate}"
        )
        results = []
        semaphore = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=32, keepalive_timeout=30),
            timeout=aiohttp.ClientTimeout(
                total=6 * 60 * 60, connect=60, sock_connect=60
            ),
        ) as session:

            async def controlled_request(prompt):
                async with semaphore:
                    return await self.make_request(prompt, session)

            tasks = []

            # Create tasks with appropriate delays
            for i, prompt in enumerate(prompts):
                await asyncio.sleep(
                    1 / self.request_rate
                )  # Delay before creating each task
                tasks.append(asyncio.create_task(controlled_request(prompt)))

            with tqdm(total=len(prompts), disable=self.disable_tqdm) as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    pbar.update(1)
                    if not result.success:
                        logger.error(f"Error: {result.error}")

            return results

    async def run_max_concurrency_benchmark(
        self, prompts: List[str]
    ) -> List[RequestResult]:
        results = []
        semaphore = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=32, keepalive_timeout=30),
            timeout=aiohttp.ClientTimeout(
                total=6 * 60 * 60, connect=60, sock_connect=60
            ),
        ) as session:

            async def controlled_request(prompt):
                async with semaphore:
                    return await self.make_request(prompt, session)

            tasks = []

            # Stagger only the initial batch (up to concurrency limit)
            for i, prompt in enumerate(prompts[: self.concurrency]):
                await asyncio.sleep(
                    i * self.initial_delay
                )  # Delay before starting each request in the initial batch
                tasks.append(asyncio.create_task(controlled_request(prompt)))

            # Queue the remaining prompts to run immediately as spots open up in the semaphore
            for prompt in prompts[self.concurrency :]:
                tasks.append(asyncio.create_task(controlled_request(prompt)))

            with tqdm(total=len(prompts), disable=self.disable_tqdm) as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    pbar.update(1)
                    if not result.success:
                        logger.error(f"Error: {result.error}")

        return results

    async def run_rate_only_benchmark(self, prompts: List[str]) -> List[RequestResult]:
        results = []
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=32, keepalive_timeout=30),
            timeout=aiohttp.ClientTimeout(
                total=6 * 60 * 60, connect=60, sock_connect=60
            ),
        ) as session:

            async def controlled_request(prompt):
                return await self.make_request(prompt, session)

            tasks = []

            # Create tasks with appropriate delays
            for i, prompt in enumerate(prompts):
                await asyncio.sleep(
                    1 / self.request_rate
                )  # Delay before creating each task
                tasks.append(asyncio.create_task(controlled_request(prompt)))

            with tqdm(total=len(prompts), disable=self.disable_tqdm) as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    pbar.update(1)
                    if not result.success:
                        logger.error(f"Error: {result.error}")

        return results


def get_llm_provider(args) -> LLMProvider:
    if args.backend == "generic":
        return GenericRestProvider(args)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")


class MetricsCalculator:
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def calculate_request_metrics(self, result: RequestResult) -> RequestMetrics:
        prompt_token_count = len(self.tokenizer.encode(result.prompt))
        completion_token_count = (
            len(self.tokenizer.encode(result.generated_text))
            if result.generated_text
            else 0
        )
        total_tokens = prompt_token_count + completion_token_count

        tokens_per_second = total_tokens / result.latency if result.latency > 0 else 0

        # Calculate TPOT
        tpot = 0
        if completion_token_count > 0:
            tpot = (result.latency - result.ttft) / completion_token_count

        return RequestMetrics(
            success=result.success,
            latency=result.latency,
            ttft=result.ttft,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            tpot=tpot,
            error=result.error,
            generated_text=result.generated_text,
            prompt=result.prompt,
        )

    def calculate_aggregate_metrics(
        self,
        request_metrics: List[RequestMetrics],
        concurrency: int,
        test_times: Dict[int, float],
    ) -> Dict[str, Any]:
        successful_requests = [r for r in request_metrics if r.success]

        if not successful_requests:
            return self._empty_metrics(len(request_metrics))

        metrics = {
            "total_requests": len(request_metrics),
            "successful_requests": len(successful_requests),
            "failure_rate": 1 - (len(successful_requests) / len(request_metrics)),
        }

        latencies = [r.latency for r in successful_requests]
        ttfts = [r.ttft for r in successful_requests]
        prompt_tokens = [r.prompt_tokens for r in successful_requests]
        completion_tokens = [r.completion_tokens for r in successful_requests]
        total_tokens = [r.total_tokens for r in successful_requests]
        tokens_per_second = [r.tokens_per_second for r in successful_requests]
        tpots = [r.tpot for r in successful_requests]

        metrics.update(
            {
                "average_latency": sum(latencies) / len(latencies),
                "p50_latency": sorted(latencies)[len(latencies) // 2],
                "p90_latency": sorted(latencies)[int(len(latencies) * 0.9)],
                "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
                "average_ttft": sum(ttfts) / len(ttfts),
                "p50_ttft": sorted(ttfts)[len(ttfts) // 2],
                "p90_ttft": sorted(ttfts)[int(len(ttfts) * 0.9)],
                "p99_ttft": sorted(ttfts)[int(len(ttfts) * 0.99)],
                "average_tpot": sum(tpots) / len(tpots),
                "p50_tpot": sorted(tpots)[len(tpots) // 2],
                "p90_tpot": sorted(tpots)[int(len(tpots) * 0.9)],
                "p99_tpot": sorted(tpots)[int(len(tpots) * 0.99)],
                "average_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
                "average_completion_tokens": sum(completion_tokens)
                / len(completion_tokens),
                "average_total_tokens": sum(total_tokens) / len(total_tokens),
                "total_prompt_tokens": sum(prompt_tokens),
                "total_completion_tokens": sum(completion_tokens),
                "total_tokens": sum(total_tokens),
                "average_perceived_tokens_per_second": sum(tokens_per_second)
                / len(tokens_per_second),
            }
        )

        metrics["average_overall_throughput"] = (
            metrics["average_perceived_tokens_per_second"] * concurrency
        )

        return metrics

    def _empty_metrics(self, total_requests: int) -> Dict[str, Any]:
        return {
            "total_requests": total_requests,
            "successful_requests": 0,
            "failure_rate": 1.0,
            "average_latency": 0,
            "p50_latency": 0,
            "p90_latency": 0,
            "p99_latency": 0,
            "average_ttft": 0,
            "p50_ttft": 0,
            "p90_ttft": 0,
            "p99_ttft": 0,
            "average_prompt_tokens": 0,
            "average_completion_tokens": 0,
            "average_total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "average_perceived_tokens_per_second": 0,
            "average_overall_throughput": 0,
            "average_tpot": 0,
            "p50_tpot": 0,
            "p90_tpot": 0,
            "p99_tpot": 0,
        }


class BenchmarkExecutor:
    def __init__(
        self, args, request_handler: RequestHandler, input_handler: InputHandler
    ):
        self.args = args
        self.input_handler = input_handler
        self.metrics_calculator = MetricsCalculator(args.tokenizer)
        self.warmup_run_counter = 0
        self.test_times = {}

    async def run(self) -> Dict[int, List[RequestResult]]:
        if self.args.duration is not None:
            return await self.run_timed_test(self.args.duration)
        else:
            return await self.run_standard_test()

    async def run_standard_test(self) -> Dict[int, List[RequestResult]]:
        all_results = {}

        for i, concurrency in enumerate(self.args.concurrency):
            start_time = time.perf_counter()
            logger.info(f"Running benchmark with concurrency: {concurrency}")

            # Create a new input handler with the appropriate prompt count index
            input_handler = InputHandler(self.args, prompt_count_index=i)
            prompts = input_handler.get_prompts()

            logger.info(
                f"Using {input_handler.prompt_count} prompts for concurrency {concurrency}"
            )

            llm_provider = get_llm_provider(self.args)
            request_handler = RequestHandler(llm_provider, self.args)
            request_handler.concurrency = concurrency  # Set concurrency for this run
            request_handler.input_handler = input_handler  # Update input handler

            # Perform warmup
            if (
                not self.args.disable_warmup and self.warmup_run_counter < 1
            ):  # dont' run warmup for each concurrency
                await request_handler.warmup(prompts, self.args.warmup_requests)
                self.warmup_run_counter += 1

            # Run the actual benchmark
            results = await request_handler.run_benchmark(prompts)
            all_results[concurrency] = results
            end_time = time.perf_counter()
            self.test_times[concurrency] = end_time - start_time

        return all_results

    async def run_timed_test(self, duration: int) -> Dict[int, List[RequestResult]]:
        concurrency = self.args.concurrency[0]  # We know there's only one value
        logger.info(
            f"\nRunning timed benchmark for {duration} seconds with concurrency {concurrency} and request rate {self.args.request_rate}"
        )

        # Create a new input handler with the appropriate prompt count index (0 for timed test)
        input_handler = InputHandler(self.args, prompt_count_index=0)
        logger.info(f"Using {input_handler.prompt_count} prompts for timed test")

        llm_provider = get_llm_provider(self.args)
        request_handler = RequestHandler(llm_provider, self.args)
        request_handler.concurrency = concurrency
        request_handler.input_handler = input_handler  # Update input handler

        # Perform warmup
        if not self.args.disable_warmup:
            warmup_prompts = input_handler.get_prompts()[: self.args.warmup_requests]
            await request_handler.warmup(warmup_prompts, self.args.warmup_requests)

        start_time = time.perf_counter()
        results = await request_handler.run_timed_benchmark(duration)
        end_time = time.perf_counter()

        self.test_times[concurrency] = end_time - start_time
        return {concurrency: results}

    def format_output(self, all_metrics: List[Dict[str, Any]]) -> str:
        output = "Benchmark Results:\n"
        for metrics in all_metrics:
            output += f"\nConcurrency: {metrics['concurrency']}, Request Rate: {metrics['request_rate']}\n"
            output += "-" * 50 + "\n"
            for key, value in metrics.items():
                if key not in ["concurrency", "request_rate"]:
                    if isinstance(value, float):
                        output += f"{key}: {value:.4f}\n"
                    else:
                        output += f"{key}: {value}\n"
        return output

    def save_results_to_csv(self, all_results: Dict[int, List[RequestResult]]):
        output_dir = os.path.join(os.getcwd(), self.args.model)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, self.args.output_file)

        with open(output_file, "w", newline="") as csvfile:
            fieldnames = [
                "concurrency",
                "request_rate",
                "success",
                "latency",
                "ttft",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "tokens_per_second",
                "tpot",
                "error",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for concurrency, results in all_results.items():
                for result in results:
                    metrics = self.metrics_calculator.calculate_request_metrics(result)
                    row = asdict(metrics)
                    row["concurrency"] = concurrency
                    row["request_rate"] = self.args.request_rate
                    # Remove prompt and generated_text from CSV output
                    row.pop("prompt", None)
                    row.pop("generated_text", None)
                    writer.writerow(row)

    def save_generated_text(self, all_results: Dict[int, List[RequestResult]]):
        output_dir = os.path.join(os.getcwd(), self.args.model)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, self.args.output_file.rsplit(".", 1)[0] + "_generated_text.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for concurrency, results in all_results.items():
                f.write(f"Concurrency: {concurrency}\n")
                f.write("=" * 50 + "\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"Request {i}:\n")
                    f.write(f"Prompt: {result.prompt}\n")
                    f.write(f"Generated Text: {result.generated_text}\n")
                    f.write("\n" + "-" * 50 + "\n\n")

    def calculate_final_metrics(
        self, all_results: Dict[int, List[RequestResult]]
    ) -> List[Dict[str, Any]]:
        all_metrics = []
        for concurrency, results in all_results.items():
            request_metrics = [
                self.metrics_calculator.calculate_request_metrics(r) for r in results
            ]
            metrics = self.metrics_calculator.calculate_aggregate_metrics(
                request_metrics, concurrency, self.test_times
            )
            metrics["concurrency"] = concurrency
            metrics["request_rate"] = self.args.request_rate
            all_metrics.append(metrics)
        return all_metrics

    async def execute(self) -> str:
        all_results = await self.run()
        self.save_results_to_csv(all_results)
        self.save_generated_text(all_results)
        aggregate_metrics = self.calculate_final_metrics(all_results)
        return self.format_output(aggregate_metrics)


async def main():
    # Create initial input handler with index 0
    input_handler = InputHandler(args, prompt_count_index=0)
    llm_provider = get_llm_provider(args)
    request_handler = RequestHandler(llm_provider, args)
    request_handler.input_handler = input_handler  # Update input handler

    executor = BenchmarkExecutor(args, request_handler, input_handler)
    output = await executor.execute()

    logger.info(output)
    logger.info(f"Detailed results saved to {args.model}/{args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
