from typing import Any, Iterator
from transformers import AutoTokenizer
import torch
import torchaudio
import fastapi
from snac import SNAC
from pathlib import Path
import numpy as np
from fastapi.responses import StreamingResponse, Response
import batched
import re
from typing import List, Awaitable
import time
import uuid
import asyncio
import threading
import logging

# force inference mode during the lifetime of the script
_inference_mode_raii_guard = torch._C._InferenceMode(True)
# torch.backends.cuda.matmul.allow_tf32 = True

# TODO(veer/michael): test decoder with bfloat16

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAC_MAX_BATCH = 64
PREPROCESS_STREAM = torch.Stream(SNAC_DEVICE)
MAX_CHARACTERS_INPUT = 6144
MAX_CHUNK_SIZE = 280

MU_LAW_MU = 255  # Standard μ-law compression parameter

class SnacModelBatched:
    def __init__(self):
        self.dtype_decoder = torch.float32
        compile_background = False
        use_compile = True
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        model = model.to(SNAC_DEVICE)

        model.decoder = model.decoder.to(self.dtype_decoder)

        self.snac_model = model
        self.stream = torch.Stream()
        if use_compile:
            if compile_background:
                # Compile the model in a separate thread, experimental.
                threading.Thread(target=self.compile, daemon=True).start()
            else:
                # Compile the model in the main thread
                self.compile()

    def compile(self):
        model = self.snac_model
        # Compile the model with torch.compile
        decoder = torch.compile(model.decoder, dynamic=True)
        quantizer = torch.compile(model.quantizer, dynamic=True)
        t = time.time()
        logging.info("starting torch.compile")
        for bs_size in range(1, max(SNAC_MAX_BATCH, 1)):
            codes = [
                torch.randint(1, 4096, (bs_size, 4)).to(SNAC_DEVICE),
                torch.randint(1, 4096, (bs_size, 8)).to(SNAC_DEVICE),
                torch.randint(1, 4096, (bs_size, 16)).to(SNAC_DEVICE),
            ]
            with torch.inference_mode():
                intermed = quantizer.from_codes(codes)
                decoder(intermed.to(self.dtype_decoder))
        logging.info(f"torch.compile took {time.time() - t:.2f} seconds")
        self.snac_model.decoder = decoder
        self.snac_model.quantizer = quantizer

    @batched.dynamically(batch_size=SNAC_MAX_BATCH, timeout_ms=15)
    def batch_snac_model(
        self, items: list[dict[str, list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
        # Custom processing logic here
        # return [model.decode(item["codes"]) for item in items]
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            all_codes = [codes["codes"] for codes in items]
            can_be_batched = len(items) > 1 and all(
                codes[0].shape == all_codes[0][0].shape for codes in all_codes
            )
            if can_be_batched:
                # stacked_codes = [(b,4), (b,8), (b,16)]
                stacked_codes: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = [
                    torch.cat(  # codes is list[torch.Tensor]
                        [item[i] for item in all_codes], dim=0
                    )
                    for i in range(3)
                ]
                stacked_z_q = self.snac_model.quantizer.from_codes(stacked_codes)
                output_batched = self.snac_model.decoder(
                    stacked_z_q.to(self.dtype_decoder)
                )[:, :, 2048:4096].to(torch.float32)

                out = output_batched.split(
                    1, dim=0
                )  # unbatch the output into len(items) tensors of shape (1, 1, x)
            else:
                # items can't be batched
                if len(items) > 1:
                    # items can't cant be concatenated (no padding)
                    logging.warning(
                        "Warning: items can't be batched, using individual decoding."
                    )
                # if we have a single item, we need to do the same thing as above
                # but without concatenating
                out: list[torch.Tensor] = []
                for codes in all_codes:
                    stacked_z_q = self.snac_model.quantizer.from_codes(codes)
                    out.append(
                        self.snac_model.decoder(stacked_z_q.to(self.dtype_decoder))[
                            :, :, 2048:4096
                        ].to(torch.float32)
                    )
            self.stream.synchronize()  # make sure the results are ready
            return out


model_snac = SnacModelBatched()


def turn_token_into_id(token_string: int, index: int):
    """Extract and convert the last custom token ID from a string."""
    return token_string - 10 - ((index % 7) * 4096)


def split_custom_tokens(s: str) -> List[int]:
    """
    Extracts all substrings enclosed in <custom_token_…> from the input string.
    """
    matches = _TOKEN_RE.findall(s)
    return [int(match) for match in matches if match != "0"]


async def tokens_decoder(
    token_gen: Iterator, request_id: str, start_time: int, encoding: str = "pcm_s16le"
) -> Iterator[bytes]:
    """Decoder that pipelines convert_to_audio calls but enforces strict in-order yields."""
    assert hasattr(token_gen, "__aiter__")
    audio_queue = asyncio.Queue()

    async def producer(token_gen: Iterator):
        buffer: list[int] = []
        count = 0
        tft = 0
        async for token_sim in token_gen:
            if tft == 0:
                tft = time.time()
            for tok_str in split_custom_tokens(token_sim):
                token = turn_token_into_id(int(tok_str), count)
                buffer.append(token)
                count += 1
                # every 7 tokens → one frame; once we have at least 28 tokens, we extract the last 28
                if count % 7 == 0 and count > 27:
                    buf_to_proc = buffer[-28:]
                    task = asyncio.create_task(convert_to_audio(buf_to_proc, encoding=encoding))
                    audio_queue.put_nowait(task)
        audio_queue.put_nowait(None)
        elapsed = time.time() - start_time
        time_to_first_token = tft - start_time
        time_of_generation = time.time() - tft
        token_generation_speed = count / time_of_generation
        logging.info(
            f"Finished `{request_id}`, total tokens : {count}, time: {elapsed:.2f}s. "
            f"tokens/s generation: {token_generation_speed:.2f} (ttft: {time_to_first_token:.2f}s, generation time: {time_of_generation:.2f}s)"
            f" real-time factor once streaming started: {(token_generation_speed / 100):.2f} "
        )

    producer_task = asyncio.create_task(producer(token_gen))

    while True:
        # wait for the next audio conversion to finish
        task: None | Awaitable[bytes | None] = await audio_queue.get()
        if task is None:
            break
        audio_bytes = await task
        if audio_bytes is not None:
            yield audio_bytes
        audio_queue.task_done()
    assert audio_queue.empty(), (
        f"audio queue is not empty: e.g. {audio_queue.get_nowait()}"
    )
    await producer_task

def encode_mulaw(audio: np.ndarray) -> np.ndarray:
    """
    Encode audio samples to μ-law format.
    
    Args:
        audio: Input audio samples as float32 array in range [-1.0, 1.0]
        
    Returns:
        μ-law encoded samples as uint8 array
    """
    # Clamp audio to valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    # Get sign
    sign = np.sign(audio)
    abs_audio = np.abs(audio)
    
    # Apply μ-law companding: y = sign(x) * ln(1 + μ|x|) / ln(1 + μ)
    # Then quantize to 8 bits
    mulaw_audio = sign * np.log1p(MU_LAW_MU * abs_audio) / np.log1p(MU_LAW_MU)
    
    # Quantize to 8-bit: scale from [-1, 1] to [0, 255]
    mulaw_quantized = ((mulaw_audio + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    return mulaw_quantized

@torch.inference_mode()
async def convert_to_audio(frame_ids: list[int], encoding: str = "pcm_s16le") -> bytes | None:
    """Convert a list of token IDs into audio bytes efficiently.

    Args:
        frame_ids: list of token IDs (phonemes) of length 28 or less. 7 tokens = 1 frame
        encoding: Encoding to use. Must be one of "pcm_s16le" or "pcm_mulaw".
    
    Returns:
        Audio bytes in either μ-law (8-bit) or PCM (16-bit) format
    """
    n = len(frame_ids) // 7
    if n == 0:
        return None

    arr = torch.tensor(frame_ids[: n * 7], dtype=torch.int32)
    mat = arr.view(n, 7)
    codes_0 = mat[:, 0]
    codes_1 = mat[:, [1, 4]].reshape(-1)
    codes_2 = mat[:, [2, 3, 5, 6]].reshape(-1)
    if (
        ((codes_0 < 0) | (codes_0 > 4096)).any()
        or ((codes_1 < 0) | (codes_1 > 4096)).any()
        or ((codes_2 < 0) | (codes_2 > 4096)).any()
    ):
        logging.warning("Warn: Invalid token IDs detected, skipping audio generation.")
        return None
    with torch.cuda.stream(PREPROCESS_STREAM):
        codes = [
            codes_0.unsqueeze(0).to(SNAC_DEVICE),
            codes_1.unsqueeze(0).to(SNAC_DEVICE),
            codes_2.unsqueeze(0).to(SNAC_DEVICE),
        ]
        PREPROCESS_STREAM.synchronize()  # only queue codes that are ready
    audio_hat = await model_snac.batch_snac_model.acall({"codes": codes})
    audio_np = audio_hat.numpy(force=True)
    
    if encoding == "pcm_mulaw":
        # Encode to μ-law (8-bit)
        audio_bytes = torchaudio.transforms.MuLawEncoding()(audio_hat).numpy(force=True).tobytes()
    else:
        # Encode to 16-bit PCM
        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
    
    return audio_bytes


class Model:
    def __init__(self, trt_llm, **kwargs) -> None:
        self._secrets = kwargs["secrets"]
        self._engine = trt_llm["engine"]
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self._tokenizer = None
        self.start_id = [128259]
        self.end_ids = [128009, 128260, 128261, 128257]

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            Path(self._data_dir) / "tokenization"
        )
        self.start_tokenized = (
            self._tokenizer.decode(self.start_id) + self._tokenizer.bos_token
        )
        self.end_tokenized = self._tokenizer.decode(self.end_ids)

        self.use_fast_fmt = self._format_prompt_fast(
            "hello world", "tara"
        ) == self._format_prompt_slow("hello world", "tara")

    def _format_prompt_slow(self, prompt, voice="tara"):
        if voice:
            adapted_prompt = f"{voice}: {prompt}"
        else:
            adapted_prompt = prompt
        input_ids = self._tokenizer.encode(
            adapted_prompt,
        )
        full_ids = self.start_id + input_ids + self.end_ids
        return self._tokenizer.decode(full_ids)

    def _format_prompt_fast(self, prompt, voice="tara"):
        token_stream = self.start_tokenized
        if voice:
            token_stream += f"{voice}: "
        token_stream += prompt
        token_stream += self.end_tokenized
        return token_stream

    def format_prompt(self, prompt: str, voice="tara"):
        """Format the prompt for the model."""
        if self.use_fast_fmt:
            return self._format_prompt_fast(prompt, voice)
        else:
            logging.warning("Warn: Using slow format")
            return self._format_prompt_slow(prompt, voice)

    def _chunk_text(self, text: str, max_len: int) -> list[str]:
        """
        Split `text` into chunks of length <= max_len, preferring boundaries in this order:
        double newlines -> sentence enders -> spaces -> hard cut.
        """
        if len(text) <= max_len:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_len, len(text))
            window = text[start:end]

            # Prefer paragraph-ish breaks
            split_at = window.rfind("\n")
            if split_at == -1 or split_at < max_len * 0.5:
                # Prefer sentence boundaries
                split_at = max(window.rfind("."), window.rfind("?"), window.rfind("!"))
                if split_at != -1:
                    split_at += 1  # include the punctuation
            if split_at == -1 or split_at < max_len * 0.33:
                # Prefer last comma
                split_at = window.rfind(",")
            if split_at == -1:
                # Prefer last space
                split_at = window.rfind(" ")

            if split_at == -1:
                split_at = len(window)

            chunk = text[start : start + split_at].strip()
            if chunk:
                chunks.append(chunk)

            start = start + split_at
            # Skip leading whitespace before next chunk
            while start < len(text) and text[start].isspace():
                start += 1

        return chunks or [""]

    async def predict(
        self, model_input: Any, request: fastapi.Request
    ) -> StreamingResponse:
        try:
            req_id = str(model_input.get("request_id", uuid.uuid4()))
            max_chunk_size = model_input.get("max_chunk_size", MAX_CHUNK_SIZE)
            voice = model_input.get("voice", "tara")
            encoding = model_input.get("encoding", "pcm_s16le")

            # 1) Chunk the ORIGINAL prompt before formatting
            original_prompt: str = model_input["prompt"]
            chunks = self._chunk_text(original_prompt, max_chunk_size)
            formatted_chunks = [
                self.format_prompt(chunk, voice=voice) for chunk in chunks
            ]

            total_input_length = len(original_prompt)
            logging.info(
                f"Starting request_id {req_id} with total input length {total_input_length} "
                f"split into {len(chunks)} chunk(s) (max {max_chunk_size} chars per chunk)."
            )

            # 2) Model params (carry across chunks)
            model_input["temperature"] = model_input.get("temperature", 0.6)
            model_input["top_p"] = model_input.get("top_p", 0.8)
            model_input["max_tokens"] = model_input.get("max_tokens", 6144)
            if model_input.get("end_id") is not None:
                logging.info(
                    "Not using end_id from model_input:", model_input["end_id"]
                )
            model_input["end_id"] = 128258
            # model_input["pad_id"] = model_input.get("end_id", [128004]) automatically infered  from AutoTokenizer.from_file(..).pad_token
            model_input["repetition_penalty"] = model_input.get(
                "repetition_penalty", 1.1
            )

            start_time = time.time()

            async def audio_stream(req_id: str):
                # 3) Stream each chunk sequentially (preserves original order)
                for idx, chunk in enumerate(formatted_chunks):
                    model_input["prompt"] = chunk

                    logging.info(f"[{req_id}] Streaming chunk {idx + 1}/{len(chunks)} ")

                    token_gen = await self._engine.predict(model_input, request)
                    if isinstance(token_gen, StreamingResponse):
                        token_gen = token_gen.body_iterator

                    # 4) Forward audio bytes as we receive them
                    async for chunk_bytes in tokens_decoder(
                        token_gen, req_id, start_time, encoding=encoding
                    ):
                        yield chunk_bytes

                logging.info(f"[{req_id}] Completed streaming {len(chunks)} chunk(s).")

            return StreamingResponse(
                audio_stream(req_id),
                media_type="audio/wav",
                headers={"X-Baseten-Input-Tokens": str(total_input_length)},
            )

        except Exception as e:
            print(f"Error in request_id {req_id}: {e} with input {model_input}")
            return Response(
                f"An internal server error occurred while processing your request {req_id}",
                status_code=500,
            )
