import asyncio
import base64
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator, List, Awaitable

import batched
import numpy as np
import pysbd
import torch
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from snac import SNAC
from transformers import AutoTokenizer
import torchaudio
from audioop import lin2ulaw

# force inference mode during the lifetime of the script
_inference_mode_raii_guard = torch._C._InferenceMode(True)
# torch.backends.cuda.matmul.allow_tf32 = True

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
SNAC_DEVICE = "cuda"
SNAC_MAX_BATCH = 64
PREPROCESS_STREAM = torch.Stream(SNAC_DEVICE)

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
        print("starting torch.compile")
        for bs_size in range(1, max(SNAC_MAX_BATCH, 1)):
            codes = [
                torch.randint(1, 4096, (bs_size, 4)).to(SNAC_DEVICE),
                torch.randint(1, 4096, (bs_size, 8)).to(SNAC_DEVICE),
                torch.randint(1, 4096, (bs_size, 16)).to(SNAC_DEVICE),
            ]
            with torch.inference_mode():
                intermed = quantizer.from_codes(codes)
                decoder(intermed.to(self.dtype_decoder))
        print("finish time for torch.compile:", time.time() - t)
        self.snac_model.decoder = decoder
        self.snac_model.quantizer = quantizer

    @batched.dynamically(batch_size=SNAC_MAX_BATCH, timeout_ms=15)
    def batch_snac_model(
        self, items: list[dict[str, list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
        # Custom processing logic here
        # return [model.decode(item["codes"]) for item in items]
        if len(items) == SNAC_MAX_BATCH:
            print("batch size is max:", len(items))
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
                    print(f"running unbatched at size {len(items)}")
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


async def tokens_decoder(token_gen: Iterator, request_id: str = "", encoding: str = "pcm_s16le") -> Iterator[dict]:
    """Decoder that pipelines convert_to_audio calls but enforces strict in-order yields.
    
    Yields:
        dict with keys:
            - 'audio_bytes': the audio data
            - 'token_gen_time': time taken to generate this chunk (in seconds)
            - 'tokens_processed': number of tokens processed so far
    """
    assert hasattr(token_gen, "__aiter__")
    audio_queue = asyncio.Queue()

    async def producer(token_gen: Iterator):
        buffer: list[int] = []
        count = 0
        chunk_start_time = time.time()
        async for token_sim in token_gen:
            for tok_str in split_custom_tokens(token_sim):
                token = turn_token_into_id(int(tok_str), count)
                buffer.append(token)
                count += 1
                # every 7 tokens → one frame; once we have at least 28 tokens, we extract the last 28
                if count % 7 == 0 and count > 27:
                    buf_to_proc = buffer[-28:]
                    chunk_end_time = time.time()
                    chunk_duration = chunk_end_time - chunk_start_time
                    task = asyncio.create_task(convert_to_audio(buf_to_proc, encoding=encoding))
                    audio_queue.put_nowait({
                        'task': task,
                        'token_gen_time': chunk_duration,
                        'tokens_processed': count
                    })
                    chunk_start_time = time.time()
        audio_queue.put_nowait(None)
        print(f"Finished `{request_id}`, total tokens : {count}")

    producer_task = asyncio.create_task(producer(token_gen))

    while True:
        # wait for the next audio conversion to finish
        item: None | dict = await audio_queue.get()
        if item is None:
            break
        audio_bytes = await item['task']
        if audio_bytes is not None:
            yield {
                'audio_bytes': audio_bytes,
                'token_gen_time': item['token_gen_time'],
                'tokens_processed': item['tokens_processed']
            }
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
        print("Warn: Invalid token IDs detected, skipping audio generation.")
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
    # Encode to 16-bit PCM
    audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()

    if encoding == "pcm_mulaw":
        audio_bytes = lin2ulaw(audio_bytes, 2)

    return audio_bytes


class Model:
    def __init__(self, trt_llm, **kwargs) -> None:
        self._secrets = kwargs["secrets"]
        self._engine = trt_llm["engine"]
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self._tokenizer = None
        self.websocket_connections: dict[str, dict] = {}
        self.start_id = [128259]
        self.end_ids = [128009, 128260, 128261, 128257]
        self.text_splitter = pysbd.Segmenter(language="en", clean=False)

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
            print("Warn: Using slow format")
            return self._format_prompt_slow(prompt, voice)

    async def websocket(self, ws: WebSocket):
        """
        WebSocket endpoint for streaming TTS.
        
        Expected metadata parameters:
        - voice: Voice to use (default: "tara")
        - max_tokens: Maximum tokens to generate (default: 6144)
        - temperature: Sampling temperature (default: 0.6)
        - top_p: Top-p sampling (default: 0.8)
        - repetition_penalty: Repetition penalty (default: 1.3)
        - buffer_size: Word buffer size before flushing (default: 10)
        - encoding: Audio encoding - "pcm_s16le" or "pcm_mulaw" (default: "pcm_s16le")
        - include_timing_info: Whether to include timing metrics in response (default: False)
        """
        # satisfy Truss's metrics/cancellation wrapper
        async def _never_disconnected():
            return False

        ws.is_disconnected = _never_disconnected

        sid = str(uuid.uuid4())
        print(f"[ws:{sid}] entered at {time.time()}")

        # 1) receive metadata
        params = await ws.receive_json()
        print(f"[ws:{sid}] metadata: {params!r}")
        voice = params.get("voice", "tara")
        max_tokens = params.get("max_tokens", 6144)
        temperature = params.get("temperature", 0.6)
        top_p = params.get("top_p", 0.8)
        rep_pen = params.get("repetition_penalty", 1.3)
        buf_sz = int(params.get("buffer_size", 10))
        encoding = params.get("encoding", "pcm_s16le")
        include_timing_info = params.get("include_timing_info", False)
        print(f" → voice={voice}, buffer_size={buf_sz}, encoding={encoding}, timing={include_timing_info}")

        # initialize per-sid state
        self.websocket_connections[sid] = {
            "text_buffer": [],  # this is your cache
            "first_text_time": None if include_timing_info else False,  # track when first text was received (False = disabled)
            "first_audio_sent": False,  # track if first audio was sent
            "include_timing_info": include_timing_info,  # whether to include timing info
        }

        async def flush(final=False):
            buf = self.websocket_connections[sid]["text_buffer"]
            if not buf:
                return
            full_text = " ".join(buf)
            sentences = self.text_splitter.segment(full_text)
            if len(sentences) > 1:  # flush all complete sentences
                complete_sents = sentences[:-1]
                prompt = " ".join(complete_sents)
                words_consumed = sum(len(s.split()) for s in complete_sents)
                del buf[:words_consumed]
            elif len(buf) >= buf_sz:
                # forcefully clear the buffer based on buffer size
                chunk = buf[:buf_sz]
                prompt = " ".join(chunk)
                del buf[:buf_sz]
            elif final:  # Clear the remaining buffer
                prompt = " ".join(buf)
                buf.clear()
            else:
                return

            print(f"Flushing prompt: {prompt}")

            inp = {
                "request_id": sid,
                "prompt": self.format_prompt(prompt, voice),
                "voice": voice,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": rep_pen,
                "end_id": 128258,
            }

            tokgen = await self._engine.predict(inp, ws)
            if isinstance(tokgen, StreamingResponse):
                tokgen = tokgen.body_iterator

            sent = 0
            async for audio_data in tokens_decoder(tokgen, sid, encoding=encoding):
                audio_bytes = audio_data['audio_bytes']
                sent += len(audio_bytes)
                
                # Create JSON payload with base64-encoded audio
                payload = {
                    'audio': base64.b64encode(audio_bytes).decode('utf-8'),
                }
                
                # Optionally include timing info
                if self.websocket_connections[sid]["include_timing_info"]:
                    payload['token_gen_time'] = audio_data['token_gen_time']
                    payload['tokens_processed'] = audio_data['tokens_processed']
                    payload['audio_size_bytes'] = len(audio_bytes)
                
                # Calculate end-to-end latency for first audio chunk (if timing enabled)
                if not self.websocket_connections[sid]["first_audio_sent"]:
                    first_text_time = self.websocket_connections[sid]["first_text_time"]
                    if first_text_time is not None and first_text_time is not False:
                        end_to_end_latency = (time.time() - first_text_time) * 1000  # ms
                        payload['first_chunk_e2e_latency_ms'] = end_to_end_latency
                        print(f"[ws:{sid}] First audio latency: {end_to_end_latency:.2f} ms")
                    self.websocket_connections[sid]["first_audio_sent"] = True
                
                # print(f"[ws:{sid}] sending {len(audio_bytes)} bytes (total {sent}) with gen_time={audio_data['token_gen_time']:.4f}s")
                await ws.send_json(payload)

            if final:
                print(f"[ws:{sid}] final flush complete - closing")
                await ws.close()

        try:
            # 2) receive loop
            while True:
                text = await ws.receive_text()
                # print(f"[ws:{sid}] got text: {text!r}")

                if text == "__END__":
                    print(f"[ws:{sid}] END sentinel received")
                    await flush(final=True)
                    break

                # Track when first text is received for end-to-end latency measurement (if timing enabled)
                if self.websocket_connections[sid]["include_timing_info"] and self.websocket_connections[sid]["first_text_time"] is None:
                    self.websocket_connections[sid]["first_text_time"] = time.time()
                    print(f"[ws:{sid}] First text received at {time.time()}")

                # append to your cached buffer
                self.websocket_connections[sid]["text_buffer"].extend(
                    text.strip().split()
                )
                current = len(self.websocket_connections[sid]["text_buffer"])

                # flush in chunks
                await flush()

        except WebSocketDisconnect:
            print(f"[ws:{sid}] disconnected unexpectedly - final flush")
            await flush(final=True)

        finally:
            print(f"[ws:{sid}] handler exit, clearing cache")
            # optionally inspect or persist your cache here:
            # cached_words = self.websocket_connections[sid]["text_buffer"]
            del self.websocket_connections[sid]
