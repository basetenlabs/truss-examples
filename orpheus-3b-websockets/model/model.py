import asyncio
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator, List

import batched
import numpy as np
import pysbd
import torch
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from snac import SNAC
from transformers import AutoTokenizer

# force inference mode during the lifetime of the script
_inference_mode_raii_guard = torch._C._InferenceMode(True)
# torch.backends.cuda.matmul.allow_tf32 = True

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
SNAC_DEVICE = "cuda"
SNAC_MAX_BATCH = 64
PREPROCESS_STREAM = torch.Stream(SNAC_DEVICE)


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


async def tokens_decoder(token_gen: Iterator, request_id: str = "") -> Iterator[bytes]:
    """Decoder that pipelines convert_to_audio calls but enforces strict in-order yields."""
    assert hasattr(token_gen, "__aiter__")
    audio_queue = asyncio.Queue()

    async def producer(token_gen: Iterator):
        buffer: list[int] = []
        count = 0
        async for token_sim in token_gen:
            for tok_str in split_custom_tokens(token_sim):
                token = turn_token_into_id(int(tok_str), count)
                buffer.append(token)
                count += 1
                # every 7 tokens → one frame; once we have at least 28 tokens, we extract the last 28
                if count % 7 == 0 and count > 27:
                    buf_to_proc = buffer[-28:]
                    task = asyncio.create_task(convert_to_audio(buf_to_proc))
                    audio_queue.put_nowait(task)
        audio_queue.put_nowait(None)
        print(f"Finished `{request_id}`, total tokens : {count}")

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


@torch.inference_mode()
async def convert_to_audio(frame_ids: list[int]) -> bytes | None:
    """Convert a list of token IDs into audio bytes efficiently.

    frame_ids:
    - list of token IDS (phonemes) of length 28 or less.
    - 7 tokens = 1 frame
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
    audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
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
        # satisfy Truss’s metrics/cancellation wrapper
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
        print(f" → voice={voice}, buffer_size={buf_sz}")

        # initialize per-sid state
        self.websocket_connections[sid] = {
            "text_buffer": [],  # this is your cache
            # you could add more, e.g. "audio_buffer": []
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
            async for audio in tokens_decoder(tokgen, sid):
                sent += len(audio)
                # print(f"[ws:{sid}] sending {len(audio)} bytes (total {sent})")
                await ws.send_bytes(audio)

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
