from typing import Any, Iterator
from transformers import AutoTokenizer
import torch
import fastapi
from snac import SNAC
import struct
from pathlib import Path
import numpy as np
from fastapi.responses import StreamingResponse
import batched
import re
from typing import List
import time

# force inference mode during the lifetime of the script
_inference_mode_raii_guard = torch._C._InferenceMode(True)

# TODO(veer/michael): test decoder with bfloat16
snac_device = "cuda"

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
snac_device = "cuda"
snac_max_batch_size = 64


class SnacModelBatched:
    def __init__(self):
        self.dtype_decoder = torch.float32
        snac_torch_compile = True

        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        model = model.to(snac_device)

        model.decoder = model.decoder.to(self.dtype_decoder)
        if snac_torch_compile:
            model.decoder = torch.compile(model.decoder, dynamic=True)
            model.quantizer = torch.compile(model.quantizer, dynamic=True)
        t = time.time()
        for bs_size in range(1, max(snac_max_batch_size, 1)):
            codes = [
                torch.randint(1, 4096, (bs_size, 4)).to(snac_device),
                torch.randint(1, 4096, (bs_size, 8)).to(snac_device),
                torch.randint(1, 4096, (bs_size, 16)).to(snac_device),
            ]
            with torch.inference_mode():
                intermed = model.quantizer.from_codes(codes)
                model.decoder(intermed.to(self.dtype_decoder))
        print("time for torch.compile/warmup:", time.time() - t)
        self.snac_model = model
        self.stream = torch.Stream()

    @batched.dynamically(batch_size=64, timeout_ms=10)
    def batch_snac_model(
        self, items: list[dict[str, list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
        # Custom processing logic here
        # return [model.decode(item["codes"]) for item in items]
        if len(items) > 1:
            assert items[0]["codes"][0].shape == items[1]["codes"][0].shape, (
                f"items[0]['codes'][0].shape: {items[0]['codes'][0].shape}, items[1]['codes'][0].shape: {items[1]['codes'][0].shape}"
            )
            assert items[0]["codes"][1].shape == items[1]["codes"][1].shape
            assert items[0]["codes"][2].shape == items[1]["codes"][2].shape
            print(f"using batch size {len(items)}")
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            all_codes = [codes["codes"] for codes in items]
            # stacked_codes = [(b,4), (b,8), (b,16)]
            stacked_codes = [
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
            self.stream.synchronize()  # make sure the results are ready
            return out


model_snac = SnacModelBatched()
non_default_stream = torch.Stream(snac_device)


def turn_token_into_id(token_string: int, index: int):
    """Extract and convert the last custom token ID from a string."""
    return token_string - 10 - ((index % 7) * 4096)


def split_custom_tokens(s: str) -> List[int]:
    """
    Extracts all substrings enclosed in <custom_token_…> from the input string.
    """
    matches = _TOKEN_RE.findall(s)
    return [int(match) for match in matches if match != "0"]


async def tokens_decoder(token_gen: Iterator):
    """Corrected to handle both async and sync iterables."""
    buffer: list[int] = []
    count = 0
    # Check if token_gen is an async iterable; if not, iterate synchronously.
    assert hasattr(token_gen, "__aiter__")
    async for token_sim in token_gen:
        split_tokens = split_custom_tokens(token_sim)
        for token_string in split_tokens:
            token = turn_token_into_id(token_string, count)
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = await convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

    # After the stream ends, yield any remaining tokens if buffer has leftovers
    if count > 27:
        remaining = buffer[-28:]
    else:
        remaining = buffer

    if remaining:
        audio_samples = await convert_to_audio(remaining, count)
        if audio_samples is not None:
            yield audio_samples


@torch.inference_mode()
async def convert_to_audio(multiframe, count):
    """Convert a list of token IDs into audio bytes efficiently."""
    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    codes_0 = torch.zeros(num_frames, dtype=torch.int32)
    codes_1 = torch.zeros(2 * num_frames, dtype=torch.int32)
    codes_2 = torch.zeros(4 * num_frames, dtype=torch.int32)

    for j in range(num_frames):
        i = 7 * j
        codes_0[j] = frame[i]
        codes_1[2 * j] = frame[i + 1]
        codes_1[2 * j + 1] = frame[i + 4]
        codes_2[4 * j] = frame[i + 2]
        codes_2[4 * j + 1] = frame[i + 3]
        codes_2[4 * j + 2] = frame[i + 5]
        codes_2[4 * j + 3] = frame[i + 6]

    if (
        torch.any(codes_0 < 0)
        or torch.any(codes_0 > 4096)
        or torch.any(codes_1 < 0)
        or torch.any(codes_1 > 4096)
        or torch.any(codes_2 < 0)
        or torch.any(codes_2 > 4096)
    ):
        return None
    with torch.cuda.stream(non_default_stream):
        codes = [
            codes_0.unsqueeze(0).to(snac_device),
            codes_1.unsqueeze(0).to(snac_device),
            codes_2.unsqueeze(0).to(snac_device),
        ]
        non_default_stream.synchronize()  # only queue codes that are ready
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

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            Path(self._data_dir) / "tokenization"
        )

    def create_wav_header(self, sample_rate=24000, bits_per_sample=16, channels=1):
        """Create a WAV file header."""
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = 0
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header

    def _format_prompt(self, prompt, voice="tara"):
        if voice:
            adapted_prompt = f"{voice}: {prompt}"
        else:
            adapted_prompt = prompt
        # TODO: make this pure python lists
        input_ids = self._tokenizer.encode(
            adapted_prompt,
        )
        start_id = 128259
        end_ids = [128009, 128260, 128261, 128257]

        full_ids = [start_id] + input_ids + end_ids
        return self._tokenizer.decode(full_ids)

    async def predict(
        self, model_input: Any, request: fastapi.Request
    ) -> StreamingResponse:
        print("Starting new request")
        model_input["prompt"] = self._format_prompt(
            model_input["prompt"], voice=model_input.get("voice", "tara")
        )
        model_input["temperature"] = model_input.get("temperature", 0.6)
        model_input["top_p"] = model_input.get("top_p", 0.8)
        model_input["max_tokens"] = model_input.get("max_tokens", 10000)
        if model_input.get("end_id") is not None:
            print("Not using end_id from model_input:", model_input["end_id"])
        model_input["end_id"] = 128258
        # model_input["pad_id"] = model_input.get("end_id", [128004]) automatically infered  from AutoTokenizer.from_file(..).pad_token
        model_input["repetition_penalty"] = model_input.get("repetition_penalty", 1.3)

        async def audio_stream():
            yield self.create_wav_header()
            token_gen = await self._engine.predict(model_input, request)
            if isinstance(token_gen, StreamingResponse):
                token_gen = token_gen.body_iterator
            async for chunk in tokens_decoder(token_gen):
                yield chunk

        return StreamingResponse(audio_stream(), media_type="audio/wav")
