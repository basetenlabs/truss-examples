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

# force inference mode during the lifetime of the script
inference_mode_raii_guard = torch._C._InferenceMode(True)

model = SNAC.from_pretrained("/app/snac_24khz").eval()
snac_device = "cuda"
model = model.to(snac_device)
# TODO(veer/michael): test decoder with bfloat16
dtype_decoder = torch.float32
# model.decoder = model.decoder.to(dtype_decoder)


def turn_token_into_id(token_string, index):
    """Extract and convert the last custom token ID from a string."""
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")

    if last_token_start == -1:
        print(f"No token found in the string '{token_string}' (terminatio of audio?)")
        return None

    last_token = token_string[last_token_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


async def tokens_decoder(token_gen: Iterator):
    """Corrected to handle both async and sync iterables."""
    buffer = []
    count = 0
    # Check if token_gen is an async iterable; if not, iterate synchronously.
    if hasattr(token_gen, "__aiter__"):
        async for token_sim in token_gen:
            # TODO(veer/michael): check if token_sim can be at most 1 token (e.g. via token_sim.split("<"))
            num_tokens = token_sim.count("<custom_token_")
            if num_tokens > 1:
                print(f"WARNING: Token string '{token_sim}' has more than one token.")
            token = turn_token_into_id(token_sim, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = await convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples
    else:
        for token_sim in token_gen:
            token = turn_token_into_id(token_sim, count)
            if token is not None and token > 0:
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


@batched.dynamically(batch_size=128, timeout_ms=2)
def batch_snac_model(items: list[dict[str, list[torch.Tensor]]]) -> list[torch.Tensor]:
    # Custom processing logic here
    # return [model.decode(item["codes"]) for item in items]
    with torch.inference_mode():
        stacked_z_q = torch.cat(  # codes is list[torch.Tensor]
            [model.quantizer.from_codes(codes["codes"]) for codes in items], dim=0
        )
        output_batched = model.decoder(stacked_z_q.to(dtype_decoder)).to(torch.float32)
        return output_batched.split(
            1, dim=0
        )  # unbatch the output into len(items) tensors of shape (1, 1, 32768)


@torch.inference_mode()
async def convert_to_audio(multiframe, count):
    """Convert a list of token IDs into audio bytes efficiently."""
    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    codes_0 = torch.zeros(num_frames, device=snac_device, dtype=torch.int32)
    codes_1 = torch.zeros(2 * num_frames, device=snac_device, dtype=torch.int32)
    codes_2 = torch.zeros(4 * num_frames, device=snac_device, dtype=torch.int32)

    for j in range(num_frames):
        i = 7 * j
        codes_0[j] = frame[i]
        codes_1[2 * j] = frame[i + 1]
        codes_1[2 * j + 1] = frame[i + 4]
        codes_2[4 * j] = frame[i + 2]
        codes_2[4 * j + 1] = frame[i + 3]
        codes_2[4 * j + 2] = frame[i + 5]
        codes_2[4 * j + 3] = frame[i + 6]

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    if (
        torch.any(codes[0] < 0)
        or torch.any(codes[0] > 4096)
        or torch.any(codes[1] < 0)
        or torch.any(codes[1] > 4096)
        or torch.any(codes[2] < 0)
        or torch.any(codes[2] > 4096)
    ):
        return None

    audio_hat = await batch_snac_model.acall({"codes": codes})

    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
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
            prompt_tokens = self._tokenizer(adapted_prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor(
                [[128009, 128260, 128261, 128257]], dtype=torch.int64
            )
            all_input_ids = torch.cat(
                [start_token, prompt_tokens.input_ids, end_tokens], dim=1
            )
            return self._tokenizer.decode(all_input_ids[0])
        else:
            prompt_tokens = self._tokenizer(prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor(
                [[128009, 128260, 128261, 128257]], dtype=torch.int64
            )
            all_input_ids = torch.cat(
                [start_token, prompt_tokens.input_ids, end_tokens], dim=1
            )
            return self._tokenizer.decode(all_input_ids[0])

    async def predict(
        self, model_input: Any, request: fastapi.Request
    ) -> StreamingResponse:
        print("This is the custom predict function")
        model_input["prompt"] = self._format_prompt(
            model_input["prompt"], voice=model_input.get("voice", "tara")
        )
        model_input["temperature"] = model_input.get("temperature", 0.6)
        model_input["top_p"] = model_input.get("top_p", 0.8)
        model_input["max_tokens"] = model_input.get("max_tokens", 10000)
        if model_input.get("end_id") is not None:
            return fastapi.responses.JSONResponse(
                content={
                    "error": "end_id is not supported in this model, set to 128258"
                }
            )
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
