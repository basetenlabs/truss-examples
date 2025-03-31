import logging
import os
import torch
import struct
from fastapi.responses import StreamingResponse
import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from typing import Optional
from transformers import AutoTokenizer
from snac import SNAC

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda"
model = model.to(snac_device)

logger = logging.getLogger(__name__)

def convert_to_audio(multiframe, count):
    """Convert a list of token IDs into audio bytes efficiently."""
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

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

    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    with torch.inference_mode():
        audio_hat = model.decode(codes)
    
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def turn_token_into_id(token_string, index):
    """Extract and convert the last custom token ID from a string."""
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        print("No token found in the string")
        return None
    
    last_token = token_string[last_token_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None
async def tokens_decoder(token_gen):
    """Async generator to decode tokens into audio chunks."""
    buffer = []
    count = 0
    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

    # after the stream ends, yield any remaining tokens if buffer has leftovers
    if count > 27:
        remaining = buffer[-28:]
    else:
        remaining = buffer

    if remaining:
        audio_samples = convert_to_audio(remaining, count)
        if audio_samples is not None:
            yield audio_samples

class OrpheusModel:
    def __init__(self, 
                 model_name, 
                 dtype=torch.bfloat16,
                 seed: int = 0,
                 max_model_len: Optional[int] = None,
                 cpu_offload_gb: float = 0,
                 gpu_memory_utilization: float = 0.90,
                 quantization: Optional[str] = None,
                 max_seq_len_to_capture: int = 8192,
                 enforce_eager: Optional[bool] = None):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.engine = self._setup_engine(seed, max_model_len, cpu_offload_gb, 
                                        gpu_memory_utilization, quantization, 
                                        max_seq_len_to_capture, enforce_eager)
        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)

    def _map_model_params(self, model_name):
        model_map = {
            "medium-3b": {"repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod"},
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if model_name in unsupported_models:
            raise ValueError(f"Model {model_name} is not supported.")
        return model_map.get(model_name, {"repo_id": model_name})["repo_id"]

    def _setup_engine(self, seed, max_model_len, cpu_offload_gb, gpu_memory_utilization, 
                      quantization, max_seq_len_to_capture, enforce_eager):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            max_model_len=max_model_len,
            cpu_offload_gb=cpu_offload_gb,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            max_seq_len_to_capture=max_seq_len_to_capture,
            enforce_eager=enforce_eager,
            seed=seed
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

    def _format_prompt(self, prompt, voice="tara"):
        if voice:
            adapted_prompt = f"{voice}: {prompt}"
            prompt_tokens = self.tokeniser(adapted_prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            return self.tokeniser.decode(all_input_ids[0])
        else:
            prompt_tokens = self.tokeniser(prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            return self.tokeniser.decode(all_input_ids[0])

    async def generate_tokens(self, prompt, voice=None, request_id="req-001", 
                              temperature=0.6, top_p=0.8, max_tokens=1200, 
                              stop_token_ids=[128258], repetition_penalty=1.3):
        prompt_string = self._format_prompt(prompt, voice)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )
        async for result in self.engine.generate(prompt=prompt_string, 
                                                 sampling_params=sampling_params, 
                                                 request_id=request_id):
            yield result.outputs[0].text

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.model = None
        self._secrets = kwargs["secrets"]
        os.environ["HF_TOKEN"] = self._secrets["hf_access_token"]

    def load(self):
        self.model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod", 
                                  dtype=torch.float16)


    def create_wav_header(self, sample_rate=24000, bits_per_sample=16, channels=1):
        """Creates a WAV file header."""
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = 0
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, channels,
            sample_rate, byte_rate, block_align, bits_per_sample, b'data', data_size
        )
        return header

    async def predict(self, model_input):
        """Async predict method to stream audio for concurrent requests."""
        text = str(model_input.get("text", "Hi, I'm Orpheus model"))
        voice = str(model_input.get("voice", "tara"))
        request_id = str(model_input.get("request_id", "req-001"))
        repetition_penalty = model_input.get("repetition_penalty", 1.1)
        max_tokens = int(model_input.get("max_tokens", 10000))
        temperature = model_input.get("temperature", 0.4)
        top_p = model_input.get("top_p", 0.9)

        logger.info(f"Generating audio from processed text ({len(text)} chars, voice {voice}): {text}")

        async def audio_stream():
            yield self.create_wav_header()
            token_gen = self.model.generate_tokens(
                prompt=text,
                voice=voice,
                request_id=request_id,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=[128258],
            )
            async for audio_chunk in tokens_decoder(token_gen):
                yield audio_chunk

        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav"
        )