import io
import logging
import re
import zlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorrt_llm
import torch
import torchaudio
from torch import Tensor
from whisper_trt.assets import download_assets, download_engine
from whisper_trt.batching import WhisperBatchProcessor
from whisper_trt.modeling import WhisperDecoding, WhisperEncoding
from whisper_trt.tokenizer import (
    END_OF_TEXT,
    LANG_TO_CODE,
    LANGUAGES,
    NO_SPEECH,
    NO_TIMESTAMPS,
    START,
    START_OF_LM,
    START_PREV,
    TRANSCRIBE,
    TRANSLATE,
    get_tokenizer,
)
from whisper_trt.types import (
    SUPPORTED_SAMPLE_RATE,
    BatchWhisperItem,
    Segment,
    WhisperResult,
)
from whisper_trt.utils import log_mel_spectrogram, pad_or_trim

SEGMENTS_PATTERN = re.compile(r"<\|([\d.]+)\|>([^<]+)(?:<\|([\d.]+)\|>)?")
LANG_CODE_PATTERN = re.compile(r"<\|([a-z]{2,3})\|>")
CLEANUP_PATTERN = re.compile(
    "|".join(
        [
            e.replace("|", "\|")
            for e in [
                END_OF_TEXT,
                NO_SPEECH,
                START,
                START_PREV,
                NO_TIMESTAMPS,
                TRANSCRIBE,
                TRANSLATE,
                START_OF_LM,
            ]
            + [f"<|{ll}|>" for ll in LANGUAGES.keys()]
        ]
    )
)


def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


class WhisperModel:
    # ## Public API
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs) -> "WhisperModel":
        # Download the appropriate engine based on the model type and gpu type
        engine_dir = download_engine(model_name)

        # Create and return a valid WhisperModel instance
        return cls(engine_dir, **kwargs)

    async def translate(
        self,
        waveform: Union[str, np.ndarray, torch.Tensor],
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        prefix: Optional[str] = None,
        max_new_tokens=128,
        raise_when_trimmed: bool = False,
    ) -> WhisperResult:
        return await self.generate(
            waveform,
            task="translate",
            prompt=prompt,
            language=language,
            prefix=prefix,
            max_new_tokens=max_new_tokens,
            raise_when_trimmed=raise_when_trimmed,
        )

    async def transcribe(
        self,
        waveform: Union[str, np.ndarray, torch.Tensor],
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        prefix: Optional[str] = None,
        max_new_tokens=128,
        raise_when_trimmed: bool = False,
    ) -> WhisperResult:
        return await self.generate(
            waveform,
            task="transcribe",
            prompt=prompt,
            language=language,
            prefix=prefix,
            max_new_tokens=max_new_tokens,
            raise_when_trimmed=raise_when_trimmed,
        )

    def preprocess_audio(self, binary_data: bytes | Tensor) -> Tensor:
        if not isinstance(binary_data, Tensor):
            audio_stream = io.BytesIO(binary_data)
            waveform, sample_rate = torchaudio.load(audio_stream)
        else:
            waveform, sample_rate = binary_data, 16_000
        # Resample audio to rate compatible with what the model was trained at
        if sample_rate != SUPPORTED_SAMPLE_RATE:
            logging.info("Resampling audio to 16kHz")
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=SUPPORTED_SAMPLE_RATE
            )
        # waveform = pad_or_trim(waveform)
        return waveform

    ## Internal API
    def __init__(
        self,
        engine_dir,
        num_beams=1,
        tokenizer_name="multilingual",
        debug_mode=False,
        assets_dir=None,
        max_queue_time=0.01,  # 10 ms by default
    ):
        self._assets_dir = assets_dir
        if self._assets_dir is None:
            self._assets_dir = download_assets()

        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        engine_dir = Path(engine_dir)

        ## Load encoder
        self._encoder = WhisperEncoding(engine_dir)
        self._n_mels = self._encoder.n_mels
        self._num_beams = num_beams

        ## Load Tokenizer
        self._tokenizer = get_tokenizer(
            name=tokenizer_name,
            num_languages=self._encoder.num_languages,
            tokenizer_dir=self._assets_dir,
        )

        ## Resolve relevant special tokens
        lang_tokens = [
            f"<|{lang}|>"
            for lang in list(LANGUAGES.keys())[: self._encoder.num_languages]
        ]
        lang_token_ids = sum(
            self._tokenizer.encode_batch(
                lang_tokens, allowed_special=self._tokenizer.special_tokens_set
            ),
            [],
        )

        def get_token_id(token_str: str):
            return self._tokenizer.encode(
                token_str, allowed_special=self._tokenizer.special_tokens_set
            )[0]

        self._eot_id = get_token_id(END_OF_TEXT)
        # self._pad_id = self._eot_id
        WHISPER_PAD_TOKEN_ID = 50256
        self._pad_id = WHISPER_PAD_TOKEN_ID  # self._eot_id
        self._pad_token = WHISPER_PAD_TOKEN_ID  # self._eot_id # TODO: confirm Justin why his pad token is different
        # self._sot_id = get_token_id(START)

        # self._no_speech_id = get_token_id(NO_SPEECH)
        self._sot_id = get_token_id(START)

        self._no_speech_id = get_token_id(NO_SPEECH)
        self._notimestamps_id = get_token_id(NO_TIMESTAMPS)

        ## initlize decoder
        self._decoder = WhisperDecoding(
            engine_dir,
            runtime_mapping,
            lang_token_ids=lang_token_ids,
            eot_id=self._eot_id,
            sot_id=self._sot_id,
            no_speech_id=self._no_speech_id,
            pad_id=self._pad_id,
            notimestamps_id=self._notimestamps_id,
            debug_mode=debug_mode,
        )
        self._batch_size = self._decoder.decoder_config["max_batch_size"]

        # We are using a single batch processor then will batch encode and decode (twice, once for language detection and once for transcription)
        # for a single batch instead of including an additional delay.
        # No lock is required since the batcher is set to only run one batch at a time, which effectively locks
        # around all the gpu/trt-llm interactions.
        self._batch_processor = WhisperBatchProcessor(
            self,
            max_batch_size=self._batch_size,
            max_queue_time=max_queue_time,
            concurrency=1,
        )

    def _get_text_prefix(
        self,
        detected_language_token: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        task: str = "transcribe",
        prefix: Optional[str] = None,
    ) -> Optional[str]:
        if detected_language_token == NO_SPEECH:
            # None indicates that we do not want to decode this with the batch
            return None
        if language is None:
            language_token = detected_language_token
        else:
            try:
                language_token = f"<|{LANG_TO_CODE[language]}|>"
            except KeyError:
                language_token = f"<|{language}|>"

        text_prefix = f"{START}{language_token}<|{task}|>"
        if prompt is not None:
            text_prefix = f"{START_PREV} {prompt.strip()}" + text_prefix

        # TODO: we are currently not adding a timestamp token and forcing
        #       the logits processor to generate it. Soon, we should pass it in
        #       and then simplify the logits processor.
        # text_prefix += ZERO_TIMESTAMP

        if prefix is not None:
            text_prefix += prefix
        return text_prefix

    async def generate(
        self,
        waveform: Union[str, np.ndarray, torch.Tensor],
        is_mel: bool = False,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        prefix: Optional[str] = None,
        task: str = "transcribe",
        max_new_tokens=512,
        raise_when_trimmed: bool = False,
    ) -> WhisperResult:
        if not is_mel:
            mel = log_mel_spectrogram(
                waveform,
                self._n_mels,
                device="cuda",
                mel_filters_dir=self._assets_dir,
                raise_when_trimmed=raise_when_trimmed,
            )
        else:
            mel = waveform.to("cuda")
        mel = mel.type(torch.float16)

        generated_text, avg_log_prob = await self._batch_processor.process(
            item=BatchWhisperItem(
                mel=mel,
                prompt=prompt,
                prefix=prefix,
                task=task,
                max_new_tokens=max_new_tokens,
                duration_secs=(len(waveform[0]) / SUPPORTED_SAMPLE_RATE),
            ),
        )

        if generated_text is None:
            # No speech was detected. Return empty segments.
            return WhisperResult(segments=[], language=None, language_code=None)
        return self._postprocess_transcript(generated_text, language, avg_log_prob)

    def _postprocess_transcript(
        self, generated_text: str, language: Optional[str], avg_log_prob: float
    ) -> WhisperResult:
        language_matches = LANG_CODE_PATTERN.findall(generated_text)
        language_code = language_matches[0] if language_matches else None
        # TODO: add test to verify that regex doesn't drop last segment when last timestamp is missing.
        segment_matches = SEGMENTS_PATTERN.findall(generated_text)
        # Try not parsing segment
        no_special_generated_text = CLEANUP_PATTERN.sub("", generated_text)

        segments = []
        for segment_match in segment_matches:
            start, text, end = segment_match
            segments.append(
                Segment(
                    start_time_sec=float(start) if start else None,
                    end_time_sec=float(end) if end else None,
                    text=text.strip(),
                )
            )

        if len(segments) == 0:
            return WhisperResult(
                avg_log_prob=avg_log_prob,
                compression_ratio=get_compression_ratio(no_special_generated_text),
                segments=[
                    Segment(
                        start_time_sec=0,
                        end_time_sec=None,
                        text=no_special_generated_text,
                    )
                ],
                language=language,
                language_code=language_code,
            )

        return WhisperResult(
            segments=segments,
            language=language,
            language_code=language_code,
            avg_log_prob=avg_log_prob,
            compression_ratio=get_compression_ratio(no_special_generated_text),
        )
