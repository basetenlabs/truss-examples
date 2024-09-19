import enum
from typing import Optional

import pydantic

# External Models ######################################################################


class MicroChunkInfo(pydantic.BaseModel):
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    chunk_index: Optional[int] = None


class WhisperParams(pydantic.BaseModel):
    prompt: Optional[str] = None
    language: Optional[str] = None
    prefix: Optional[str] = None
    task: str = "transcribe"
    max_new_tokens: int = 512
    raise_when_trimmed: bool = False


class WhisperInput(WhisperParams):
    audio_b64: str


## whisper_trt types ##################################################################
class Segment(pydantic.BaseModel):
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    text: str


class WhisperResult(pydantic.BaseModel):
    segments: list[Segment]
    language_code: Optional[str] = None


## API Types ##################################################################


class MacroChunkInfo(pydantic.BaseModel):
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    duration_sec: Optional[float] = None
    start_time_str: Optional[str] = None
    macro_chunk: Optional[int] = None
    is_last: Optional[bool] = False
    is_valid: Optional[bool] = False


class ChunkingParams(pydantic.BaseModel):
    chunk_size_sec: int = pydantic.Field(
        30,
        description="Each macro-chunk is split into micro-chunks. When using silence "
        "detection, this is the *maximal* size (i.e. an actual micro-chunk could be "
        "smaller): A point of minimal silence searched in the second half of the "
        "maximal micro chunk duration using the smoothed absolute waveform.",
    )
    silence_detection_smoothing_num_samples: int = pydantic.Field(
        800,
        description="Number of samples to determine width of box smoothing "
        "filter. With sampling of 16 kHz, 1600 samples is 1/10 second.",
    )


class AudioSource(pydantic.BaseModel):
    url: Optional[str] = None
    # TODO: convert this to `bytes` type
    audio_b64: Optional[str] = None


class InputSchema(pydantic.BaseModel):
    audio: AudioSource
    whisper_params: WhisperParams = WhisperParams()
