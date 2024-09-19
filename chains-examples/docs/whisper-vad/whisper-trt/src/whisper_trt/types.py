import pydantic
from torch import Tensor
from typing import NamedTuple, Optional

SUPPORTED_SAMPLE_RATE = 16_000
DEFAULT_MAX_NEW_TOKENS = 128


class BatchWhisperItem(NamedTuple):
    mel: Tensor
    prompt: Optional[str] = (None,)
    task: str = ("transcribe",)
    prefix: Optional[str] = (None,)
    language: Optional[str] = None
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    duration_secs: Optional[float] = None


class Segment(pydantic.BaseModel):
    # start and end times are ommited in the case of no timestamps.
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    text: str


class WhisperResult(pydantic.BaseModel):
    avg_log_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    segments: list[Segment]
    language: Optional[str] = None
    language_code: Optional[str] = pydantic.Field(
        ...,
        description="IETF language tag, e.g. 'en', see. "
        "https://en.wikipedia.org/wiki/IETF_language_tag.",
    )
