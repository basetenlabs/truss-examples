import collections
import contextlib
import time
from typing import Sequence, Union

import numpy as np
import pandas as pd
import pydantic
import tritonclient.grpc as triton_grpc
import tritonclient.utils as triton_utils


class _Timing(object):
    """Accumulator that collects statistics (esp. execution time) over repeated updates and offers pretty printing."""

    def __init__(self):
        self._values = collections.deque(maxlen=50)
        self._count = 0
        self._sum = 0
        self._t0 = 0

    def update(self, value: Union[float, int]):
        self._values.append(value)
        self._count += 1
        self._sum += value

    def reset(self):
        self.__init__()

    def start_time(self):
        assert self._t0 == 0
        self._t0 = time.time()

    def record_time(self):
        t = time.time() - self._t0
        self.update(t)
        self._t0 = 0

    def __str__(self) -> str:
        vals = np.array(self._values)
        avg = self._sum / self._count
        its = 1 / avg
        return (
            f"Σ={self._sum:.3g} ({its:.3g}/s)\tμ={avg:.3g}\tmin={vals.min():.3g}\t"
            f"max={vals.max():.3g}\tσ/μ={vals.var() / avg:.3g}\tN={self._count}"
        )

    def to_dict(self) -> dict[str, float | int | str]:
        vals = np.array(self._values)
        avg = self._sum / self._count
        its = 1 / avg
        return {
            "Σ": self._sum,
            "it/s": its,
            "μ": avg,
            "N": self._count,
            "min": vals.min(),
            "max": vals.max(),
        }


class _TimingManager(object):
    """Container to manage a set of global metrics for execution times."""

    def __init__(self):
        self._timings = collections.defaultdict(_Timing)
        self._order = []
        self._total = None

    def start_time(self, name: str):
        if name not in self._timings:
            self._order.append(name)
        self._timings[name].start_time()

    def record_time(self, name: str):
        self._timings[name].record_time()

    def __str__(self):
        parts = ["Global Wall Time Totals:"]
        for n in self._order:
            m = self._timings[n]
            if self._total:
                t = self._timings[self._total].sum
                fraction = f"    {m.sum / t * 100:2.0f}% "
            else:
                fraction = ""

            parts.append(f"{n:<20}: {fraction}{m}")
        return "\n".join(parts)

    def to_pd(self) -> pd.DataFrame:
        rows = []
        section_key = "section"
        columns = [section_key, "Σ", "μ", "N", "it/s", "min", "max"]
        for n in self._order:
            t_dict = self._timings[n].to_dict()
            t_dict[section_key] = n
            rows.append(t_dict)
        return pd.DataFrame(data=rows, columns=columns).set_index(section_key)

    def reset(self):
        self.__init__()

    def set_total(self, name: str):
        self._total = name


_global_metrics = _TimingManager()


@contextlib.contextmanager
def timeit(name: str):
    try:
        _global_metrics.start_time(name)
        yield
    finally:
        _global_metrics.record_time(name)


def show_timings():
    print(_global_metrics.to_pd())


########################################################################################


class SamplingConfig(pydantic.BaseModel):
    # Field order follows
    # https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/inference_request.md
    beam_width: int | None = None
    temperature: float | None = None
    runtime_top_k: int | None = None
    runtime_top_p: float | None = None
    len_penalty: float | None = None
    repetition_penalty: float | None = None
    min_len: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    random_seed: int | None = None


class GenerationRequest(pydantic.BaseModel):
    # TODO: embedding_bias, prompt_embedding_table, prompt_vocab_size, lora
    prompt: str
    max_num_generated_tokens: int
    request_id: str
    streaming: bool = False
    bad_word_list: Sequence[str] | None = None
    stop_words_list: Sequence[str] | None = None
    sampling_config: SamplingConfig = SamplingConfig()


def _fill_inputs(
    name: str,
    input_data,
    dtype: np.dtype,
    mutable_inputs: list[triton_grpc.InferInput],
    make_2d: bool = True,
) -> None:
    if input_data is None:
        return

    array_input = np.asarray(input_data, dtype=dtype)
    if make_2d:
        array_input = np.atleast_2d(array_input)
    t = triton_grpc.InferInput(
        name, array_input.shape, triton_utils.np_to_triton_dtype(dtype)
    )
    t.set_data_from_numpy(array_input)
    mutable_inputs.append(t)


_SAMPLIG_CONFIG_DTYPES = {
    "beam_width": np.uint32,
    "temperature": np.float32,
    "runtime_top_k": np.uint32,
    "runtime_top_p": np.float32,
    "len_penalty": np.float32,
    "repetition_penalty": np.float32,
    "min_len": np.float32,
    "presence_penalty": np.float32,
    "frequency_penalty": np.float32,
    "random_seed": np.uint64,
}


def make_trtllm_inputs(
    input_ids: Sequence[int],
    max_num_generated_tokens: int,
    sampling_config: SamplingConfig | None,
    draft_tokens: Sequence[int] | None = None,
    end_id: int | None = None,
    pad_id: int | None = None,
    bad_words_ids: np.ndarray[int] | None = None,
    stop_words_ids: np.ndarray[int] | None = None,
) -> list[triton_grpc.InferInput]:
    input_length = len(input_ids)
    inputs = []
    # Add batch dimension.
    _fill_inputs("input_ids", input_ids, np.int32, inputs)
    _fill_inputs("input_lengths", input_length, np.int32, inputs)
    _fill_inputs("request_output_len", max_num_generated_tokens, np.uint32, inputs)

    # All below are optional inputs.
    _fill_inputs("draft_input_ids", draft_tokens, np.int32, inputs)
    # Generation.
    _fill_inputs("end_id", end_id, np.uint32, inputs)
    _fill_inputs("pad_id", pad_id, np.uint32, inputs)
    _fill_inputs("bad_words_list", bad_words_ids, np.int32, inputs)
    _fill_inputs("stop_words_list", stop_words_ids, np.int32, inputs)
    if sampling_config:
        for key, dtype in _SAMPLIG_CONFIG_DTYPES.items():
            _fill_inputs(key, getattr(sampling_config, key), dtype, inputs)
    # _fill_inputs("return_log_probs", True, bool, inputs)
    # The return_X_logits tensor names were only added in
    # https://github.com/NVIDIA/TensorRT-LLM/pull/846.
    # "return_context_logits", "return_generation_logits"
    return inputs


def extract_trtllm_outputs(result: triton_grpc.InferResult) -> np.ndarray[np.int32]:
    # TODO: Get context_logits, generation_logits and find out why output_log_probs is
    #  always zero.
    # Get batch 0, beam 0 output_ids
    output_ids = np.squeeze(result.as_numpy("output_ids").astype(np.int32), axis=(0, 1))
    sequence_len = int(
        np.squeeze(result.as_numpy("sequence_length").astype(np.int32), axis=(0, 1))
    )
    assert sequence_len == len(output_ids), f"{sequence_len} vs. {len(output_ids)}"
    return output_ids
