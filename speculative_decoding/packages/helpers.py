"""This file should be converted into a library in a truss/modelling package."""

import collections
import contextlib
import csv
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import MutableMapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pydantic
import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http
import tritonclient.utils as triton_utils
from numpy.typing import DTypeLike, NDArray

# Timing.
########################################################################################


class _Timing(object):
    """Oject to collect code section timing statistics."""

    _values: collections.deque
    _count: int
    _sum: int
    _t0: float

    def __init__(self):
        self._values = collections.deque(maxlen=50)
        self._count = 0
        self._sum = 0
        self._t0 = 0.0

    def update(self, value: Union[float, int]):
        self._values.append(value)
        self._count += 1
        self._sum += value

    def reset(self):
        self.__init__()

    def start_time(self):
        if self._t0 > 0.0:
            raise RuntimeError(
                "Section timeing for must be completed with `record_time` "
                "before a new timing can be started."
            )
        self._t0 = time.time()

    def record_time(self):
        t = time.time() - self._t0
        self.update(t)
        self._t0 = 0.0

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
    """Object to manage collection of `_Timing` objects."""

    _timings: MutableMapping[str, _Timing]
    _order: list[str]
    _total_section_name: Optional[str]
    _enabled: bool

    def __init__(self):
        self._timings = collections.defaultdict(_Timing)
        self._order = []
        self._total_section_name = None
        self._enabled = False

    def start_time(self, section_name: str):
        if section_name not in self._timings:
            self._order.append(section_name)
        self._timings[section_name].start_time()

    def record_time(self, section_name: str):
        self._timings[section_name].record_time()

    def set_total_section_name(self, section_name: str):
        self._total_section_name = section_name

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def to_pd(self) -> pd.DataFrame:
        rows = []
        section_key = "section"
        columns = [section_key, "Σ", "μ", "N", "it/s", "min", "max"]
        for n in self._order:
            t_dict = self._timings[n].to_dict()
            t_dict[section_key] = n
            rows.append(t_dict)

        if self._total_section_name:
            columns.append("%")
            total = self._timings[self._total_section_name]
            for i, n in enumerate(self._order):
                rows[i]["%"] = self._timings[n].to_dict()["Σ"] / total.to_dict()["Σ"]

        return pd.DataFrame(data=rows, columns=columns).set_index(section_key)

    def reset(self):
        self._timings.clear()
        self._order.clear()
        self._total_section_name = None


_global_metrics = _TimingManager()


@contextlib.contextmanager
def timeit(section_name: str, skip: bool = False):
    if _global_metrics.is_enabled() and not skip:
        try:
            _global_metrics.start_time(section_name)
            yield
        finally:
            _global_metrics.record_time(section_name)
    else:
        yield


def enable_timing() -> None:
    _global_metrics.enable()


def disable_timing() -> None:
    _global_metrics.disable()


def set_total_timing_section(section_name: str) -> None:
    """Designates a section as total E2E, other sections are shown with percentages."""
    _global_metrics.set_total_section_name(section_name)


def reset_timings():
    _global_metrics.reset()


def show_timings():
    print(_global_metrics.to_pd())


# Request API.
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
    # Note: embedding_bias, prompt_embedding_table, prompt_vocab_size, loras are not
    # integrated yet due to lack of precedent usage.
    prompt: str
    max_num_generated_tokens: int
    streaming: bool = True
    bad_word_list: Sequence[str] | None = None
    stop_words_list: Sequence[str] | None = None
    sampling_config: SamplingConfig = SamplingConfig()
    num_draft_tokens: int | None = None


# Triton inference client helpers.
########################################################################################


def _fill_inputs(
    name: str,
    input_data,
    dtype: DTypeLike,
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
    bad_words_ids: NDArray[np.int32] | None = None,
    stop_words_ids: NDArray[np.int32] | None = None,
) -> list[triton_grpc.InferInput]:
    inputs: list[triton_grpc.InferInput] = []
    _fill_inputs("input_ids", input_ids, np.int32, inputs)
    _fill_inputs("input_lengths", len(input_ids), np.int32, inputs)
    _fill_inputs("request_output_len", max_num_generated_tokens, np.uint32, inputs)
    # All below are optional inputs.
    _fill_inputs("draft_input_ids", draft_tokens, np.int32, inputs)
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


def extract_trtllm_outputs(result: triton_grpc.InferResult) -> NDArray[np.int32]:
    # Get batch 0, beam 0 output_ids
    output_ids = np.squeeze(result.as_numpy("output_ids").astype(np.int32), axis=(0, 1))
    sequence_len = int(
        np.squeeze(result.as_numpy("sequence_length").astype(np.int32), axis=(0, 1))
    )
    assert sequence_len == len(output_ids), f"{sequence_len} vs. {len(output_ids)}"
    return output_ids


def to_word_list_format(
    word_dict: list[list[str]], tokenizer=None, add_special_tokens=False
) -> NDArray[np.int32]:
    """
    Taken from
    https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/runtime/generation.py

    - do not edit / maintain.

    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    """
    assert tokenizer != None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word, add_special_tokens=add_special_tokens)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


# Triton inference server helpers.
########################################################################################

GRPC_SERVICE_PORT = 8001
HTTP_SERVICE_PORT = 8003


def is_triton_server_alive() -> bool:
    def port_is_available(port):
        available = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                available = True
            except OSError:
                pass
        return available

    return not port_is_available(GRPC_SERVICE_PORT) or not port_is_available(
        HTTP_SERVICE_PORT
    )


class TritonServer:
    _process: subprocess.Popen | None

    def __init__(self, model_repository_dir: Path, parallel_count=1):
        self._model_repository_dir: Path = model_repository_dir
        self._parallel_count = parallel_count
        self._process = None

    def load_server_and_model(self, env: dict) -> None:
        """Loads the Triton server and the model."""
        if not is_triton_server_alive():
            self._process = self._start_server(mpi=self._parallel_count, env=env)

        http_client = triton_http.InferenceServerClient(
            url=f"localhost:{HTTP_SERVICE_PORT}", verbose=False
        )

        is_server_up = False
        while not is_server_up:
            try:
                is_server_up = http_client.is_server_live()
            except ConnectionRefusedError:
                time.sleep(0.2)
                continue

        while not http_client.is_server_ready():
            time.sleep(0.2)
            continue

    def build_server_start_command(self, mpi: int = 1, env: dict = {}) -> list:
        base_command = [
            "tritonserver",
            "--model-repository",
            str(self._model_repository_dir),
            "--grpc-port",
            str(GRPC_SERVICE_PORT),
            "--http-port",
            str(HTTP_SERVICE_PORT),
        ]
        if mpi == 1:
            return base_command

        mpirun_command = ["mpirun", "--allow-run-as-root"]
        # Generate mpi_commands with a unique shm-region-prefix-name for each MPI process
        mpi_commands = []
        for i in range(mpi):
            mpi_command = [
                "-n",
                "1",
                *base_command,
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{str(i)}_",
            ]
            mpi_commands.append(" ".join(mpi_command))

        # Join the individual mpi commands with ' : ' as required by mpirun
        # syntax for multiple commands.
        combined_mpi_commands = " : ".join(mpi_commands)
        return mpirun_command + [combined_mpi_commands]

    def _start_server(self, mpi: int = 1, env: dict = {}) -> subprocess.Popen:
        """Triton Inference Server has different startup commands depending on
        whether it is running in a TP=1 or TP>1 configuration. This function
        starts the server with the appropriate command."""
        command = self.build_server_start_command(mpi, env)
        return subprocess.Popen(command, env={**os.environ, **env})

    def shutdown(self) -> None:
        if self._process:
            self._process.terminate()
