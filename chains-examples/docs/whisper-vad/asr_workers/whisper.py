# flake8: noqa F402
# This location assumes `fde`-repo is checked out at the same level as `truss`-repo.
from pathlib import Path

_LOCAL_WHISPER_LIB = Path(__file__).resolve().parent.parent / "whisper-trt" / "src"

import base64
from typing import List

import truss_chains as chains
import utilities.data_types as data_types


def patch_signint_bug():
    import fileinput
    import importlib.util
    import sys

    package_name = "tensorrt_llm"
    package_spec = importlib.util.find_spec(package_name)
    package_path = package_spec.origin if package_spec else None
    if not package_path:
        return
    f = Path(package_path).parent / "hlapi" / "utils.py"
    search_text = "signal.signal(signal.SIGINT, sigint_handler)"
    if not f.exists():
        return

    with fileinput.FileInput(str(f), inplace=True) as file:
        for line in file:
            if search_text in line:
                line = "    # " + line.lstrip()
            sys.stdout.write(line)


patch_signint_bug()


def load_requirements(file_path: Path) -> List:
    import toml

    if not file_path.exists():
        return []

    # Load the toml file
    with open(file_path, "r") as f:
        toml_data = toml.load(f)

    # Extract dependencies from project.dependencies
    dependencies = toml_data.get("project", {}).get("dependencies", [])
    return dependencies


class WhisperModel(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            base_image="baseten/truss-server-base:3.10-gpu-v0.9.0",
            apt_requirements=["python3.10-venv", "openmpi-bin", "libopenmpi-dev"],
            pip_requirements=[
                "--extra-index-url https://pypi.nvidia.com",
                *load_requirements(_LOCAL_WHISPER_LIB.parent / "pyproject.toml"),
                "toml",
            ],
            external_package_dirs=[chains.make_abs_path_here(str(_LOCAL_WHISPER_LIB))],
        ),
        compute=chains.Compute(gpu="A100", predict_concurrency=28),
    )

    def __init__(
        self,
    ) -> None:
        import whisper_trt

        # from huggingface_hub import snapshot_download
        # # Define the model ID and local directory
        # model_id = "baseten/whisper-trt-12-large-v2-sanity-checkpoint"
        # local_dir = "/data/engine"
        # # Download the model
        # snapshot_download(
        #     repo_id=model_id,
        #     local_dir=local_dir,
        #     max_workers=8,
        # )
        # Default to 100ms batching delay. Should be tuned for throughput requirements.
        self._model = whisper_trt.WhisperModel.from_model_name(
            "large-v2", max_queue_time=0.100, num_beams=5
        )

    async def run_remote(
        self, whisper_input: data_types.WhisperInput
    ) -> data_types.WhisperResult:
        # TODO: consider splitting out types from whisper-trt into their own package to eliminate all this ser/deser.
        binary_data = base64.b64decode(whisper_input.audio_b64.encode("utf-8"))
        waveform = self._model.preprocess_audio(binary_data)
        return data_types.WhisperResult(
            **(
                await self._model.generate(
                    waveform,
                    prompt=whisper_input.prompt,
                    language=whisper_input.language,
                    prefix=whisper_input.prefix,
                    max_new_tokens=whisper_input.max_new_tokens,
                    task=whisper_input.task,
                    raise_when_trimmed=whisper_input.raise_when_trimmed,
                )
            ).model_dump()
        )
