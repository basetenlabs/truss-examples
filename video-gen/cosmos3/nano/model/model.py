import base64
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The cu130 deps (torch + flash-attn-3-nv + natten dev6 with libnatten + ...)
# AND the cosmos3 Python package itself live in the base image's
# /workspace/.venv, which python_executable_path points at.

CHECKPOINT_PATH = os.environ.get("COSMOS3_CHECKPOINT_PATH", "/app/model_cache/cosmos3-nano")

# OmniSampleOverrides fields that get threaded straight through from the
# request when present. Action-conditioned modes (forward_dynamics,
# inverse_dynamics, policy) only set the action_* / domain_name / image_size
# fields; vision-only modes only set vision_path. Fields outside this list
# get rejected by pydantic.
_PASSTHROUGH_FIELDS = (
    "vision_path",
    "shift",
    "num_steps",
    "guidance",
    "guidance_interval",
    "action_path",
    "action_chunk_size",
    "domain_name",
    "image_size",
    "raw_action_dim",
)

# Modes that emit a generated video/image at output_dir/vision.{mp4,jpg}.
_VISION_OUTPUT_MODES = {"text2image", "text2video", "image2video", "forward_dynamics"}
# Modes that emit a predicted action tensor under sample_outputs.json.
_ACTION_OUTPUT_MODES = {"inverse_dynamics", "policy"}


class Model:
    def __init__(self, **kwargs: Any) -> None:
        self._secrets = kwargs["secrets"]
        self._pipe: Any = None
        self._OmniSampleOverrides: Any = None
        self._get_sample_data: Any = None
        self._inference_root = Path(tempfile.mkdtemp(prefix="cosmos3-out-"))

    def load(self) -> None:
        token = self._secrets.get("hf_access_token")
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

        # cosmos3's `init_script` does several things (TOKENIZERS_PARALLELISM,
        # distributed setup, log config, ...) plus a sanity check that blows
        # up under truss's in-process load() retry. Skip it and explicitly
        # set the only piece that affects inference correctness on a single
        # GPU.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        torch.set_grad_enabled(False)

        # cosmos3's checkpoint downloader shells out to `uvx hf@1.13.0
        # download`, but hf 1.13.0's CLI imports click without declaring it
        # as a dep so the uvx-isolated env always fails. Replace _hf_download
        # with a direct huggingface_hub Python-API call (the same hf
        # package is already installed in the base image's venv).
        from cosmos3._src.imaginaire.utils import checkpoint_db as _ckpt
        from huggingface_hub import hf_hub_download, snapshot_download

        def _hf_download_python(cmd_args: list[str]) -> str:
            repo_id = cmd_args[0]
            repo_type = "model"
            revision: str | None = None
            allow: list[str] = []
            ignore: list[str] = []
            filename: str | None = None
            i = 1
            while i < len(cmd_args):
                arg = cmd_args[i]
                if arg == "--repo-type":
                    repo_type = cmd_args[i + 1]
                    i += 2
                elif arg == "--revision":
                    revision = cmd_args[i + 1]
                    i += 2
                elif arg == "--include":
                    allow.append(cmd_args[i + 1])
                    i += 2
                elif arg == "--exclude":
                    ignore.append(cmd_args[i + 1])
                    i += 2
                elif arg.startswith("--"):
                    i += 2
                else:
                    filename = arg
                    i += 1
            if filename is not None:
                return hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    filename=filename,
                )
            return snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                allow_patterns=allow or None,
                ignore_patterns=ignore or None,
            )

        _ckpt._hf_download = _hf_download_python

        from cosmos3.args import OmniSampleOverrides, OmniSetupOverrides
        from cosmos3.inference import OmniInference, get_sample_data

        self._OmniSampleOverrides = OmniSampleOverrides
        self._get_sample_data = get_sample_data

        setup_args = OmniSetupOverrides(
            checkpoint_path=CHECKPOINT_PATH,
            output_dir=self._inference_root,
        ).build_setup()
        self._pipe = OmniInference.create(setup_args)
        logger.info("Cosmos3 loaded from %s", CHECKPOINT_PATH)

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._pipe is None:
            raise RuntimeError("Model not loaded")

        sample_name = request.get("name") or f"sample-{uuid.uuid4().hex[:8]}"
        model_mode = request.get("model_mode", "text2video")
        # cosmos3 writes vision.{mp4,jpg} and sample_outputs.json directly
        # into output_dir (it does NOT append `name` as a subdir), so each
        # request gets its own per-sample directory.
        output_dir = self._inference_root / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)

        overrides_kwargs: dict[str, Any] = {
            "name": sample_name,
            "output_dir": output_dir,
            "prompt": request.get("prompt", ""),
            "model_mode": model_mode,
            "num_frames": int(request.get("num_frames", 189)),
            "fps": int(request.get("fps", 24)),
            "resolution": request.get("resolution", "720"),
            "aspect_ratio": request.get("aspect_ratio", "16,9"),
            "seed": int(request.get("seed", 0)),
        }
        for field in _PASSTHROUGH_FIELDS:
            if request.get(field) is not None:
                overrides_kwargs[field] = request[field]

        overrides = self._OmniSampleOverrides(**overrides_kwargs)
        # Resolves any URLs in vision_path / action_path to local files in
        # output_dir. Without this, build_sample raises "Must call `download()`
        # before building vision data" when the request points at a URL.
        overrides.download(output_dir)
        sample_args = overrides.build_sample(model_config=self._pipe.model_config)
        data_batch = self._get_sample_data(sample_args, model=self._pipe.model)
        self._pipe.generate_batch([sample_args], data_batch)

        response: dict[str, Any] = {"name": sample_name, "model_mode": model_mode}

        # Vision output (when the mode generates one).
        out_mp4 = output_dir / "vision.mp4"
        out_jpg = output_dir / "vision.jpg"
        out_path = out_mp4 if out_mp4.exists() else (out_jpg if out_jpg.exists() else None)
        if out_path is not None:
            response["format"] = out_path.suffix.lstrip(".")
            with out_path.open("rb") as f:
                response["data"] = base64.b64encode(f.read()).decode("ascii")

        # Action output (inverse_dynamics, policy). cosmos3 writes a
        # sample_outputs.json with the predicted action tensor (+ raw_action_dim,
        # generation status). Forward it through.
        outputs_path = output_dir / "sample_outputs.json"
        if outputs_path.exists():
            response["sample_outputs"] = json.loads(outputs_path.read_text())

        produces_vision = model_mode in _VISION_OUTPUT_MODES
        produces_action = model_mode in _ACTION_OUTPUT_MODES
        if produces_vision and out_path is None and not produces_action:
            raise RuntimeError(f"Inference produced no output under {output_dir}")

        return response
