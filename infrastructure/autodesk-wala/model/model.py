import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import seed_everything

# Add packages directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "packages"))

# Enable TF32 for better performance on NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True


def _decode_image_b64_to_bytes(image_b64: str) -> bytes:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return base64.b64decode(image_b64)


def _file_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class Model:
    def __init__(self, **kwargs):
        self._data_dir: str = kwargs["data_dir"]
        self._config: Dict[str, Any] = kwargs["config"]
        self._secrets: Dict[str, Any] = kwargs["secrets"]
        self._environment: Dict[str, Any] = kwargs["environment"]

        self._model = None
        self._image_transform = None

        # Hugging Face access token (Baseten secret)
        try:
            self.hf_access_token: Optional[str] = self._secrets.get("hf_access_token")
        except Exception:
            self.hf_access_token = None

        # Lazy-loaded module references
        self._dataset_utils = None
        self._loaded_model_name: Optional[str] = None

    def load(self):
        default_model = "ADSKAILab/WaLa-SV-1B"
        configured = self._config.get("model_name")
        model_name = configured if isinstance(configured, str) else default_model

        from src.dataset_utils import (
            get_singleview_data,
            get_image_transform_latent_model,
        )
        from src.model_utils import Model as WaLaModel

        self._dataset_utils = {
            "get_singleview_data": get_singleview_data,
            "get_image_transform_latent_model": get_image_transform_latent_model,
        }

        if self.hf_access_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_access_token

        self._model = WaLaModel.from_pretrained(
            pretrained_model_name_or_path=model_name
        )
        self._image_transform = get_image_transform_latent_model()
        self._loaded_model_name = model_name

    def _ensure_model_loaded(self, model_name_override: Optional[str]):
        if self._model is None:
            self.load()
            return
        if model_name_override and model_name_override != self._loaded_model_name:
            if self.hf_access_token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_access_token
            from src.model_utils import Model as WaLaModel

            self._model = WaLaModel.from_pretrained(
                pretrained_model_name_or_path=model_name_override
            )
            self._loaded_model_name = model_name_override

    def _simplify_mesh_if_requested(
        self, obj_path: Path, target_num_faces: Optional[int]
    ):
        if not target_num_faces:
            return
        try:
            import open3d as o3d
        except Exception:
            return
        try:
            mesh = o3d.io.read_triangle_mesh(str(obj_path))
            simplified_mesh = mesh.simplify_quadric_decimation(int(target_num_faces))
            o3d.io.write_triangle_mesh(str(obj_path), simplified_mesh)
        except Exception:
            return

    def predict(self, model_input: Dict[str, Any]) -> Any:
        start_time = time.time()

        image_b64: Optional[str] = model_input.get("image_b64") or model_input.get(
            "image"
        )
        if not image_b64:
            return {
                "status": "error",
                "message": "Missing required 'image_b64' (or 'image') input.",
            }

        model_name_override: Optional[str] = model_input.get("model_name")
        self._ensure_model_loaded(model_name_override)

        scale: float = float(model_input.get("scale", 3.0))
        diffusion_rescale_timestep: int = int(
            model_input.get("diffusion_rescale_timestep", 100)
        )
        output_format: str = model_input.get("output_format", "obj").lower()
        target_num_faces: Optional[int] = model_input.get("target_num_faces")
        if target_num_faces is not None:
            target_num_faces = int(target_num_faces)
        seed: int = int(model_input.get("seed", 42))

        try:
            image_bytes = _decode_image_b64_to_bytes(image_b64)
            image_fileobj = io.BytesIO(image_bytes)
            data = self._dataset_utils["get_singleview_data"](
                image_file=image_fileobj,
                image_transform=self._image_transform,
                device=self._model.device,
                image_over_white=False,
            )
        except Exception as e:
            return {"status": "error", "message": f"Failed to decode image: {e}"}

        request_id = uuid.uuid4().hex[:8]
        save_dir = Path(self._data_dir) / "outputs" / request_id
        save_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(seed, workers=True)
        self._model.set_inference_fusion_params(scale, diffusion_rescale_timestep)

        try:
            output_path = self._model.test_inference(
                data, 0, save_dir=save_dir, output_format=output_format
            )
        except Exception as e:
            return {"status": "error", "message": f"Inference failed: {e}"}

        if output_format == "obj" and target_num_faces:
            self._simplify_mesh_if_requested(Path(output_path), target_num_faces)

        try:
            result_b64 = _file_to_b64(Path(output_path))
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read output file: {e}",
                "output_path": str(output_path),
            }

        elapsed = time.time() - start_time
        key = "obj_b64" if output_format == "obj" else "sdf_b64"
        return {
            "status": "success",
            key: result_b64,
            "data": result_b64,
            "format": output_format,
            "output_path": str(output_path),
            "time": elapsed,
        }
