from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import base64
from io import BytesIO
from PIL import Image

from cuda import cudart
from huggingface_hub import snapshot_download

from diffusion.utilities import PIPELINE_TYPE, TRT_LOGGER, add_arguments, process_pipeline_args
from diffusion.stable_diffusion_pipeline import StableDiffusionPipeline

torch.backends.cuda.matmul.allow_tf32 = True

VAE_SCALING_FACTOR = 0.13025
STABLE_DIFFUSION_VERSION = 'xl-1.0'
PYTORCH_MODEL_DIR = 'pytorch_model'

# TODO(pankaj) Support using cuda graphs. Setting this to
# True would break generation right now.
USE_CUDA_GRAPH = False


class StableDiffusionXLPipeline(StableDiffusionPipeline):
    def __init__(self, vae_scaling_factor=0.13025, **kwargs):
        self.base = StableDiffusionPipeline(
            pipeline_type=PIPELINE_TYPE.XL_BASE,
            vae_scaling_factor=vae_scaling_factor,
            return_latents=True,
            **kwargs)
        self.refiner = StableDiffusionPipeline(
            pipeline_type=PIPELINE_TYPE.XL_REFINER,
            vae_scaling_factor=vae_scaling_factor,
            return_latents=False,
            **kwargs)

    def load_engines(
        self, 
        framework_model_dir, 
        onnx_dir, 
        engine_dir, 
        onnx_refiner_dir='onnx_xl_refiner', 
        engine_refiner_dir='engine_xl_refiner', 
        **kwargs,
    ):
        self.base.loadEngines(
            engine_dir, 
            framework_model_dir, 
            onnx_dir, 
            **kwargs,
        )
        self.refiner.loadEngines(
            engine_refiner_dir, 
            framework_model_dir, 
            onnx_refiner_dir, 
            **kwargs,
        )

    def activate_engines(self, shared_device_memory=None):
        self.base.activateEngines(shared_device_memory)
        self.refiner.activateEngines(shared_device_memory)

    def load_resources(self, image_height, image_width, batch_size, seed):
        """Among other things, cleanly allocate memory buffers."""
        self.base.loadResources(image_height, image_width, batch_size, seed)
        # Use a different seed for refiner - we arbitrarily use base seed+1, if specified.
        self.refiner.loadResources(
            image_height, 
            image_width, 
            batch_size, 
            ((seed+1) if seed is not None else None),
        )

    def get_max_device_memory(self):
        max_device_memory = self.base.calculateMaxDeviceMemory()
        max_device_memory = max(max_device_memory, self.refiner.calculateMaxDeviceMemory())
        return max_device_memory

    def run(
        self, 
        prompt, 
        negative_prompt, 
        height, 
        width, 
        denoising_end: float, 
        guidance_scale: float,
        num_inference_steps: int,
    ):
        # Run a part of steps in base and the rest on refiner, as governed
        # by denoising_end setting.
        latents, time_base = self.base.infer(
            prompt, 
            negative_prompt, 
            height, 
            width, 
            denoising_end=denoising_end,
            denoising_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            warmup=False,
        )

        images, time_refiner = self.refiner.infer(
            prompt, 
            negative_prompt, 
            height, 
            width, 
            input_image=latents, 
            refiner_denoising_start=denoising_end,
            denoising_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            warmup=False,
        )
        return images


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._data_dir = kwargs["data_dir"]

    def load(self):
        snapshot_download(repo_id="baseten/sdxl-1.0-trt-8.6.1.post1-engine", local_dir="/app/data")
        self.pipe = StableDiffusionXLPipeline(
            vae_scaling_factor=VAE_SCALING_FACTOR, 
            version=STABLE_DIFFUSION_VERSION,
            use_cuda_graph=USE_CUDA_GRAPH,
            framework_model_dir=PYTORCH_MODEL_DIR,
        )

        # Load TensorRT engines and pytorch modules
        kwargs_load_refiner = {
            'onnx_refiner_dir': 'onnx_refiner', 
            'engine_refiner_dir': str(self._data_dir / 'engine_xl_refiner'),
        }
        kwargs_load_engine = {
            'enable_all_tactics': False,
            'enable_refit': False,
            'onnx_opset': 18,
            'opt_batch_size': 1,
            'opt_image_height': 1024,
            'opt_image_width': 1024,
            'static_batch': True,
            'static_shape': True,
            'timing_cache': None,
        }
        self.pipe.load_engines(
            framework_model_dir=PYTORCH_MODEL_DIR,
            onnx_dir='onnx',
            engine_dir=str(self._data_dir / 'engine'),
            **kwargs_load_refiner,
            **kwargs_load_engine)

        # Activate engines
        _, shared_device_memory = cudart.cudaMalloc(self.pipe.get_max_device_memory())
        self.pipe.activate_engines(shared_device_memory)

    def predict(
        self,
        args: Any,
    ) -> Any:
        # prompt and negative_prompt are lists to support batching in future.
        # Batching is not supported right now.
        prompt = [args.pop("prompt")]
        negative_prompt = [args.pop("negative_prompt", '')]
        num_inference_steps = args.pop("num_inference_steps", 30)
        guidance_scale = args.pop("guidance_scale", 5.0)
        denoising_end = args.pop("denoising_end", 0.8)
        height = args.pop("height", 1024)
        width = args.pop("width", 1024)

        print("generating with settings:")
        print(
            {
                **args,
                "height": height,
                "width": width,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "denoising_end": denoising_end,
                "guidance_scale": guidance_scale,
            }
        )

        # TODO(pankaj) Support seed, batch_size
        self.pipe.load_resources(height, width, 1, None)
        image_tensors = self.pipe.run(
            prompt, 
            negative_prompt, 
            height, 
            width, 
            denoising_end=denoising_end, 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        # Convert image tensors to jpeg
        image_tensors = (
            ((image_tensors + 1) * 255 / 2)
            .clamp(0, 255)
            .detach()
            .permute(0, 2, 3, 1)
            .round()
            .type(torch.uint8)
            .cpu()
            .numpy())
        images = [Image.fromarray(image_tensors[i]) 
                  for i in range(image_tensors.shape[0])]

        return {
            "status": "success",
            "data": [
                {"base64": _convert_to_b64(image)}
                for image in images
            ],
            "message": None,
        }


def _convert_to_b64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64
