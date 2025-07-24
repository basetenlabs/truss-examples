import argparse
import base64
import os
import sys
import time
from io import BytesIO
from typing import Any, List, Optional

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
from huggingface_hub import snapshot_download
from PIL import Image

# Add the current directory to Python path so demo_diffusion can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from demo_diffusion import dd_argparse
from demo_diffusion import pipeline as pipeline_module
from huggingface_hub import snapshot_download
from PIL import Image


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self.pipe = None
        self.args = None
        self._secrets = kwargs.get("secrets")

    def load(self):
        """Load the Flux model with TensorRT engines and perform warmup runs."""
        print("[I] Initializing Flux txt2img model using TensorRT")

        if self._secrets and "hf_access_token" in self._secrets:
            os.environ["HF_TOKEN"] = self._secrets["hf_access_token"]
        
        # Download TensorRT engine files (with error handling for private repos)
        try:
            snapshot_download(
                repo_id="baseten-admin/flux.1-dev-trt-10.11.0.33.engine-B200",
                local_dir="/app/data",
            )
            print("[I] TensorRT engine files downloaded successfully")
        except Exception as e:
            print(f"[W] Could not download TensorRT engines from Hugging Face: {e}")
            print("[I] Assuming TensorRT engine files are already present in /app/data")
        
        # Create arguments for the pipeline
        self.args = self._create_args()
        
        # Initialize the pipeline
        self.pipe = pipeline_module.FluxPipeline.FromArgs(
            self.args, 
            pipeline_type=pipeline_module.PIPELINE_TYPE.TXT2IMG
        )
        
        # Load TensorRT engines and pytorch modules
        print("[I] Loading TensorRT engines and ONNX models...")
        _, kwargs_load_engine, _ = dd_argparse.process_pipeline_args(self.args)
        self.pipe.load_engines(
            framework_model_dir="/app/data",
            **kwargs_load_engine,
        )
        
        # Allocate device memory
        if self.pipe.low_vram:
            self.pipe.device_memory_sizes = self.pipe.get_device_memory_sizes()
        else:
            _, shared_device_memory = cudart.cudaMalloc(self.pipe.calculate_max_device_memory())
            self.pipe.activate_engines(shared_device_memory)
        
        # Load resources for default dimensions
        self.pipe.load_resources(
            self.args.height, 
            self.args.width, 
            self.args.batch_size, 
            self.args.seed
        )
        
        # Perform warmup runs
        print("[I] Performing warmup runs...")
        self._perform_warmup_runs()
        print("[I] Model loaded successfully")

    def _create_args(self) -> argparse.Namespace:
        """Create argument namespace for the Flux pipeline."""
        parser = argparse.ArgumentParser(description="Flux Txt2Img Model")
        
        # Add all necessary arguments
        parser = dd_argparse.add_arguments(parser)
        
        # Add only Flux-specific arguments that aren't already in dd_argparse
        parser.add_argument(
            "--prompt2",
            default=None,
            nargs="*",
            help="Text prompt(s) to be sent to the T5 tokenizer and text encoder. If not defined, prompt will be used instead",
        )
        parser.add_argument(
            "--max_sequence_length",
            type=int,
            default=512,
            help="Maximum sequence length to use with the prompt. Can be up to 512 for the dev and 256 for the schnell variant.",
        )
        parser.add_argument(
            "--t5-ws-percentage",
            type=int,
            default=None,
            help="Set runtime weight streaming budget as the percentage of the size of streamable weights for the T5 model.",
        )
        parser.add_argument(
            "--transformer-ws-percentage",
            type=int,
            default=None,
            help="Set runtime weight streaming budget as the percentage of the size of streamable weights for the transformer model."
        )
        
        # Set default values for required arguments
        default_args = [
            "a beautiful landscape",  # prompt
            "--version", "flux.1-dev",
            "--height", "1024",
            "--width", "1024",
            "--batch-size", "1",
            "--batch-count", "1",
            "--denoising-steps", "50",
            "--guidance-scale", "3.5",
            "--max_sequence_length", "512",
            "--fp4",
            "--download-onnx-models",  # Required for FP4 models since native export is not supported
            "--onnx-dir", "/app/data/onnx",  # Directory for ONNX files
            "--engine-dir", "/app/data/engine",  # Directory for TensorRT engines
            "--custom-engine-paths", 
            "clip:/app/data/clip/engine_trt10.13.0.35.plan,t5:/app/data/t5/engine_trt10.13.0.35.plan,transformer:/app/data/transformer_fp4/engine_trt10.13.0.35.plan,vae:/app/data/vae/engine_trt10.13.0.35.plan",
            "--num-warmup-runs", "2",
            "--seed", "42",
            "--framework-model-dir", "/app/data",  # Point to the directory containing the model files
        ]
        
        return parser.parse_args(default_args)

    def _perform_warmup_runs(self):
        """Perform warmup runs to optimize CUDA kernels."""
        warmup_prompt = "a beautiful landscape with mountains and a lake, photorealistic, high quality"
        
        for i in range(self.args.num_warmup_runs):
            print(f"[I] Warmup run {i+1}/{self.args.num_warmup_runs}")
            
            # Create warmup arguments
            kwargs_run_demo = {
                "prompt": [warmup_prompt],
                "prompt2": [warmup_prompt],
                "height": self.args.height,
                "width": self.args.width,
                "batch_count": 1,
                "num_warmup_runs": 0,  # Don't do nested warmups
                "use_cuda_graph": self.args.use_cuda_graph,
            }
            
            # Run warmup inference
            self.pipe.run(**kwargs_run_demo)

    def predict(self, model_input: Any) -> Any:
        """Generate image from text prompt."""
        # Extract parameters from model input
        prompt = model_input.pop("prompt", "a beautiful landscape")
        negative_prompt = model_input.pop("negative_prompt", "")
        height = model_input.pop("height", 1024)
        width = model_input.pop("width", 1024)
        num_inference_steps = model_input.pop("num_inference_steps", 50)
        guidance_scale = model_input.pop("guidance_scale", 3.5)
        seed = model_input.pop("seed", None)
        batch_size = model_input.pop("batch_size", 1)
        batch_count = model_input.pop("batch_count", 1)
        
        # Validate dimensions
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Height and width must be multiples of 8")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            self.args.seed = seed
        
        # Update pipeline resources if dimensions changed
        if height != self.args.height or width != self.args.width or batch_size != self.args.batch_size:
            self.pipe.load_resources(height, width, batch_size, self.args.seed)
            self.args.height = height
            self.args.width = width
            self.args.batch_size = batch_size
        
        # Prepare prompts
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = prompt * batch_size
        
        # Use prompt2 if provided, otherwise use prompt
        prompt2 = model_input.pop("prompt2", None)
        if prompt2 is None:
            prompt2 = prompt
        elif not isinstance(prompt2, list):
            prompt2 = [prompt2]
        if len(prompt2) == 1:
            prompt2 = prompt2 * batch_size
        
        # Prepare negative prompt
        if not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt]
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * batch_size
        
        # Update guidance scale
        self.args.guidance_scale = guidance_scale
        
        # Update denoising steps
        self.args.denoising_steps = num_inference_steps
        
        start_time = time.time()
        
        try:
            # Run inference directly using the infer method
            latents, walltime_ms = self.pipe.infer(
                prompt=prompt,
                prompt2=prompt2,
                image_height=height,
                image_width=width,
                warmup=False,
                save_image=False  # Don't save to file, we'll handle it ourselves
            )
            
            # Process the returned latents the same way the pipeline does when save_image=True
            # The latents returned are raw tensor data that need to be processed into images
            processed_images = (
                ((latents + 1) * 255 / 2)
                .clamp(0, 255)
                .detach()
                .permute(0, 2, 3, 1)
                .round()
                .type(torch.uint8)
                .cpu()
                .numpy()
            )
            
            # Convert numpy arrays to PIL Images
            pil_images = []
            for image in processed_images:
                pil_image = Image.fromarray(image)
                pil_images.append(pil_image)
            
            # Convert images to base64
            b64_images = []
            for image in pil_images:
                b64_images.append(self.convert_to_b64(image))
            
            end_time = time.time() - start_time
            
            print(f"[I] Generated {len(processed_images)} images in {end_time:.2f} seconds")
            
            # Return results
            if len(b64_images) == 1:
                return {
                    "status": "success",
                    "data": b64_images[0],
                    "time": end_time,
                    "prompt": prompt[0] if len(prompt) == 1 else prompt,
                    "negative_prompt": negative_prompt[0] if len(negative_prompt) == 1 else negative_prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                }
            else:
                return {
                    "status": "success",
                    "data": b64_images,
                    "time": end_time,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                }
                
        except Exception as e:
            end_time = time.time() - start_time
            print(f"[E] Error during inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "time": end_time,
            }

    def convert_to_b64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64
