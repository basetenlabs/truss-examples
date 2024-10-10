import threading

import base64
import fastapi
import logging
import math
import random
import subprocess
import time
from io import BytesIO

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image

logging.basicConfig(level=logging.INFO)
MAX_SEED = np.iinfo(np.int32).max


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs["secrets"]
        self.model_name = kwargs["config"]["model_metadata"]["repo_id"]
        self.hf_access_token = self._secrets["hf_access_token"]
        self.pipe = None
        self._thread = None
        self._exception = None
        self._result = None

    def load(self):
        self.pipe = FluxPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, token=self.hf_access_token
        ).to("cuda")
        # self.pipe.enable_model_cpu_offload()
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            logging.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed with code {e.returncode}: {e.stderr}")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def _generate_image(self, prompt, prompt2, guidance_scale, max_sequence_length,
                        num_inference_steps, width, height, generator):
        time.sleep(15)
        try:
            image = self.pipe(
                prompt=prompt,
                prompt_2=prompt2,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                output_type="pil",
                generator=generator,
            ).images[0]
            self._result = image
            return
        except Exception as e:
            logging.info(f"Image generation was aborted or failed: {e}")
            self._exception = e
            return

    async def predict(self, model_input, request: fastapi.Request):
        start = time.perf_counter()
        timeout_sec = model_input.get("timeout_sec", 60)
        seed = model_input.get("seed")
        prompt = model_input.get("prompt")
        prompt2 = model_input.get("prompt2")
        logging.info(f"Starting: {prompt}")
        max_sequence_length = model_input.get(
            "max_sequence_length", 256
        )  # 256 is max for FLUX.1-schnell
        guidance_scale = model_input.get(
            "guidance_scale", 0.0
        )  # 0.0 is the only value for FLUX.1-schnell
        num_inference_steps = model_input.get(
            "num_inference_steps", 4
        )  # schnell is timestep-distilled
        width = model_input.get("width", 1024)
        height = model_input.get("height", 1024)
        if not math.isclose(guidance_scale, 0.0):
            logging.warning(
                "FLUX.1-schnell does not support guidance_scale other than 0.0"
            )
            guidance_scale = 0.0
        if not seed:
            seed = random.randint(0, MAX_SEED)
        if len(prompt.split()) > max_sequence_length:
            logging.warning(
                "FLUX.1-schnell does not support prompts longer than 256 tokens, truncating"
            )
            tokens = prompt.split()
            prompt = " ".join(tokens[: min(len(tokens), max_sequence_length)])
        if prompt2 and len(prompt2.split()) > max_sequence_length:
            logging.warning(
                f"Input prompt2 longer than {max_sequence_length} tokens, truncating"
            )
            tokens = prompt2.split()
            prompt2 = " ".join(tokens[: min(len(tokens), max_sequence_length)])
        generator = torch.Generator().manual_seed(seed)

        logging.info(f"Starting: thread.")
        self._reset()
        self._thread = threading.Thread(target=self._generate_image, args=(
            prompt, prompt2, guidance_scale, max_sequence_length, num_inference_steps,
            width, height, generator))
        self._thread.start()
        logging.info(f"started thread.")

        logging.info(f"Polling")
        while self._thread.is_alive():
            elapsed_sec = time.perf_counter() - start
            if await request.is_disconnected():
                logging.info("Aborting due to client disconnect.")
                self._abort_thread()
                raise fastapi.HTTPException(status_code=408, detail="Client disconnected.")
            elif elapsed_sec > timeout_sec:
                logging.info("Aborting due to timeout.")
                self._abort_thread()
                raise fastapi.HTTPException(status_code=408, detail="Timeout.")
            time.sleep(1.0)
            if self._result:
                logging.info("Result there.")
                logging.info(f"Thread alive: {self._thread.is_alive()}")
                break

        logging.info(f"Polling done.")
        self._abort_thread()

        if not self._result:
            assert self._exception
            raise self._exception
        else:
            image = self._result

        b64_results = self.convert_to_b64(image)
        end = time.perf_counter()
        logging.info(f"Total time taken: {end - start} seconds")
        return {"data": b64_results}

    def _abort_thread(self):
        if not self._thread or not self._thread.is_alive():
            return
        t0 = time.perf_counter()
        logging.info(f"Setting interrupt")
        self.pipe._interrupt = True
        logging.info(f"Waiting to join")
        self._thread.join()
        logging.info(f"Joined after {time.perf_counter() - t0} seconds.")

    def _reset(self):
        self.pipe._interrupt = False
        self._thread = None
        self._result = None
        self._exception = None
