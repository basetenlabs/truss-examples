from typing import Optional, List
import torch
import numpy as np
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from typing import Optional, List
from omegaconf import OmegaConf

from src.mvdream.model_zoo import build_model
from src.mvdream.camera_utils import get_camera
from src.mvdream.ldm.models.diffusion.ddim import DDIMSampler
from src.mvdream.camera_utils import get_camera, get_camera_build3d
from src.mvdream.ldm.util import instantiate_from_config


def enforce_zero_terminal_snr(noise_scheduler):

    # Compute alphas_bar_sqrt from alphas
    alphas = noise_scheduler.alphas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])

    # Reset noise scheduler
    noise_scheduler.alphas_cumprod = alphas_bar
    noise_scheduler.alphas = alphas
    noise_scheduler.betas = 1 - alphas


def process_line(line):
    before_comma = line.split(",")[0]
    words = before_comma.split()
    if words[0].lower() in ["the", "a", "an"]:
        words = words[1:]

    # Join words with underscores
    return "_".join(words)


class MVDreamModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "sd-v2.1-base-4view",
        prediction_type: str = "epsilon",
        ckpt_path: Optional[str] = None,
        config_path: Optional[str] = None,
        zero_terminal_SNR: bool = False,
        use_default_camera: bool = False,
        camera_elev: int = 15,
        camera_azim: int = 50,
        camera_azim_span: int = 360,
        testing_views: List[int] = [0, 6, 10, 26],
    ):
        super().__init__()
        self.model_name = model_name
        self.prediction_type = prediction_type
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.zero_terminal_SNR = zero_terminal_SNR
        self.use_default_camera = use_default_camera
        self.camera_elev = camera_elev
        self.camera_azim = camera_azim
        self.camera_azim_span = camera_azim_span
        self.testing_views = testing_views

        if self.ckpt_path is not None:
            print(f"Loading from checkpoint: {self.ckpt_path}")

        if self.config_path is None:
            self.model = build_model(
                model_name=self.model_name, ckpt_path=self.ckpt_path
            )
        else:
            assert self.ckpt_path is not None, "ckpt_path must be specified!"
            config = OmegaConf.load(self.config_path)
            self.model = instantiate_from_config(config.model)
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu"))
        self.sampler = self.setup_sampler()
        self.model = self.model

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, *args, **kwargs):
        return cls(*args, ckpt_path=checkpoint_path, **kwargs)

    def setup_sampler(self):
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            prediction_type=self.prediction_type,
        )
        if self.zero_terminal_SNR:
            enforce_zero_terminal_snr(noise_scheduler)

        self.model.alphas_cumprod = noise_scheduler.alphas_cumprod
        self.model.betas = noise_scheduler.betas
        self.model.alphas_cumprod_prev = (
            noise_scheduler.alphas_cumprod
        )  # NOTE: don't know what this is ?

        if noise_scheduler.config.prediction_type == "epsilon":
            self.model.parameterization = "eps"
        elif noise_scheduler.config.prediction_type == "v_prediction":
            self.model.parameterization = "v"
        return DDIMSampler(self.model)

    def setup_camera(
        self,
        num_frames: int = 4,
        testing_views = [0, 6, 10, 26],
    ):
        indices_list = testing_views
        batch_size = num_frames
        if self.use_default_camera:
            camera = get_camera(
                batch_size,
                elevation=self.camera_elev,
                azimuth_start=self.camera_azim,
                azimuth_span=self.camera_azim_span,
            )
        else:
            camera = get_camera_build3d(indices_list)
        camera = camera.repeat(batch_size // num_frames, 1).to(self.device)
        return camera

    def _inference(
        self,
        prompt,
        batch_size,
        uc,
        camera,
        num_frames,
        image_size,
        step,
        guidance_scale,
        ddim_discretize,
    ):
        c = self.model.get_learned_conditioning(prompt).to(self.device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = self.sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc_,
            eta=0.0,
            ddim_discretize=ddim_discretize,
            x_T=None,
        )
        x_sample = self.model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()
        return x_sample

    def inference_step(
        self,
        prompt: str,
        suffix: str = ", 3d asset",
        num_frames: int = 4,
        image_size: int = 256,
        step: int = 50,
        guidance_scale: float = 10,
        ddim_discretize: str = "trailing",
        fp16: bool = False,
        testing_views: List[int] = [0, 6, 10, 26]
    ):
        self.model.eval()
        dtype = torch.float16 if fp16 else torch.float32
        batch_size = num_frames
        if "3D" not in prompt and "3d" not in prompt:
            prompt = prompt + suffix

        uc = self.model.get_learned_conditioning([""]).to(self.device)
        camera = self.setup_camera(num_frames=num_frames, testing_views=testing_views)

        if type(prompt) != list:
            prompt = [prompt]

        with torch.no_grad():
            x_sample = self._inference(
                prompt=prompt,
                batch_size=batch_size,
                uc=uc,
                camera=camera,
                num_frames=num_frames,
                image_size=image_size,
                step=step,
                guidance_scale=guidance_scale,
                ddim_discretize=ddim_discretize,
            )
        images = list(x_sample.astype(np.uint8))
        return images, testing_views
