from omegaconf import OmegaConf

from build3d.models.mvdream.ldm.util import instantiate_from_config

# from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from build3d.models.mvdream.model_zoo import build_model
from build3d.models.mvdream.camera_utils import get_camera_objaverse

import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_

from diffusers import DDPMScheduler
from diffusers.training_utils import compute_snr
from diffusers.optimization import get_scheduler

import boto3
from botocore.exceptions import NoCredentialsError


def download_from_s3(s3_path, local_path):
    """
    Download a file from S3 to a local path.
    :param s3_path: The S3 path to the file, in the format s3://bucket-name/path/to/file
    :param local_path: The local path where to save the file
    """
    try:
        # Extract bucket name and file path from s3_path
        bucket_name = s3_path.split("/")[2]
        s3_file_path = "/".join(s3_path.split("/")[3:])

        # Initialize S3 client
        s3 = boto3.client("s3")

        # Download file
        s3.download_file(bucket_name, s3_file_path, local_path)
        print(f"Downloaded {s3_path} to {local_path}")
    except NoCredentialsError:
        print("Credentials not available for AWS S3.")
    except Exception as e:
        print(f"Failed to download from S3: {e}")


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class MVdream_trainer(pl.LightningModule):
    def __init__(self, args=None, **kwargs):
        super().__init__()

        ### IF NOT NAMESAPCE
        if type(args) == dict:
            args = Args(**args)

        if args is None and len(kwargs) is not None:
            args = Args(**kwargs)

        self.args = args

        torch.set_float32_matmul_precision("medium")
        print("TORCH VERSION", torch.__version__)

        # self.device = args.device
        # self.weight_dtype =torch.float32
        # self.dtype = torch.float16 if args.fp16 else torch.float32

        # Load pre-trained model:
        if args.config_path is None:
            if args.ckpt_path is not None and args.ckpt_path.startswith("s3"):
                local_ckpt_path = "./init_model.ckpt"
                download_from_s3(args.ckpt_path, local_ckpt_path)
                model = build_model(args.model_name, ckpt_path=local_ckpt_path)
            else:
                model = build_model(args.model_name, ckpt_path=args.ckpt_path)
        else:
            assert args.ckpt_path is not None, "ckpt_path must be specified!"
            config = OmegaConf.load(args.config_path)
            model = instantiate_from_config(config.model)
            model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
        self.model = model  # .to(args.device)

        # NOTE these parameters were hard coded in mvdream LatentDiffusionInterface
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            prediction_type=self.args.prediction_type,
        )
        if hasattr(self.args, "zero_terminal_SNR") and self.args.zero_terminal_SNR:
            self.enforce_zero_terminal_snr()

        self.model.alphas_cumprod = self.noise_scheduler.alphas_cumprod
        self.model.betas = self.noise_scheduler.betas
        # self.model.alphas_cumprod_prev = self.noise_scheduler.alphas_cumprod # NOTE: don't know what this is

        if self.noise_scheduler.config.prediction_type == "epsilon":
            self.model.parameterization = "eps"
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            self.model.parameterization = "v"

        # self.vae = self.model.first_stage_model
        # text_encoder = self.model.cond_stage_model

        # self.unet = self.model.model

        # Freeze vae and text_encoder and set unet to trainable
        self.model.first_stage_model.requires_grad_(False)
        self.model.cond_stage_model.requires_grad_(False)
        # self.model.first_stage_model.to(self.weight_dtype)
        # self.model.cond_stage_model.to(self.weight_dtype)
        self.model.model.train()

        # Precompute camera matrices
        self.all_cameras = get_camera_objaverse(args.camera_path)  # .to(self.device)

        self.uc = None

        # if args.use_ema:
        #     ema_unet = build_model(args.model_name, ckpt_path=args.ckpt_path).model
        #     ema_unet = EMAModel(ema_unet.parameters())

    def enforce_zero_terminal_snr(self):

        # Compute alphas_bar_sqrt from alphas
        alphas = self.noise_scheduler.alphas
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
        self.noise_scheduler.alphas_cumprod = alphas_bar
        self.noise_scheduler.alphas = alphas
        self.noise_scheduler.betas = 1 - alphas

    def get_conditioning(self, camera, prompt, num_frames=1):
        assert type(prompt) == list
        # with torch.autocast(device_type=self.args.device, dtype=self.dtype):
        c = self.model.get_learned_conditioning(prompt)
        c_ = {"context": c}
        # if self.uc is not None:
        #     uc_ = {"context": self.uc.repeat(c.shape[0],1,1)}
        if camera is not None:
            c_["camera"] = camera
            c_["num_frames"] = num_frames
            # if self.uc is not None:
            #     uc_["camera"] = camera
            #     uc_["num_frames"] = num_frames
            # else:
            #     uc_ = None
        return c_  # , uc_

    def apply_unet(
        self,
        x,
        c,
        t,
        #   unconditional_guidance_scale=10., unconditional_conditioning=None,
    ):

        model_output = self.model.apply_model(x, t, c)

        # if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        #     model_output = self.model.apply_model(x, t, c)
        # else:
        #     x_in = torch.cat([x] * 2)
        #     t_in = torch.cat([t] * 2)
        #     if isinstance(c, dict):
        #         assert isinstance(unconditional_conditioning, dict)
        #         c_in = dict()
        #         for k in c:
        #             if isinstance(c[k], list):
        #                 c_in[k] = [torch.cat([
        #                     unconditional_conditioning[k][i],
        #                     c[k][i]]) for i in range(len(c[k]))]
        #             elif isinstance(c[k], torch.Tensor):
        #                 c_in[k] = torch.cat([
        #                         unconditional_conditioning[k],
        #                         c[k]])
        #             else:
        #                 assert c[k] == unconditional_conditioning[k]
        #                 c_in[k] = c[k]
        #     elif isinstance(c, list):
        #         c_in = list()
        #         assert isinstance(unconditional_conditioning, list)
        #         for i in range(len(c)):
        #             c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
        #     else:
        #         c_in = torch.cat([unconditional_conditioning, c])

        #     model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        #     # model_t = self.model.apply_model(x, t, c, **kwargs)
        #     # model_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
        #     model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        return model_output

    def training_step(self, batch, batch_idx):

        # Convert images to latent space
        # latents = self.model.first_stage_model.encode(batch["images"].to(self.weight_dtype)).sample()
        latents = self.model.first_stage_model.encode(batch["images"]).sample()
        latents = latents * self.model.scale_factor
        # print('TYPES !!!!! ', self.dtype, batch["images"].dtype, latents.dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if self.args.input_perturbation:
            new_noise = noise + self.args.input_perturbation * torch.randn_like(noise)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.args.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(
                latents, new_noise, timesteps
            )
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # encoder_hidden_states = text_encoder(batch["caption"])[0]
        all_cameras = self.all_cameras.to(self.device)
        camera = all_cameras[batch["img_idx"]]
        prompt = batch["caption"]
        # cond, uc = self.get_conditioning(camera, prompt, self.args.num_frames)
        cond = self.get_conditioning(camera, prompt, self.args.num_frames)

        # Get the target for loss depending on the prediction type
        if self.args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(
                prediction_type=self.args.prediction_type
            )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        # Predict the noise residual and compute loss
        # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # model_pred = self.model.apply_model(noisy_latents, timesteps, cond)
        model_pred = self.apply_unet(
            noisy_latents, cond, timesteps
        )  # , unconditional_conditioning=uc)

        if self.args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # Convert images to latent space
        latents = self.model.first_stage_model.encode(val_batch["images"]).sample()
        latents = latents * self.model.scale_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if self.args.input_perturbation:
            new_noise = noise + self.args.input_perturbation * torch.randn_like(noise)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.args.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(
                latents, new_noise, timesteps
            )
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # encoder_hidden_states = text_encoder(batch["caption"])[0]
        all_cameras = self.all_cameras.to(self.device)
        camera = all_cameras[val_batch["img_idx"]]
        prompt = val_batch["caption"]
        # cond, uc = self.get_conditioning(camera, prompt)
        cond = self.get_conditioning(camera, prompt)

        # Get the target for loss depending on the prediction type
        if self.args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(
                prediction_type=self.args.prediction_type
            )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        # Predict the noise residual and compute loss
        # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # model_pred = self.model.apply_model(noisy_latents, timesteps, cond)
        model_pred = self.apply_unet(
            noisy_latents, cond, timesteps
        )  # , unconditional_conditioning=uc)

        if self.args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            self.model.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        # if self.args.max_train_steps is None:
        #     self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

        if self.args.use_lr_scheduler:
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.args.lr_warmup_steps,  # * accelerator.num_processes,
                num_training_steps=self.args.max_train_steps,  # * accelerator.num_processes,
            )

            return ([optimizer], [lr_scheduler])
        else:
            return optimizer

    def on_after_backward(self):
        # Specify a very high value for max_norm to avoid actual clipping if undesired
        max_norm = 1e6
        total_norm = clip_grad_norm_(self.parameters(), max_norm=max_norm)

        # # Compute the norm of the gradients after clipping
        # total_norm_clipped = 0
        # norm_type = 2
        # for p in self.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(norm_type)
        #         total_norm_clipped += param_norm.item() ** norm_type
        # total_norm_clipped = total_norm_clipped ** (1. / norm_type)

        # print(total_norm, total_norm_clipped)
        self.log("gradients_norm", total_norm)
