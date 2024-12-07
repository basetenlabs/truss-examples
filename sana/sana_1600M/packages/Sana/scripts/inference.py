# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import re
import subprocess
import tarfile
import time
import warnings
from dataclasses import dataclass, field

# from datetime import datetime
from typing import List, Optional

import pyrallis
import torch
from termcolor import colored
from torchvision.utils import save_image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning

from diffusion import DPMS, FlowEuler, SASolverSampler
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
)
from diffusion.model.builder import (
    build_model,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_decode,
)
from diffusion.model.utils import prepare_prompt_ar
from diffusion.utils.config import SanaConfig
from diffusion.utils.logger import get_root_logger
from tools.download import find_model


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


def get_dict_chunks(data, bs):
    keys = []
    for k in data:
        keys.append(k)
        if len(keys) == bs:
            yield keys
            keys = []
    if keys:
        yield keys


def create_tar(data_path):
    tar_path = f"{data_path}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(data_path, arcname=os.path.basename(data_path))
    print(f"Created tar file: {tar_path}")
    return tar_path


def delete_directory(exp_name):
    if os.path.exists(exp_name):
        subprocess.run(["rm", "-r", exp_name], check=True)
        print(f"Deleted directory: {exp_name}")


@torch.inference_mode()
def visualize(config, args, model, items, bs, sample_steps, cfg_scale, pag_scale=1.0):
    if isinstance(items, dict):
        get_chunks = get_dict_chunks
    else:
        from diffusion.data.datasets.utils import get_chunks

    generator = torch.Generator(device=device).manual_seed(args.seed)
    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    for chunk in tqdm(
        list(get_chunks(items, bs)),
        desc=tqdm_desc,
        unit="batch",
        position=args.gpu_id,
        leave=True,
    ):
        # data prepare
        prompts, hw, ar = (
            [],
            torch.tensor(
                [[args.image_size, args.image_size]], dtype=torch.float, device=device
            ).repeat(bs, 1),
            torch.tensor([[1.0]], device=device).repeat(bs, 1),
        )
        if bs == 1:
            prompt = data_dict[chunk[0]]["prompt"] if dict_prompt else chunk[0]
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(
                prompt, base_ratios, device=device, show=False
            )
            latent_size_h, latent_size_w = (
                (
                    int(hw[0, 0] // config.vae.vae_downsample_rate),
                    int(hw[0, 1] // config.vae.vae_downsample_rate),
                )
                if args.image_size == 1024
                else (latent_size, latent_size)
            )
            prompts.append(prompt_clean.strip())
        else:
            for data in chunk:
                prompt = data_dict[data]["prompt"] if dict_prompt else data
                prompts.append(
                    prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[
                        0
                    ].strip()
                )
            latent_size_h, latent_size_w = latent_size, latent_size

        # check exists
        save_file_name = f"{chunk[0]}.jpg" if dict_prompt else f"{prompts[0][:100]}.jpg"
        save_path = os.path.join(save_root, save_file_name)
        if os.path.exists(save_path):
            # make sure the noise is totally same
            torch.randn(
                bs,
                config.vae.vae_latent_dim,
                latent_size,
                latent_size,
                device=device,
                generator=generator,
            )
            continue

        # prepare text feature
        if not config.text_encoder.chi_prompt:
            max_length_all = config.text_encoder.model_max_length
            prompts_all = prompts
        else:
            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
            prompts_all = [chi_prompt + prompt for prompt in prompts]
            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = (
                num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
            )  # magic number 2: [bos], [_]

        caption_token = tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
        caption_embs = text_encoder(
            caption_token.input_ids, caption_token.attention_mask
        )[0][:, None][:, :, select_index]
        emb_masks = caption_token.attention_mask[:, select_index]
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

        # start sampling
        with torch.no_grad():
            n = len(prompts)
            z = torch.randn(
                n,
                config.vae.vae_latent_dim,
                latent_size,
                latent_size,
                device=device,
                generator=generator,
            )
            model_kwargs = dict(
                data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks
            )

            if args.sampling_algo == "dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == "sa-solver":
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(config.vae.vae_latent_dim, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]
            elif args.sampling_algo == "flow_euler":
                flow_solver = FlowEuler(
                    model,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )
                samples = flow_solver.sample(
                    z,
                    steps=sample_steps,
                )
            elif args.sampling_algo == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model,
                    condition=caption_embs,
                    uncondition=null_y,
                    guidance_type=guidance_type,
                    cfg_scale=cfg_scale,
                    pag_scale=pag_scale,
                    pag_applied_layers=pag_applied_layers,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                    interval_guidance=args.interval_guidance,
                )
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=flow_shift,
                )
            else:
                raise ValueError(f"{args.sampling_algo} is not defined")

        samples = samples.to(weight_dtype)
        samples = vae_decode(config.vae.vae_type, vae, samples)
        torch.cuda.empty_cache()

        os.umask(0o000)
        for i, sample in enumerate(samples):
            save_file_name = (
                f"{chunk[i]}.jpg" if dict_prompt else f"{prompts[i][:100]}.jpg"
            )
            save_path = os.path.join(save_root, save_file_name)
            # logger.info(f"Saving path: {save_path}")
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]


@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = (
        "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    )
    model_path: Optional[str] = (
        "hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"
    )
    work_dir: str = "output/inference"
    version: str = "sigma"
    txt_file: str = "asset/samples_mini.txt"
    json_file: Optional[str] = None
    sample_nums: int = 100_000
    bs: int = 1
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    sampling_algo: str = "flow_dpm-solver"
    seed: int = 0
    dataset: str = "custom"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_image_size: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    debug: bool = False
    if_save_dirname: bool = False


if __name__ == "__main__":

    args = get_args()
    config = args = pyrallis.parse(config_class=SanaInference, config_path=args.config)

    args.image_size = config.model.image_size
    if args.custom_image_size:
        args.image_size = args.custom_image_size
        print(f"custom_image_size: {args.image_size}")

    set_env(args.seed, args.image_size // config.vae.vae_downsample_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    # only support fixed latent size currently
    latent_size = args.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.model_max_length
    pe_interpolation = config.model.pe_interpolation
    micro_condition = config.model.micro_condition
    flow_shift = config.scheduler.flow_shift
    pag_applied_layers = config.model.pag_applied_layers
    guidance_type = "classifier-free_PAG"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [
        max(0, args.interval_guidance[0]),
        min(1, args.interval_guidance[1]),
    ]
    sample_steps_dict = {
        "dpm-solver": 20,
        "sa-solver": 25,
        "flow_dpm-solver": 20,
        "flow_euler": 28,
    }
    sample_steps = (
        args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    )
    if config.model.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.model.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif config.model.mixed_precision == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"weigh precision {config.model.mixed_precision} is not defined"
        )
    logger.info(
        f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}"
    )

    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device).to(
        weight_dtype
    )
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=config.text_encoder.text_encoder_name, device=device
    )

    null_caption_token = tokenizer(
        "",
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    null_caption_embs = text_encoder(
        null_caption_token.input_ids, null_caption_token.attention_mask
    )[0]

    # model setting
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": config.text_encoder.model_max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": text_encoder.config.hidden_size,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
    }
    model = build_model(
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False),
        **model_kwargs,
    ).to(device)
    # model = build_model(config.model, **model_kwargs).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    model.eval().to(weight_dtype)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")
    args.sampling_algo = (
        args.sampling_algo
        if ("flow" not in args.model_path or args.sampling_algo == "flow_dpm-solver")
        else "flow_euler"
    )

    if args.work_dir is None:
        work_dir = (
            f"/{os.path.join(*args.model_path.split('/')[:-2])}"
            if args.model_path.startswith("/")
            else os.path.join(*args.model_path.split("/")[:-2])
        )
        img_save_dir = os.path.join(str(work_dir), "vis")
    else:
        img_save_dir = args.work_dir

    logger.info(colored(f"Saving images at {img_save_dir}", "green"))
    dict_prompt = args.json_file is not None
    if dict_prompt:
        data_dict = json.load(open(args.json_file))
        items = list(data_dict.keys())
    else:
        with open(args.txt_file) as f:
            items = [item.strip() for item in f.readlines()]
    logger.info(f"Eval first {min(args.sample_nums, len(items))}/{len(items)} samples")
    items = items[: max(0, args.sample_nums)]
    items = items[max(0, args.start_index) : min(len(items), args.end_index)]

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}")

    def create_save_root(
        args, dataset, epoch_name, step_name, sample_steps, guidance_type
    ):
        save_root = os.path.join(
            img_save_dir,
            # f"{datetime.now().date() if args.exist_time_prefix == '' else args.exist_time_prefix}_"
            f"{dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
            f"_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}"
            f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
        )

        if args.pag_scale != 1.0:
            save_root = save_root.replace(
                f"scale{args.cfg_scale}",
                f"scale{args.cfg_scale}_pagscale{args.pag_scale}",
            )
        if flow_shift != 1.0:
            save_root += f"_flowshift{flow_shift}"
        if guidance_type != "classifier-free":
            save_root += f"_{guidance_type}"
        if args.interval_guidance[0] != 0 and args.interval_guidance[1] != 1:
            save_root += f"_intervalguidance{args.interval_guidance[0]}{args.interval_guidance[1]}"

        save_root += f"_imgnums{args.sample_nums}" + args.add_label
        return save_root

    def guidance_type_select(default_guidance_type, pag_scale, attn_type):
        guidance_type = default_guidance_type
        if not (pag_scale > 1.0 and attn_type == "linear"):
            logger.info("Setting back to classifier-free")
            guidance_type = "classifier-free"
        return guidance_type

    dataset = (
        "MJHQ-30K" if args.json_file and "MJHQ-30K" in args.json_file else args.dataset
    )
    if args.ablation_selections and args.ablation_key:
        for ablation_factor in args.ablation_selections:
            setattr(args, args.ablation_key, eval(ablation_factor))
            print(f"Setting {args.ablation_key}={eval(ablation_factor)}")
            sample_steps = (
                args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
            )
            guidance_type = guidance_type_select(
                guidance_type, args.pag_scale, config.model.attn_type
            )

            save_root = create_save_root(
                args, dataset, epoch_name, step_name, sample_steps, guidance_type
            )
            os.makedirs(save_root, exist_ok=True)
            if args.if_save_dirname and args.gpu_id == 0:
                # save at work_dir/metrics/tmp_xxx.txt for metrics testing
                with open(
                    f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w"
                ) as f:
                    print(
                        f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt"
                    )
                    f.write(os.path.basename(save_root))
            logger.info(
                f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}"
            )

            visualize(
                config=config,
                args=args,
                model=model,
                items=items,
                bs=args.bs,
                sample_steps=sample_steps,
                cfg_scale=args.cfg_scale,
                pag_scale=args.pag_scale,
            )
    else:
        guidance_type = guidance_type_select(
            guidance_type, args.pag_scale, config.model.attn_type
        )
        logger.info(
            f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}"
        )

        save_root = create_save_root(
            args, dataset, epoch_name, step_name, sample_steps, guidance_type
        )
        os.makedirs(save_root, exist_ok=True)
        if args.if_save_dirname and args.gpu_id == 0:
            # save at work_dir/metrics/tmp_xxx.txt for metrics testing
            with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
                print(
                    f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt"
                )
                f.write(os.path.basename(save_root))

        if args.debug:
            items = [
                'a blackboard wrote text "Hello World"'
                'Text" Super Dad Mode ON", t shirt design, This is a graffiti-style image.The letters are surrounded by a playful, abstract design of paw prints and pet-related shapes, such as a heart-shaped bone and a cat-whisker-shaped element.',
                '"NR Beauty Hair" logo para peluqueria, product, typography, fashion, painting',
                'Text"Goblins gone wild.", The text is written in an elegant, vintage-inspired font and each letter in the text showed in different colors.',
                "An awe-inspiring 3D render of the mahir Olympics logo, set against the backdrop of a fiery, burning Olympic flame. The flames dance and intertwine to form the iconic Olympic rings and typography, while the Eiffel Tower stands tall in the distance. The cinematic-style poster is rich in color and detail, evoking a sense of excitement and anticipation for the upcoming games., ukiyo-e, vibrant, cinematic, 3d render, typography, poster",
                'Cute cartoon back style of a couple, wearing a black t shirts , she have long hair with the name "C". He have staright hair and light beard with the name "J"white color,heart snowy atmosphere, typography, 3d render, portrait photography, fashion',
                'A captivating 3D render of a whimsical, colorful scene, featuring the word "Muhhh" spelled out in vibrant, floating balloons. The wordmark hovers above a lush, emerald green field. A charming, anthropomorphic rabbit with a wide smile and twinkling eyes hops alongside the balloon letters. The background showcases a serene, dreamy sky with soft pastel hues, creating an overall atmosphere of joy, enchantment, and surrealism. The 3D render is a stunning illustration that blends fantasy and realism effortlessly., illustration, 3d render',
                'create a logo for a company named "FUN"',
                "A stunningly realistic image of an Asian woman sitting on a plush sofa, completely engrossed in a book. She is wearing cozy loungewear and has headphones on, indicating her desire for a serene and quiet environment. In one hand, she holds a can of water, providing a refreshing sensation. The adjacent table features an array of snacks and books, adding to the cozy ambiance of the scene. The room is filled with natural light streaming through vibrantly decorated windows, and tasteful decorations contribute to the overall relaxing and soothing atmosphere.",
                'A captivating 3D logo illustration of the name "ANGEL" in a romantic and enchanting Follow my Page poster design. The lettering is adorned with a majestic, shimmering crown encrusted with intricate gemstones. Swirling pink and purple patterns, reminiscent of liquid or air, surround the crown, with beautiful pink flowers in full bloom and bud adorning the design. Heart-shaped decorations enhance the romantic ambiance, and a large, iridescent butterfly with intricate wings graces the right side of the crown. The muted purple background contrasts with the bright and lively elements within the composition, creating a striking visual effect. The 3D rendering showcases the intricate details and depth of the design, making it a truly mesmerizing piece of typography, 3D render, and illustration art., illustration, typography, poster, 3d render',
                'A human wearing a T-shirt with Text "NVIDIA" and logo',
                'Logo with text "Hi"',
            ]
        visualize(
            config=config,
            args=args,
            model=model,
            items=items,
            bs=args.bs,
            sample_steps=sample_steps,
            cfg_scale=args.cfg_scale,
            pag_scale=args.pag_scale,
        )

        if args.tar_and_del:
            create_tar(save_root)
            delete_directory(save_root)

    print(
        colored(f"Sana inference has finished. Results stored at ", "green"),
        colored(f"{img_save_dir}", attrs=["bold"]),
        ".",
    )
