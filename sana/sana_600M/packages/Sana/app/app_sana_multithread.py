#!/usr/bin/env python
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
from __future__ import annotations

import argparse
import os
import random
import uuid
from datetime import datetime

import gradio as gr
import numpy as np
import spaces
import torch
from app import safety_check
from app.sana_pipeline import SanaPipeline
from diffusers import FluxPipeline
from PIL import Image
from torchvision.utils import make_grid, save_image
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
DEMO_PORT = int(os.getenv("DEMO_PORT", "15432"))
os.environ["GRADIO_EXAMPLES_CACHE"] = "./.gradio/cache"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, "
        "glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, "
        "disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
SCHEDULE_NAME = ["Flow_DPM_Solver"]
DEFAULT_SCHEDULE_NAME = "Flow_DPM_Solver"
NUM_IMAGES_PER_PROMPT = 1
TEST_TIMES = 0
FILENAME = f"output/port{DEMO_PORT}_inference_count.txt"


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)


def read_inference_count():
    global TEST_TIMES
    try:
        with open(FILENAME) as f:
            count = int(f.read().strip())
    except FileNotFoundError:
        count = 0
    TEST_TIMES = count

    return count


def write_inference_count(count):
    with open(FILENAME, "w") as f:
        f.write(str(count))


def run_inference(num_imgs=1):
    TEST_TIMES = read_inference_count()
    TEST_TIMES += int(num_imgs)
    write_inference_count(TEST_TIMES)

    return (
        f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: "
        f"16px; color:red; font-weight: bold;'>{TEST_TIMES}</span>"
    )


def update_inference_count():
    count = read_inference_count()
    return (
        f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: "
        f"16px; color:red; font-weight: bold;'>{count}</span>"
    )


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument(
        "--model_path",
        nargs="?",
        default="output/Sana_D20/SANA.pth",
        type=str,
        help="Path to the model file (positional)",
    )
    parser.add_argument("--output", default="./", type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--cfg_scale", default=5.0, type=float)
    parser.add_argument("--pag_scale", default=2.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--step", default=-1, type=int)
    parser.add_argument("--custom_image_size", default=None, type=int)
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="google/shieldgemma-2b",
    )

    return parser.parse_args()


args = get_args()

if torch.cuda.is_available():
    weight_dtype = torch.float16
    model_path = args.model_path
    pipe = SanaPipeline(args.config)
    pipe.from_pretrained(model_path)
    pipe.register_progress_bar(gr.Progress())

    repo_name = "black-forest-labs/FLUX.1-dev"
    pipe2 = FluxPipeline.from_pretrained(repo_name, torch_dtype=torch.float16).to(
        "cuda"
    )

    # safety checker
    safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
    safety_checker_model = AutoModelForCausalLM.from_pretrained(
        args.shield_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)

    set_env(42)


def save_image_sana(img, seed="", save_img=False):
    unique_name = f"{str(uuid.uuid4())}_{seed}.png"
    save_path = os.path.join(f"output/online_demo_img/{datetime.now().date()}")
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(save_path, exist_ok=True)
    unique_name = os.path.join(save_path, unique_name)
    if save_img:
        save_image(img, unique_name, nrow=1, normalize=True, value_range=(-1, 1))

    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU(enable_queue=True)
async def generate_2(
    prompt: str = None,
    negative_prompt: str = "",
    style: str = DEFAULT_STYLE_NAME,
    use_negative_prompt: bool = False,
    num_imgs: int = 1,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    flow_dpms_guidance_scale: float = 5.0,
    flow_dpms_pag_guidance_scale: float = 2.0,
    flow_dpms_inference_steps: int = 20,
    randomize_seed: bool = False,
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"PORT: {DEMO_PORT}, model_path: {model_path}")
    if safety_check.is_dangerous(
        safety_checker_tokenizer, safety_checker_model, prompt
    ):
        prompt = "A red heart."

    print(prompt)

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    with torch.no_grad():
        images = pipe2(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=50,
            num_images_per_prompt=num_imgs,
            max_sequence_length=256,
            generator=generator,
        ).images

    save_img = False
    img = images
    if save_img:
        img = [save_image_sana(img, seed, save_img=save_image) for img in images]
        print(img)
    torch.cuda.empty_cache()

    return img


@spaces.GPU(enable_queue=True)
async def generate(
    prompt: str = None,
    negative_prompt: str = "",
    style: str = DEFAULT_STYLE_NAME,
    use_negative_prompt: bool = False,
    num_imgs: int = 1,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    flow_dpms_guidance_scale: float = 5.0,
    flow_dpms_pag_guidance_scale: float = 2.0,
    flow_dpms_inference_steps: int = 20,
    randomize_seed: bool = False,
):
    global TEST_TIMES
    # seed = 823753551
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"PORT: {DEMO_PORT}, model_path: {model_path}, time_times: {TEST_TIMES}")
    if safety_check.is_dangerous(
        safety_checker_tokenizer, safety_checker_model, prompt
    ):
        prompt = "A red heart."

    print(prompt)

    num_inference_steps = flow_dpms_inference_steps
    guidance_scale = flow_dpms_guidance_scale
    pag_guidance_scale = flow_dpms_pag_guidance_scale

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    pipe.progress_fn(0, desc="Sana Start")

    with torch.no_grad():
        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            pag_guidance_scale=pag_guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_imgs,
            generator=generator,
        )

    pipe.progress_fn(1.0, desc="Sana End")

    save_img = False
    if save_img:
        img = [save_image_sana(img, seed, save_img=save_image) for img in images]
        print(img)
    else:
        if num_imgs > 1:
            nrow = 2
        else:
            nrow = 1
        img = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
        img = (
            img.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        img = [Image.fromarray(img.astype(np.uint8))]

    torch.cuda.empty_cache()

    return img


TEST_TIMES = read_inference_count()
model_size = "1.6" if "D20" in args.model_path else "0.6"
title = """
    <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
        <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="50%" alt="logo"/>
    </div>
"""
DESCRIPTION = f"""
        <p><span style="font-size: 36px; font-weight: bold;">Sana-{model_size}B</span><span style="font-size: 20px; font-weight: bold;">{args.image_size}px</span></p>
        <p style="font-size: 16px; font-weight: bold;">Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer</p>
        <p><span style="font-size: 16px;"><a href="https://arxiv.org/abs/2410.10629">[Paper]</a></span> <span style="font-size: 16px;"><a href="https://github.com/NVlabs/Sana">[Github(coming soon)]</a></span> <span style="font-size: 16px;"><a href="https://nvlabs.github.io/Sana">[Project]</a></span</p>
        <p style="font-size: 16px; font-weight: bold;">Powered by <a href="https://hanlab.mit.edu/projects/dc-ae">DC-AE</a> with 32x latent space</p>
        <p style="font-size: 16px; font-weight: bold;">Unsafe word will give you a 'Red Heart' in the image instead.</p>
        """
if model_size == "0.6":
    DESCRIPTION += "\n<p>0.6B model's text rendering ability is limited.</p>"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

examples = [
    'a cyberpunk cat with a neon sign that says "Sana"',
    "A very detailed and realistic full body photo set of a tall, slim, and athletic Shiba Inu in a white oversized straight t-shirt, white shorts, and short white shoes.",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "portrait photo of a girl, photograph, highly detailed face, depth of field",
    'make me a logo that says "So Fast"  with a really cool flying dragon shape with lightning sparks all over the sides and all of it contains Indonesian language',
    "🐶 Wearing 🕶 flying on the 🌈",
    # "👧 with 🌹 in the ❄️",
    # "an old rusted robot wearing pants and a jacket riding skis in a supermarket.",
    # "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
    # "Astronaut in a jungle, cold color palette, muted colors, detailed",
    # "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests",
]

css = """
.gradio-container{max-width: 1024px !important}
h1{text-align:center}
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    info_box = gr.Markdown(
        value=f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: 16px; color:red; font-weight: bold;'>{read_inference_count()}</span>"
    )
    demo.load(
        fn=update_inference_count, outputs=info_box
    )  # update the value when re-loading the page
    # with gr.Row(equal_height=False):
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run-sana", scale=0)
            run_button2 = gr.Button("Run-flux", scale=0)

        with gr.Row():
            result = gr.Gallery(
                label="Result from Sana",
                show_label=True,
                columns=NUM_IMAGES_PER_PROMPT,
                format="webp",
            )
            result_2 = gr.Gallery(
                label="Result from FLUX",
                show_label=True,
                columns=NUM_IMAGES_PER_PROMPT,
                format="webp",
            )

    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            with gr.Row(visible=True):
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
            with gr.Row():
                flow_dpms_inference_steps = gr.Slider(
                    label="Sampling steps",
                    minimum=5,
                    maximum=40,
                    step=1,
                    value=18,
                )
                flow_dpms_guidance_scale = gr.Slider(
                    label="CFG Guidance scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    value=5.0,
                )
                flow_dpms_pag_guidance_scale = gr.Slider(
                    label="PAG Guidance scale",
                    minimum=1,
                    maximum=4,
                    step=0.5,
                    value=2.0,
                )
            with gr.Row():
                use_negative_prompt = gr.Checkbox(
                    label="Use negative prompt", value=False, visible=True
                )
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=True,
            )
            style_selection = gr.Radio(
                show_label=True,
                container=True,
                interactive=True,
                choices=STYLE_NAMES,
                value=DEFAULT_STYLE_NAME,
                label="Image Style",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row(visible=True):
                schedule = gr.Radio(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=SCHEDULE_NAME,
                    value=DEFAULT_SCHEDULE_NAME,
                    label="Sampler Schedule",
                    visible=True,
                )
                num_imgs = gr.Slider(
                    label="Num Images",
                    minimum=1,
                    maximum=6,
                    step=1,
                    value=1,
                )

    run_button.click(fn=run_inference, inputs=num_imgs, outputs=info_box)

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result_2],
        fn=generate_2,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    run_button.click(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            num_imgs,
            seed,
            height,
            width,
            flow_dpms_guidance_scale,
            flow_dpms_pag_guidance_scale,
            flow_dpms_inference_steps,
            randomize_seed,
        ],
        outputs=[result],
        queue=True,
    )

    run_button2.click(
        fn=generate_2,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            num_imgs,
            seed,
            height,
            width,
            flow_dpms_guidance_scale,
            flow_dpms_pag_guidance_scale,
            flow_dpms_inference_steps,
            randomize_seed,
        ],
        outputs=[result_2],
        queue=True,
    )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0", server_port=DEMO_PORT, debug=True, share=True
    )
