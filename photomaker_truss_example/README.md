# PhotoMaker Truss Example README

## Overview

This example demonstrates the use of TencentARC's PhotoMaker with a truss structure for customizing realistic human photos. It leverages the power of stacked ID embedding for rapid customization without additional LoRA training, ensuring high ID fidelity for realistic human photo generation. This example can also serve as an Adapter with other Base Models and LoRA modules.

## Key Features

- Rapid customization within seconds without additional LoRA training.
- High ID fidelity for realistic human photo generation.
- Can serve as an Adapter with other Base Models and LoRA modules.

## System Dependencies

- Minimum GPU memory requirement: **11G**.
- For GPUs not supporting bfloat16, change to `torch_dtype = torch.float16` for improved speed.

## Environment Setup

```bash
conda create --name photomaker python=3.10
conda activate photomaker
pip install -U pip
pip install -r requirements.txt
pip install git+https://github.com/TencentARC/PhotoMaker.git
```

## Model Usage

```python
from photomaker import PhotoMakerStableDiffusionXLPipeline

# Load base model
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16"
).to(device)

# Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img"
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
```

## Running Predictions

```python
prompt = "a half-body portrait of a man img wearing sunglasses in an Iron Man suit, best quality"
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
generator = torch.Generator(device=device).manual_seed(42)
images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=10,
    generator=generator,
).images[0]
gen_images.save('out_photomaker.png')
```

## Local Gradio Demo

Run the following command to start a local Gradio demo:

```bash
python gradio_demo/app.py
```

## Usage Tips

- Upload more photos of the person to be customized to improve ID fidelity.
- For stylization, adjust the Style strength to 30-50 for better effects.
- Reduce the number of generated images and sampling steps for faster speed, but be aware of potential ID fidelity loss.
