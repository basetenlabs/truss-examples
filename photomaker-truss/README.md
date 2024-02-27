# PhotoMaker Truss Example

## Overview

The `PhotoMaker` model by TencentARC is designed for customizing realistic human photos with high fidelity and diverse text controllability. It serves as an adapter to collaborate with other base models alongside LoRA modules in the community.

## Key Features

- Rapid customization within seconds without additional LoRA training.
- High-quality generation with impressive ID fidelity.
- Diverse text controllability and promising diversity.

## System Dependencies

- For GPUs not supporting bfloat16, use `torch_dtype = torch.float16` for improved speed.
- Minimum GPU memory requirement for `PhotoMaker` is **11G**.

## Installation

```bash
conda create --name photomaker python=3.10
conda activate photomaker
pip install -U pip
pip install -r requirements.txt
pip install git+https://github.com/TencentARC/PhotoMaker.git
```

## Usage

Load the `PhotoMakerStableDiffusionXLPipeline` and use it for image generation tasks.

```python
from photomaker import PhotoMakerStableDiffusionXLPipeline
```

## Model Download

Automatically download the model using the following Python code:

```python
from huggingface_hub import hf_hub_download
photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
```

## Testing the Model

Refer to the `photomaker_demo.ipynb` and `photomaker_style_demo.ipynb` notebooks for examples on how to generate realistic and stylized images.

## Truss Configuration (`config.yaml`)

The `config.yaml` for Truss should include the following keys based on the [Truss reference config](https://truss.baseten.co/reference/config):
- `description`: A brief description of the model's purpose.
- `environment_variables`: Any required environment variables.
- `python_version`: Specify `py310` for Python 3.10.
- `requirements`: List of Python packages required, such as `torch`, `transformers`, and `diffusers`.
- `resources`: Define the compute resources needed, including CPU, memory, and GPU.
- `model_cache`: Cache the `PhotoMaker` model weights at build-time for faster startup.

## Related Resources

- [Replicate Demo (Realistic)](https://replicate.com/jd7h/photomaker)
- [Replicate Demo (Stylization)](https://replicate.com/yorickvp/photomaker-style)
- [WebUI version of PhotoMaker](https://github.com/lllyasviel/stable-diffusion-webui-forge)
- [Windows version of PhotoMaker](https://github.com/bmaltais/PhotoMaker)

## Acknowledgements

The `PhotoMaker` model is co-hosted by Tencent ARC Lab and Nankai University MCG-NKU. It is inspired by demos and repositories such as IP-Adapter and FastComposer.

## Disclaimer

The `PhotoMaker` project is designed for positive impact in AI-driven image generation. Users are expected to comply with local laws and use the tool responsibly.
