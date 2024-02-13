# PhotoMaker Truss Example

Welcome to the PhotoMaker Truss example! This README provides an overview of the truss example, including key features such as rapid customization, high ID fidelity, and adapter usage with other base models. It also contains detailed instructions on setting up the environment, system requirements, and how to run the model with the PhotoMakerStableDiffusionXLPipeline.

## Key Features

- **Rapid Customization**: Customize realistic human photos within seconds, with no additional LoRA training required.
- **High ID Fidelity**: Ensures impressive ID fidelity, offering diversity, promising text controllability, and high-quality generation.
- **Adapter Usage**: Can serve as an adapter to collaborate with other base models alongside LoRA modules in the community.

## Dependencies and Installation

Ensure you have the following system dependencies:

- Python >= 3.10
- PyTorch >= 2.0.0

To set up your environment, follow these steps:

```bash
conda create --name photomaker python=3.10
conda activate photomaker
pip install -U pip
pip install -r requirements.txt
pip install git+https://github.com/TencentARC/PhotoMaker.git
```

## System Requirements

- For non-bfloat16 GPUs, change to `torch.float16` for improved speed.
- Minimum GPU memory: 11G.

## Model Usage

Import the pipeline and download the model automatically:

```python
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
```

## Running the Model

To run the model with the PhotoMakerStableDiffusionXLPipeline, use the following code snippet:

```python
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    "path_to_your_model",
    torch_dtype=torch.bfloat16
).to("cuda")

# Generate an image
generated_image = pipe(prompt="A portrait of a person img in the style of Pixar").images[0]
generated_image.save("output.png")
```

## Usage Tips

- Upload multiple photos for better ID fidelity.
- Adjust the Style strength for stylization. The larger the number, the less ID fidelity, but the stylization ability will be better.
- Reduce the number of generated images and sampling steps for faster speed.

For more information and advanced usage, please refer to the official [PhotoMaker GitHub repository](https://github.com/TencentARC/PhotoMaker).

Thank you for using PhotoMaker!
