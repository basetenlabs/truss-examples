# PhotoMaker Example

This README provides an overview of the PhotoMaker example, including a brief description of the PhotoMakerStableDiffusionXLPipeline, system requirements, setup instructions, and usage examples.

## Overview

PhotoMaker leverages the power of Stable Diffusion models to generate customized realistic human photos. The core of this example is the `PhotoMakerStableDiffusionXLPipeline`, which integrates seamlessly with the Stable Diffusion XL model to offer rapid customization of realistic human photos with high ID fidelity and diverse, controllable text prompts.

## System Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- It is recommended to use Anaconda or Miniconda for environment setup.

## Setup Instructions

1. Clone the PhotoMaker repository:

```bash
git clone https://github.com/TencentARC/PhotoMaker.git
cd PhotoMaker
```

2. Create and activate a new conda environment:

```bash
conda create --name photomaker python=3.10
conda activate photomaker
```

3. Install the required dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
pip install git+https://github.com/TencentARC/PhotoMaker.git
```

## Usage Examples

### Importing the Pipeline

```python
from photomaker import PhotoMakerStableDiffusionXLPipeline
```

### Model Download

The model can be downloaded automatically using:

```python
from huggingface_hub import hf_hub_download
photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
```

Alternatively, you can manually download the model from [this URL](https://huggingface.co/TencentARC/PhotoMaker).

### Running Predictions

```python
# Initialize the pipeline
pipe = PhotoMakerStableDiffusionXLPipeline()

# Load the model
pipe.load_model(photomaker_path)

# Generate an image
image = pipe.predict("A happy person walking in the park")
image.save("output_image.png")
```

### Testing the Model

Run a local Gradio demo using the `app.py` file in the `gradio_demo` directory for interactive testing. For MAC users, follow the instructions in `MacGPUEnv.md`.

## Acknowledgements

This example is based on the work done by TencentARC and the PhotoMaker community. For more information, visit the [PhotoMaker GitHub repository](https://github.com/TencentARC/PhotoMaker).

## Disclaimer

This project is intended for educational and research purposes only. Users are responsible for the lawful and ethical use of the generated content.
