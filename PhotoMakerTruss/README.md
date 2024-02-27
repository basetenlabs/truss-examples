# PhotoMaker Truss Example

This README provides an overview of the PhotoMaker truss example, including setup instructions, usage details, and information about the `config.yaml` and `model.py` files.

## Overview

The PhotoMaker truss example demonstrates how to use the PhotoMaker model for customizing realistic human photos via stacked ID embedding. This example includes a configuration file (`config.yaml`) and a model file (`model.py`) that together define how to run the model with the provided example.

## Setup

To set up the PhotoMaker truss example, follow these steps:

1. Ensure you have Python >= 3.8 and PyTorch >= 2.0.0 installed.
2. Install the PhotoMaker package via pip:
   ```bash
   pip install git+https://github.com/TencentARC/PhotoMaker.git
   ```
3. Clone the repository containing the truss example:
   ```bash
   git clone https://github.com/basetenlabs/truss-examples
   cd truss-examples/PhotoMakerTruss
   ```

## Configuration (`config.yaml`)

The `config.yaml` file specifies the system dependencies, Python and PyTorch versions required, and the process for downloading and caching the PhotoMaker model weights. It defines the model class name as 'Model', the folder for the model class as 'model', and includes the installation command for the PhotoMaker package.

## Model (`model.py`)

The `model.py` file contains the definition of the `Model` class with `load` and `predict` methods. The `load` method includes logic to download the PhotoMaker model weights from the HuggingFace hub, while the `predict` method implements the inference logic based on the provided example.

## Running the Model

To run the model with the provided example, execute the following command:

```bash
python model/model.py
```

This command will use the configuration specified in `config.yaml` to set up the environment, download the necessary model weights, and run the inference based on the example provided in the `model.py` file.

## System Dependencies

- Python version: >= 3.8
- PyTorch version: >= 2.0.0
- Additional packages: torchvision, torchaudio, cuda

Ensure these dependencies are installed and properly configured before running the model.

## Downloading and Caching the Model

The `config.yaml` file includes a section for caching the PhotoMaker model weights at build-time. This ensures that the model weights are downloaded and available for the `load` method in the `model.py` file to use during inference.

For more information on the PhotoMaker model and its capabilities, visit the [PhotoMaker GitHub Repository](https://github.com/TencentARC/PhotoMaker).
