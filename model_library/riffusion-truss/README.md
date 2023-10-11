# Riffusion Truss

Riffusion is an app for real-time music generation with stable diffusion.

Read about it at https://www.riffusion.com/about and try it at https://www.riffusion.com/.

* Web app: https://github.com/hmartiro/riffusion-app
* Inference server: https://github.com/hmartiro/riffusion-inference
* Model checkpoint: https://huggingface.co/riffusion/riffusion-model-v1

This repository contains the Python backend, packaged as a [Truss](https://truss.baseten.co), that runs the model inference and audio processing, including:

 * a diffusers pipeline that performs prompt interpolation combined with image conditioning
 * a module for (approximately) converting between spectrograms and waveforms
 * the configuration files for the Truss


## Install
Tested with Python 3.9 and diffusers 0.9.0

```
pip install -r dev_requirements.txt
pip install -r requirements.txt
sudo apt update
sudo apt install ffmpeg
```

## Run Riffusion Truss locally

If your local system has a GPU capable of running Riffusion, you can invoke it locally via Truss.

After installing the necessary packages, open a Python shell or notebook in this directory and run:

```python
th = truss.from_directory("./riffusion/")
test_req = {
  "alpha": 0.75,
  "num_inference_steps": 50,
  "seed_image_id": "og_beat",

  "start": {
    "prompt": "church bells on sunday",
    "seed": 42,
    "denoising": 0.75,
    "guidance": 7.0
  },

  "end": {
    "prompt": "jazz with piano",
    "seed": 123,
    "denoising": 0.75,
    "guidance": 7.0
  }
}
th.server_predict(test_req)
```

This should give you an output formatted as follows:

```json
{
  "image": "< base64 encoded JPEG image >",
  "audio": "< base64 encoded MP3 clip >"
}
```

For the full APIs, see [InferenceInput](https://github.com/hmartiro/riffusion-inference/blob/main/riffusion/datatypes.py#L28) and [InferenceOutput](https://github.com/hmartiro/riffusion-inference/blob/main/riffusion/datatypes.py#L54).

## Deploy Riffusion to Baseten

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd riffusion-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `riffusion-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Citation

This work is built upon the original Riffusion, cited it as follows:

```
@software{Forsgren_Martiros_2022,
  author = {Forsgren, Seth* and Martiros, Hayk*},
  title = {{Riffusion - Stable diffusion for real-time music generation}},
  url = {https://riffusion.com/about},
  year = {2022}
}
```
