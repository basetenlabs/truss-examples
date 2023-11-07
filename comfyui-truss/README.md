## ComfyUI Truss

This truss is designed to allow ComfyUI users to easily convert their workflows into a production grade API service. 

## Exporting the ComfyUI workflow

Inside ComfyUI, you can save workflows as a JSON file. However, the regular JSON format the ComfyUI uses will not work. Instead, the workflow has to be saved in the API format. Here is how you can do that:

1. Go to ComfyUI and click on the gear icon for the project
![gear_icon](images/comfyui-screenshot-1.png)

2. Next, checkmark the box which says `Enable Dev Mode Options`
![enable_dev_mode_options](images/comfyui-screenshot-2.png)

3. Now, if you go back to the project you will see an option called `Save(API Format)`. This is the one you want to use to save your workflow. Using this method you can save any ComfyUI workflow as a JSON file in the API format.
![save_api_format](images/comfyui-screenshot-3.png)


## Setting up the project

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd comfyui-truss
```

For your ComfyUI workflow, you probably used one or more models. Those models need to be defined inside truss. From the root of the truss project, open up the file inside `data/model.json`. This file will contain all of the models that need to get downloaded in order for your ComfyUI workflow to run. Here is an example of the contents of this file:

```json
[
    {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "path": "checkpoints/sd_xl_base_1.0.safetensors"
    },
    {
        "url": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        "path": "controlnet/diffusers_xl_canny_full.safetensors"
    }
]
```

In this case, I have 2 models: SDXL and a controlnet. Each model needs to have 2 things, `url` and `path`. The `url` is the location for downloading the model. The `path` is where this model will get stored inside truss. For the path, follow the same guidelines as used in ComfyUI. Models should get stored inside `checkpoints`, controlnets should be stored inside `controlnet`, etc.


## Deployment

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `comfyui-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Model Inference

The main thing we need for inference is the JSON workflow we exported in step 1. Inside the JSON workflow file, there might be some inputs such as the positive prompt or negative prompt that are hard coded. We want these inputs to be dynamically sent to the model, so we can use handlebars to templatize them. Here is an example of a JSON workflow with templatized inputs:

```json
{
  "6": {
    "inputs": {
      "text": "{{positive_prompt}}",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "{{negative_prompt}}",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "11": {
    "inputs": {
      "image": "{{controlnet_image}}",
      "choose file to upload": "image"
    },
    "class_type": "LoadImage"
  },
  "12": {
    "inputs": {
      "control_net_name": "diffusers_xl_canny_full.safetensors"
    },
    "class_type": "ControlNetLoader"
  },
  "14": {
    "inputs": {
      "ckpt_name": "sd_xl_base_1.0.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "15": {
    "inputs": {
      "images": [
        "16",
        0
      ]
    },
    "class_type": "PreviewImage"
  },
  "18": {
    "inputs": {
      "images": [
        "8",
        0
      ]
    },
    "class_type": "PreviewImage"
  }
}
```

This is not the entire JSON workflow file, but the nodes 6, 7, and 11 accept variable inputs. You can do this by using the handlebars format of `{{variable_name_here}}`. Inside a seperate JSON object we can define the values for these variables such as:

```python
values = {
  "positive_prompt": "An igloo on a snowy day, 4k, hd",
  "negative_prompt": "blurry, text, low quality",
  "controlnet_image": "https://storage.googleapis.com/logos-bucket-01/baseten_logo.png"
}
```

Just be sure that the variable names in the workflow template match the names inside the values object. 

Here is a complete example of how you make a prediction request to your truss in python:

```python
headers = {"Authorization": f"Api-Key YOUR-BASETEN-API-KEY-HERE"}

sdxl_controlnet_workflow = {
  "3": {
    "inputs": {
      "seed": 972197629127129,
      "steps": 40,
      "cfg": 7,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "14",
        0
      ],
      "positive": [
        "10",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler"
  },
  "5": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage"
  },
  "6": {
    "inputs": {
      "text": "{{positive_prompt}}",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "{{negative_prompt}}",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "14",
        2
      ]
    },
    "class_type": "VAEDecode"
  },
  "10": {
    "inputs": {
      "strength": 0.6,
      "conditioning": [
        "6",
        0
      ],
      "control_net": [
        "12",
        0
      ],
      "image": [
        "16",
        0
      ]
    },
    "class_type": "ControlNetApply"
  },
  "11": {
    "inputs": {
      "image": "{{controlnet_image}}",
      "choose file to upload": "image"
    },
    "class_type": "LoadImage"
  },
  "12": {
    "inputs": {
      "control_net_name": "diffusers_xl_canny_full.safetensors"
    },
    "class_type": "ControlNetLoader"
  },
  "14": {
    "inputs": {
      "ckpt_name": "sd_xl_base_1.0.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "15": {
    "inputs": {
      "images": [
        "16",
        0
      ]
    },
    "class_type": "PreviewImage"
  },
  "16": {
    "inputs": {
      "low_threshold": 0.2,
      "high_threshold": 0.6,
      "image": [
        "11",
        0
      ]
    },
    "class_type": "Canny"
  },
  "18": {
    "inputs": {
      "images": [
        "8",
        0
      ]
    },
    "class_type": "PreviewImage"
  }
}

values = {
  "positive_prompt": "An igloo on a snowy day, 4k, hd",
  "negative_prompt": "blurry, text, low quality",
  "controlnet_image": "https://storage.googleapis.com/logos-bucket-01/baseten_logo.png"
}

data = {"json_workflow": sdxl_controlnet_workflow, "values": values}
res = requests.post("https://app.baseten.co/model_versions/<YOUR-MODEL-ID>/predict", headers=headers, json=data)
res = res.json()
model_output = res.get("model_output").get("result")
print(model_output)
```

Here is the output of the request above:

```json
[
    {
        "node_id": "18",
        "image": "base64-image-string"
    },
    {
        "node_id": "15",
        "image": "base64-image-string"
    }
]
```

The output of the model is a list of JSON objects containing the ID of the output node along with the generated image as a base64 string.

