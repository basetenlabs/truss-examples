# Stable Diffusion Video

This Truss allows you to create small videos ( < 10 seconds) using the [Stable Video Diffusion model](https://stability.ai/news/stable-video-diffusion-open-ai-video-model). The model only requires a single image as input and converts it into a video.


https://github.com/htrivedi99/truss-examples/assets/15642666/53d3ea1f-952e-4224-8bd4-3ff95eeac8c0


## Deploying the Truss

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion/stable-video-diffusion
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stable-video-diffusion` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Using the model

The model takes a JSON payload with one required field and 4 optional fields:

- `image` (required): The input image as a base64 string. This model was trained on images that are 1024 × 576 in size. Other image dimensions work as well for the most part, but to get the best results using 1024 × 576 images is recommended.
- `num_frames` (optional) - The total number of frames in the output clip
- `num_steps` (optional) - Steps takes at each iteration
- `fps` (optional) - Frames per second for the clip
- `decoding_t` (optional) - The number of frames decoded per second. This number cannot be greater than 10 as the GPU runs out of memory at that point.

It returns a JSON object with the `output` key containing the generated video as a base64 string.

## Example Usage

Here is how you can invoke the model using Python:

```python
def base64_to_mp4(base64_string, output_file_path):
    binary_data = base64.b64decode(base64_string)
    with open(output_file_path, "wb") as output_file:
        output_file.write(binary_data)


def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        binary_data = image_file.read()
        base64_data = base64.b64encode(binary_data)
        base64_string = base64_data.decode("utf-8")

    return base64_string

data = {
    "image": image_to_base64("path/to/image/cheetah.jpeg"),
    "num_frames": 14,
    "fps": 10,
    "decoding_t": 5
}
headers = {"Authorization": f"Api-Key <BASETEN-API-KEY>"}
res = requests.post("https://model-<model-id>.api.baseten.co/development/predict", headers=headers, json=data)
res = res.json()

base64_output = res.get("output")
base64_to_mp4(base64_output, "output_video.mp4")
```

You can also invoke the model via REST API:

```bash
curl -X POST "https://model-<model-id>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Api-Key {BASETEN-API-KEY}" \
     -d '{"image": "<image-as-b64-string>"}'
```

For inspiration, here are some sample input images you can use:

![image1](sample_images/cheetah.jpeg)
![image2](sample_images/racecar.jpeg)
