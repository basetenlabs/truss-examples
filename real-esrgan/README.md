# Real-ESRGAN Truss

This is a [Truss](https://truss.baseten.co/) for Real-ESRGAN which is an AI image upscaling model.
Open-source image generation models like Stable Diffusion 1.5 can sometime produce blurry or low resolution images. Using Real-ESRGAN, those low quality images can be upscaled making them look sharper and more detailed.

## Deployment

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd real-esrgan-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `real-esrgan-truss` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## API route: `predict`
The predict route is the primary method for upscaling an image. In order to send the image to our model, the image must first be converted into a base64 string.

- __image__: The image converted to a base64 string


## Invoking the model

```sh
truss predict -d '{"image": "<BASE64-STRING-HERE>"}'
```

You can also use python to call the model:

```python
BASE64_PREAMBLE = "data:image/png;base64,"

def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

img = Image.open("/path/to/image/ship.jpeg")
b64_img = pil_to_b64(img)

headers = {"Authorization": f"Api-Key <BASETEN-API-KEY>"}
data = {"image": b64_img}
res = requests.post("https://model-{MODEL_ID}.api.baseten.co/development/predict", headers=headers, json=data)
output = res.json()

result_b64 = output.get("model_output").get("upscaled_image")
pil_img = b64_to_pil(result_b64)
pil_img.save("upscaled_output_img.png")
```

The model returns a JSON object containing the key `upscaled_image`, which is the upscaled image as a base64 string.

## Results

<div style="display: flex; justify-content: space-between;">
    <div style="flex: 1; margin-right: 10px;">
        <img src="ship.jpeg" alt="original image" style="width: 100%;">
        <p>Original Image Stable Diffusion 1.5</p>
    </div>
    <div style="flex: 1;">
        <img src="result_image.jpeg" alt="upscaled image" style="width: 100%;">
        <p>Upscaled Image</p>
    </div>
</div>

<div style="display: flex; justify-content: space-between;">
    <div style="flex: 1; margin-right: 10px;">
        <img src="racecar.jpeg" alt="original image" style="width: 100%;">
        <p>Original Image SDXL</p>
    </div>
    <div style="flex: 1;">
        <img src="racecar_upscaled.jpeg" alt="upscaled image" style="width: 100%;">
        <p>Upscaled Image</p>
    </div>
</div>
