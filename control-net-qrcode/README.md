# Control Net QR Code Truss

This truss allows you to generate QR codes using Stable Diffusion and Control Net. By typing in a prompt using the same guidelines as stable diffusion, a new image gets created that combines the image from the prompt with the image of a qr code. For this truss [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) is used along with [Control QR Code Monster](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster) as the control net. 

![controlnet_qr_code](controlnet_qr_code_results.gif)

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd control-net-qrcode
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `control-net-qrcode` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

This two models combined only take up about 5 GB of VRAM so a T4 is enough for this truss. 

### API route: `predict`

The predict route is the primary method for generating images based on a given prompt. It takes several parameters:

- __prompt__ (required): The input text required for image generation.
- __negative_prompt__ (optional, default=""): Use this to refine the image generation by discarding unwanted items.
- __guidance_scale__ (optional, default=7.5): Used to control image generation.
- __condition_scale__ (optional, default=1.2): The lower the condition_scale, the more creative the results. A higher condition_scale will result in less creative and more scannable QR Codes. 
- __sampler__(optional, default="Euler a"): This controls which sampler to use resulting in more image variations.

## Example usage

```sh
truss predict -d '{"prompt": "A cubism painting of the Garden of Eaden with animals walking around, Andreas Rocha, matte painting concept art, a detailed matte painting"}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "A cubism painting of the Garden of Eaden with animals walking around, Andreas Rocha, matte painting concept art, a detailed matte painting"
         }'
```
