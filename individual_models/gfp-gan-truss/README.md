[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/gfp_gan)

# GFP-GAN Truss

This is a [Truss](https://truss.baseten.co/) for serving an implementation of TencentARC
[GFPGAN](https://github.com/TencentARC/GFPGAN). GFPGAN is an algorithm for real-world face restoration.
It can be used on old photos of faces to remove blur, and increase clarity and resolution.

It leverages rich and diverse priors encapsulated in a pretrained face GAN (e.g., StyleGAN2) for
"blind face" restoration.

## Deploying GFP-GAN

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd gfp-gan-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `gfp-gan-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## GFP-GAN API documentation

### Input

The input should be a dictionary with the following key:
* `image` - the image to be restored, encoded as base64.

### Output

The model returns a dictionary containing the base64-encoded restored image:
* `status` - either `success` or `failed`
* `data` - the restored image, encoded as base64
* `message` - will contain details in the case of errors


## Example usage

```sh
truss predict -d '{"image": "{BASE_64_INPUT}"}'
```

You can also invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```
$ curl -X POST https://app.baseten.co/models/{MODEL_VERSION_ID}/predict \
    -H 'Authorization: Api-Key {YOUR_API_KEY}' \
    -d '{"image": "{BASE_64_INPUT}"}'
```
