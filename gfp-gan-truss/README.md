[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/gfp_gan)

# GFP-GAN Truss

This is a [Truss](https://truss.baseten.co/) for serving an implementation of TencentARC
[GFPGAN](https://github.com/TencentARC/GFPGAN). GFPGAN is an algorithm for real-world face restoration.
It can be used on old photos of faces to remove blur, and increase clarity and resolution.

It leverages rich and diverse priors encapsulated in a pretrained face GAN (e.g., StyleGAN2) for
"blind face" restoration.

## Deploying GFP-GAN

To deploy the GFP-GAN Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the GFP-GAN Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

gfp_gan_truss = truss.load("path/to/gfp_gan_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the GFP-GAN Truss__: Deploy the GFP-GAN Truss to Baseten with the following command:
```
baseten.deploy(gfp_gan_truss)
```

Once your Truss is deployed, you can start using the GFP-GAN model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

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

You can invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```
$ curl -X POST https://app.baseten.co/models/{MODEL_VERSION_ID}/predict \
    -H 'Authorization: Api-Key {YOUR_API_KEY}' \
    -d '{"image": "{BASE_64_INPUT}"}'
```
