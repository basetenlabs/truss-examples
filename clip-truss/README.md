[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/clip)

# CLIP Truss

This is a [Truss](https://truss.baseten.co/) for serving an implementation of the
[CLIP](https://github.com/openai/CLIP)(Contrastive Language-Image Pre-Training)
neural network that has been trained on image and text pairs. This allows for matching of an image with the most
relevant provided labels, without being specifically trained for that task.

Packaging this model in a Truss makes it easy to deploy it on hosting providers.

## Deploying CLIP

To deploy the CLIP Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the CLIP Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

clip_truss = truss.load("path/to/clip_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the CLIP Truss__: Deploy the CLIP Truss to Baseten with the following command:
```
baseten.deploy(clip_truss)
```

Once your Truss is deployed, you can start using the CLIP model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## CLIP API documentation

### Input

The input should be a dictionary. It should contain the following:

* `labels` - a list of strings representing the labels to apply

And one of:

* `image` - a 3 dimensional list in RGB representation, or
* `image_url` - a URL to of an image to be used in the model.

### Output

The result will a dictionary keyed by label with corresponding prediction scores.

## Example usage

You can invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```
$ curl -X POST https://app.baseten.co/model_versions/{MODEL_VERSION_ID}/predict \
    -H 'Authorization: Api-Key {YOUR_API_KEY}' \
    -d '{"image_url": "https://source.unsplash.com/gKXKBY-C-Dk/300x300", "labels": ["small cat", "not cat", "big cat"]}'
```
