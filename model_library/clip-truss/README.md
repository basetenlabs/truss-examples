[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/clip)

# CLIP Truss

This is a [Truss](https://truss.baseten.co/) for serving an implementation of the
[CLIP](https://github.com/openai/CLIP)(Contrastive Language-Image Pre-Training)
neural network that has been trained on image and text pairs. This allows for matching of an image with the most
relevant provided labels, without being specifically trained for that task.

Packaging this model in a Truss makes it easy to deploy it on hosting providers.

## Deploying CLIP

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd clip-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `clip-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

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

```sh
truss predict -d '{"image_url": "https://source.unsplash.com/gKXKBY-C-Dk/300x300", "labels": ["small cat", "not cat", "big cat"]}'
```

You can also invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```sh
$ curl -X POST https://app.baseten.co/model_versions/{MODEL_VERSION_ID}/predict \
    -H 'Authorization: Api-Key {YOUR_API_KEY}' \
    -d '{"image_url": "https://source.unsplash.com/gKXKBY-C-Dk/300x300", "labels": ["small cat", "not cat", "big cat"]}'
```
