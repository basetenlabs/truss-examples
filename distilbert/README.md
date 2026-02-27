# DistilBERT
This truss runs the [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) model as an endpoint on Baseten.

## Deploy
```
pip install --upgrade truss
truss push --publish # grab an api key from https://app.baseten.co/settings/api_keys
```

The deployment will take a few minutes the first. Once it's ready in the you UI you can proceed to calling the API.

## Test
```
truss predict --published -d '{"text": "some text to embed"}'
```