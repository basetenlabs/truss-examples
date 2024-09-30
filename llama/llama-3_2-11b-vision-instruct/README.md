# Llama 3.2 11B Vision Instruct VLLM Truss

This is a [Truss](https://truss.baseten.co/) for Llama 3.2 11B Vision Instruct with VLLM. Llama 3.2 11B Vision Instruct is a multimodal (text + vision) LLM. This README will walk you through how to deploy this Truss on Baseten to get your own instance of it.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd llama/llama-3_2-11b-vision-instruct
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Apply for access to the Llama 3.2 11B Vision Instruct model on hugging face [here](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).
4. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
5. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`. Note that you will *not* be able to successfully deploy this model without doing this.

With `llama-3_2-11b-vision-instruct` as your working directory, you can deploy the model with:

```sh
truss push --publish --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Notes

Limitations from VLLM allow for a maximum of 1 image as input. You will get a memory error otherwise. You can keep track of the issue [here](https://github.com/vllm-project/vllm/issues/8826).

## Example usage

```sh
truss predict -d '{"messages": [{"role": "user", "content": "Tell me about yourself"}]}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "What type of animal is this? Answer in French only"
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": "https://vetmed.illinois.edu/wp-content/uploads/2021/04/pc-keller-hedgehog.jpg"
                    }
                }
                ]
            }
            ],
           "stream": true,
           "max_tokens": 64,
           "temperature": 0.2
         }'
```
