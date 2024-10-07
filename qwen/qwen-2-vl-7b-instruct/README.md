# Qwen-2-VL 7B Instruct vLLM Truss

This is a [Truss](https://truss.baseten.co/) for Qwen-2-VL 7B Instruct with vLLM. Qwen-2-VL 7B Instruct is a multimodal (text + vision) LLM. This README will walk you through how to deploy this Truss on Baseten to get your own instance of it.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd qwen/qwen-2-vl-7b-instruct
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Apply for access to the Qwen-2-VL 7B Instruct model on hugging face [here](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).
4. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
5. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`. Note that you will *not* be able to successfully deploy this model without doing this.

With `qwen-2-vl-7b-instruct` as your working directory, you can deploy the model with:

```sh
truss push --publish --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Notes

If you need to send multiple images to the model, you will need to update the following value in `config.yaml->model_metadata->vllm_config->limit_mm_per_prompt`

## Example usage

```sh
truss predict -d '{"messages": [{"role": "user", "content": "Tell me about yourself"}]}'
```

Here's another example of invoking your model via a REST API but for image input:

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
