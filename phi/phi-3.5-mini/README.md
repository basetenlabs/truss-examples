# Phi 3.5 Mini Instruct

This is a [Truss](https://truss.baseten.co/) example using our general purpose [vLLM Template](https://github.com/basetenlabs/truss-examples/tree/main/vllm) but for Phi-3.5-Mini-instruct, one of the [compatible chat completion models](https://docs.vllm.ai/en/latest/models/supported_models.html).

> Note: `prefix_caching` is not supported by vLLM for this model, please do not include `enable_prefix_caching` as part of the vllm config in `config.yaml`



## Deploy your Truss

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. With `vllm` as your working directory, you can deploy the model with:

    ```sh
    truss push --trusted
    ```

    Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Call your model

Once your deployment is up, there are [many ways](https://docs.baseten.co/invoke/quickstart) to call your model.

### curl command

#### If you are NOT using OpenAI compatible server

```
curl -X POST https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict \
     -H "Authorization: Api-Key $BASETEN_API_KEY" \
     -d '{"prompt": "what is the meaning of life"}'
```


#### If you are using OpenAI compatible server

```
curl -X POST "https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {BASETEN_API_KEY}' \
     -d '{
           "messages": [{"role": "user", "content": "What even is AGI?"}],
           "max_tokens": 256
         }'
```

To access [production metrics](https://docs.vllm.ai/en/latest/serving/metrics.html) reported by OpenAI compatible server, simply add `metrics: true` to the request.

```
curl -X POST "https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {BASETEN_API_KEY}' \
     -d '{
           "metrics": true
         }'
```

### OpenAI SDK (if you are using OpenAI compatible server)

```
from openai import OpenAI
import os

model_id = "a2345678" # Replace with your model ID

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/{model_id}/v1/direct"
)

response = client.chat.completions.create(
  model="microsoft/Phi-3.5-mini-instruct",
  messages=[
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)

```

For more information, see [API reference](https://docs.baseten.co/api-reference/openai).

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.
