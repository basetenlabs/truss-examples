[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/llama)

# Llama-2 70B Chat Truss

This is a [Truss](https://truss.baseten.co/) for Llama-2 70B Chat. Llama-2 is a family of language models released by Meta. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Llama-2 70B Chat.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying Llama-2 70B Chat

To deploy the Llama-2 70B Chat Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Set HF API key__: To download the model, you'll need to pass your HF API key and be approved to download weights. Visit this [link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) to begin the approval process. Once approved, update the `config.yaml` with your Huggingface API key (look for the key, `hf_api_key` in the config.yaml.)

4. __Load the Llama-2 70B Chat Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

llama_2_70b_truss = truss.load("path/to/llama_2_70b_truss")
```

5. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

6. __Deploy the Llama-2 70B Chat Truss__: Deploy the Llama-2 70B Chat Truss to Baseten with the following command:
```
baseten.deploy(llama_2_70b_truss)
```

Once your Truss is deployed, you can start using the Llama-2 70B Chat model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Llama-2 70B Chat API documentation
This section provides an overview of the Llama-2 70B Chat API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided instruction.

### API route: `predict`
The predict route is the primary method for generating text completions based on a given instruction. It takes several parameters:

- __instruction__: The input text that you want the model to generate a response for.
- __temperature__ (optional, default=0.1): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.75): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=40): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __num_beams__ (optional, default=4): The number of beams used for beam search. Increasing this value can result in higher-quality output but will increase the computational cost.

The API also supports passing any parameter supported by Huggingface's `Transformers.generate`.

## Example usage

You can use the `baseten` model package to invoke your model from Python
```
import baseten
# You can retrieve your deployed model ID from the UI
model = baseten.deployed_model_version_id('YOUR_MODEL_ID')

request = {
    "prompt": "What's the meaning of life?",
    "temperature": 0.1,
    "top_p": 0.75,
    "top_k": 40,
    "num_beams": 4,
}

response = model.predict(request)
```

You can also invoke your model via a REST API
```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What's the meaning of life?",
           "temperature": 0.1,
           "top_p": 0.75,
           "top_k": 40,
           "num_beams": 4
         }'

```
