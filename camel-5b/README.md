## Camel-5B Truss

This is a [Truss](https://truss.baseten.co/) for Camel-5B, a 5 billion parameter model trained by [Writer](https://writer.com/). This README will walk you through how to deploy this Truss on Baseten to get your own instance of Camel-5B.

## Deploying Camel

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd camel-5b-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `camel-5b-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Camel-5B API Documentation
This section provides an overview of the Camel-5B API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided instruction.

### API Route: `predict`
The predict route is the primary method for generating text completions based on a given instruction. It takes several parameters:

- __instruction__: The instruction text that you want the model to follow.
- __input__ (optional): The input text provided by the user that is referenced in the `instruction` value.
- __temperature__ (optional): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional: The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __num_beams__ (optional): The number of beams used for beam search. Increasing this value can result in higher-quality output but will increase the computational cost.

The API also supports passing any parameter supported by Huggingface's `Transformers.generate`.

#### Example Usage

```sh
truss predict -d '{"instruction": "Write a story about a new revolutionary space technology"}'
```

The response will follow the format:

```
{
    'completion': "In a world where humans have colonized the moon, a brilliant scientist discovers a hidden chamber in the lunar crust that holds the key to unlocking the secrets of the universe. Together with a daring team of astronauts, they embark on a daring mission to explore the chamber and unlock its incredible potential. As they venture through the uncharted regions of the moon's surface, they encounter unexpected challenges and uncover a hidden world teeming with life. As the team races against time to return home and share their discovery with the world, they must confront the consequences of their actions and decide whether to embrace a new age of space exploration or return to the safety of Earth."
}
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -d '{
        "instruction": "Write a story about a new revolutionary space technology"
         }'

```
