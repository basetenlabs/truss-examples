## Camel-5B Truss

This is a [Truss](https://truss.baseten.co/) for Camel-5B, a 5 billion parameter model trained by [Writer](https://writer.com/). This README will walk you through how to deploy this Truss on Baseten to get your own instance of Camel-5B.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp)). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying Camel

To deploy the Camel-5B Truss, you'll need to follow these steps:

1. __Prerequisites__: 
- _Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup)._
- _You'll also need a Huggingface [API key](https://huggingface.co/settings/tokens)_

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Set Huggingface API key__: Assuming you've cloned this repo, navigate to the `config.yaml` and set the key `hf_access_token` to your Huggingface API key. 

4. __Load the Camel-5B Truss__: Spin up an IPython shell and load the Truss into memory:
```
import truss

camel5b_truss = truss.load("path/to/camel-5b-truss")
```

5. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

6. __Deploy the Camel-5B Truss__: Deploy the Camel-5B Truss to Baseten with the following command:
```
baseten.deploy(camel5b_truss, model_name="Camel-5b", publish=True)
```

Once your Truss is deployed, you can start using the Camel-5B model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

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

You can use the `baseten` model package to invoke your model from Python
```
import baseten
# You can retrieve your deployed model ID from the UI
model = baseten.deployed_model_version_id('YOUR_MODEL_ID')

request = {
    "instruction": "Write a story about a new revolutionary space technology"
}

response = model.predict(request)
```

The response will follow the format
```
{'completion': "In a world where humans have colonized the moon, a brilliant scientist discovers a hidden chamber in the lunar crust that holds the key to unlocking the secrets of the universe. Together with a daring team of astronauts, they embark on a daring mission to explore the chamber and unlock its incredible potential. As they venture through the uncharted regions of the moon's surface, they encounter unexpected challenges and uncover a hidden world teeming with life. As the team races against time to return home and share their discovery with the world, they must confront the consequences of their actions and decide whether to embrace a new age of space exploration or return to the safety of Earth."}
```

You can also invoke your model via a REST API
```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -d '{
        "instruction": "Write a story about a new revolutionary space technology"
         }'

```

## Contributing

We're always looking for contributors and we're excited to welcome contributions. Some guidelines to follow: 

1. Choose an issue to work on: Check the project's issue tracker for open issues that you can work on. Look for issues that are labeled "help wanted" or "good first issue" if you're new to the project.

2. Discuss your ideas: Before starting to work on an issue or feature, it's a good idea to discuss your plans with the project maintainers. Leave a comment on the issue, or create a new issue if one doesn't exist for your proposed changes.

3. When your PR is ready, tag one of the maintainers of this repository (@aspctu, @amiruci, or @bolasim) forr a review. Do not commit to `main` directly, all PRs must go through a review process. 

By following these guidelines, you can contribute effectively to the project and help maintain a high-quality codebase. Thank you for your interest in contributing to our project!
