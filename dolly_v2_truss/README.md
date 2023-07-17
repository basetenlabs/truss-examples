## Dolly-v2 Truss

This is a [Truss](https://truss.baseten.co/) for Dolly-v2, an instruction-following large language model based on `pythia-12b` . This README will walk you through how to deploy this Truss on Baseten to get your own instance of Dolly-v2.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying Dolly-v2

To deploy the Dolly-v2 Truss, you'll need to follow these steps:

1. __Prerequisites__:
- _Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup)._

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the Dolly-v2 Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

dolly_v2_truss = truss.load("path/to/dolly_v2_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the Dolly-v2 Truss__: Deploy the Dolly-v2 Truss to Baseten with the following command:
```
baseten.deploy(dolly_v2_truss, model_name="Dolly-v2", publish=True)
```

Once your Truss is deployed, you can start using the Dolly-v2 model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

#### Example Usage

You can use the `baseten` model package to invoke your model from Python
```
import baseten
# You can retrieve your deployed model version ID from the UI
model = baseten.deployed_model_version_id('YOUR_MODEL_ID')

request = {
    "prompt": "Explain to me the difference between nuclear fission and fusion."
}

response = model.predict(request)
```

You can also invoke your model via a REST API
```
curl -X POST https://app.baseten.co/model_versions/<YOUR_MODEL_VERSION_ID>/predict \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Explain to me the difference between nuclear fission and fusion."
         }'
```
