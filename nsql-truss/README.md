[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/nsql)

# NSQL Truss

This is a [Truss](https://truss.baseten.co/) for [Number Station](https://www.numbersstation.ai/)'s 350M parameter NSQL model. NSQL is a text-to-SQL foundation model, enabling users to query their databases using natual language. There are also 2B and 6B NSQL variants available, which you can alternatively deploy by editing the HuggingFace paths in `model/model.py`.

This README will walk you through how to deploy this Truss on Baseten to get your own instance of NSQL 350M.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying NSQL

To deploy the NSQL Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the NSQL Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

nsql_truss = truss.load("path/to/nsql_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the NSQL Truss__: Deploy the NSQL Truss to Baseten with the following command:
```
baseten.deploy(nsql_truss)
```

Once your Truss is deployed, you can start using the NSQL model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## NSQL API documentation
This section provides an overview of the NSQL API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided instruction.

### API route: `predict`
The predict route is the primary method for generating text completions based on a given instruction. It takes several parameters:

- __schema__: An SQL schema for the table you want to query. You can provide multiple schemas as a single string.
- __query__: A natural language query over the provided database schemas.

## Example usage

You can use the `baseten` model package to invoke your model from Python
```
import baseten
# You can retrieve your deployed model ID from the UI
model = baseten.deployed_model_version_id('YOUR_MODEL_ID')

schema = """CREATE TABLE stadium (
    stadium_id number,
    location text,
    name text,
    capacity number,
    highest number,
    lowest number,
    average number
)

CREATE TABLE singer (
    singer_id number,
    name text,
    country text,
    song_name text,
    song_release_year text,
    age number,
    is_male others
)

CREATE TABLE concert (
    concert_id number,
    concert_name text,
    theme text,
    stadium_id text,
    year text
)

CREATE TABLE singer_in_concert (
    concert_id number,
    singer_id text
)"""

request = {
    "schema": schema,
    "query": "What is the maximum, the average, and the minimum capacity of stadiums?"
}

response = model.predict(request)
```
