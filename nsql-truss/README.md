[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/nsql)

# NSQL Truss

This is a [Truss](https://truss.baseten.co/) for [Number Station](https://www.numbersstation.ai/)'s 350M parameter NSQL model. NSQL is a text-to-SQL foundation model, enabling users to query their databases using natual language. There are also 2B and 6B NSQL variants available, which you can alternatively deploy by editing the HuggingFace paths in `model/model.py`.

This README will walk you through how to deploy this Truss on Baseten to get your own instance of NSQL 350M.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying NSQL

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd nsql-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `nsql-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

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
