# NexusRaven V2 Truss

NexusRaven is an open-source and commercially viable function calling LLM that surpasses the state-of-the-art in function calling capabilities. This README covers deploying and invoking this model.

This model is packaged using [Truss](https://trussml.com), the simplest way to serve AI/ML models in production.

## Deploy NexusRaven V2

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd nexusraven-v2-13b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `nexusraven-v2-13b` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using NexusRaven V2 through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Calling functions via NexusRaven


NexusRaven returns the function to be called as a string. In python you can run them using `eval` as shown in the following example.

```python
import requests

resp = requests.post(
    "BASETEN_API_ENDPOINT",
    headers={"Authorization": "Api-Key DEPLOYMENT_API_KEY"},
    json={'prompt': 'Function:\ndef get_weather_data(coordinates):\n    """\n    Fetches weather data from the Open-Meteo API for the given latitude and longitude.\n\n    Args:\n    coordinates (tuple): The latitude and longitude of the location.\n\n    Returns:\n    float: The current temperature in the coordinates you\'ve asked for.\n    """\n\nFunction:\ndef get_coordinates_from_city(city_name):\n    """\n    Fetches the latitude and longitude of a given city name using the Maps.co Geocoding API.\n\n    Args:\n    city_name (str): The name of the city.\n\n    Returns:\n    tuple: The latitude and longitude of the city.\n    """\n\nUser Query: What\'s the weather like in Seattle right now?<human_end>\n'},
)

functions_to_be_called = resp.json() # get_weather_data(coordinates=get_coordinates_from_city(city_name='Seattle'))

eval(functions_to_be_called)
```