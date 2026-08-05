This is an implementation of a JSON-mode for small LLMs, using a combination of a fine-tuned Mistral 7B, Hermes 2 Pro, and Jsonformers. 

Hermes 2 Pro is finetuned from Mistral's 7b-v0.1 model, incorporating a newly developed Function Calling and JSON Mode dataset provided by Nous Research. As a result, Hermes is finetuned to better perform for both function calling as well as general structured data tasks. It was decided to go with the Hermes 2 Pro model over the base Mistral 7B due to its fantastic performance on structured JSON Output, achieving 84% on the evaluation created in partnership with Fireworks.AI. More information about the model and its development can be found on its HuggingFace card: https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B

In order to further mitigate the risk of hallucination, we use the open-source library Jsonformer (https://github.com/1rgs/jsonformer/?tab=readme-ov-file). Jsonformer is a wrapper around Hugging Face models that fill in the _fixed_ tokens during the generation process, delegating only the task of generating the content tokens to the language model. As a result, the generated JSON will always be syntatically correct (as there is no opportunity for hallucinations thanks to the separation of concerns) with a high overall efficiency as only the content tokens need to be generated, not an entire JSON string. By wrapping Hermes with Jsonformer, we hope to prevent any possibility of malformed or invalid JSON structure while increasing model performance and speed on content token generation.

The modifications I made to the Model class structure are the addition of a `schema` parameter, to allow the user to specify the desired JSON schema for generation, as well as adding a `latency_metrics` dictionary which records various metrics related to the latency of the model, namely prefill time, time to first token, time per output token, and total generation time.

Although the model curently uses an LLM finetuned for the task of constrained decoding, due to wrapping the model in Jsonformer, it is possible to switch between various models for domain-specifc tasks (e.g. a JSON of medical information). As such, it should be quite easy to generalize, with the default model selected to optimize performance across a broad set of domains. 

A preliminary assessment of this model against the baseline model, Mistral-7B-v0.1, showed immensely promising results. Given the following schema, 
```json
car = {
  "type": "object",
  "properties": {
    "car": {
      "type": "object",
      "properties": {
        "make": {"type": "string"},
        "model": {"type": "string"},
        "year": {"type": "number"},
        "colors": {
          "type": "array",
          "items": {"type": "string"}
        },
        "features": {
          "type": "object",
          "properties": {
            "audio": {
              "type": "object",
              "properties": {
                "brand": {"type": "string"},
                "speakers": {"type": "number"},
                "hasBluetooth": {"type": "boolean"}
              }
            },
            "safety": {
              "type": "object",
              "properties": {
                "airbags": {"type": "number"},
                "parkingSensors": {"type": "boolean"},
                "laneAssist": {"type": "boolean"}
              }
            },
            "performance": {
              "type": "object",
              "properties": {
                "engine": {"type": "string"},
                "horsepower": {"type": "number"},
                "topSpeed": {"type": "number"}
              }
            }
          }
        }
      }
    },
    "owner": {
      "type": "object",
      "properties": {
        "firstName": {"type": "string"},
        "lastName": {"type": "string"},
        "age": {"type": "number"},
      }
    }
  }
}
```
the models were asked to generate an example car. The Hermes-2-Pro model + Jsonformer were able to successfully generate an example in __1min 4s ± 267 ms per loop__ (mean ± std. dev. of 7 runs, 1 loop each):
```json
{
  car: {
    make: "Toyota",
    model: "Corolla",
    year: 2020.5,
    colors: [
      "white",
      "silver",
      "gray",
      "blue",
      "black",
      "red",
      "green",
      "yellow",
      "orange",
      "purple"
    ],
    features: {
      audio: {
        brand: "JBL",
        speakers: 12.123,
        hasBluetooth: True
      },
      safety: {
        airbags: 7.8989,
        parkingSensors: True,
        laneAssist: True
      },
      performance: {
        engine: "4-Cylinder Turbocharged E",
        horsepower: 184.42,
        topSpeed: 145.02
      }
    }
  },
  owner: {
    firstName: "John",
    lastName: "Doe",
    age: 38.456
  }
}
```

Mistral, on the other hand, was unable to successfully generate an example (instead creating a false accident report) and took **3min 18s ± 75.3 ms per loop** (mean ± std. dev. of 7 runs, 1 loop each):

```
Car Accident Report

Date: [Insert Date]
Time: [Insert Time]
Location: [Insert Address]

Driver 1:
Name: [Insert Name]
Age: [Insert Age]
Gender: [Insert Gender]
Address: [Insert Address]
Phone: [Insert Phone Number]

Driver 2:
Name: [Insert Name]
Age: [Insert Age]
Gender: [Insert Gender]
Address: [Insert Address]
Phone: [Insert Phone Number]

Vehicle 1:
Make: [Insert Make]
Model: [Insert Model]
Year: [Insert Year]
Color: [Insert Color]
License Plate Number: [Insert License Plate Number]

Vehicle 2:
Make: [Insert Make]
Model: [Insert Model]
Year: [Insert Year]
Color: [Insert Color]
License Plate Number: [Insert License Plate Number]

Accident Summary:

On [Insert Date] at [Insert Time], a car accident occurred at [Insert Address]. The accident involved two vehicles, a [Insert Make] [Insert Model] [Insert Year] [Insert Color] with license plate number [Insert License Plate Number], driven by [Insert Name], and a [Insert Make] [Insert Model] [Insert Year] [Insert Color] with license plate number [Insert License Plate Number], driven by [Insert Name].

The accident occurred when Driver 1, who was traveling northbound on [Insert Road], failed to yield the right of way to Driver 2, who was traveling eastbound on [Insert Road]. The two vehicles collided at the intersection of [Insert Road] and [Insert Road], causing damage to both vehicles.

There were no injuries reported as a result of the accident.

Witnesses to the accident include [Insert Witness 1 Name], [Insert Witness 2 Name], and [Insert Witness 3 Name].

The investigation into the accident is ongoing.
```

This model is both more accurate as well as efficient when compared to its base, as a result both of the fine-tuning, allowing the model to more effectively handle and understand JSON, as well as the constrained decoding methodology of Jsonformer which allowed for a separation of concerns between schema and output.