import os
from openai import OpenAI

model_id = "YOUR_MODEL_ID"
client = OpenAI(
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1",
    api_key=os.environ["BASETEN_API_KEY"],
)

resp = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-Parse-v1.2",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/michaelfeil/infinity/blob/main/docs/assets/cats_coco_sample.jpg?raw=true",
                    },
                },
            ],
        }
    ],
    max_tokens=8192,
    temperature=0.0,
    extra_body={
        "repetition_penalty": 1.1,
        "top_k": 1,
        "skip_special_tokens": False,
    },
)

print(resp.choices[0].message.content)
