"""example script on how to use the chat template deployment"""

import os
from functools import cache

import requests

MESSAGE_TYPE = list[dict]
HEADERS = {f"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"}
DEPLOYMENT_ID = "03y1d2v3"


@cache
def get_tokenizer():
    """optmization,"""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot_download(
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        local_dir="/tmp/skywork-reward-llama-3.1-8b-v0.2",
        allow_patterns=["*.json", "*.txt"],
    )
    return AutoTokenizer.from_pretrained("/tmp/skywork-reward-llama-3.1-8b-v0.2")


def apply_chat_template(messages: MESSAGE_TYPE) -> str:
    tokenizer = get_tokenizer()
    with_template = tokenizer.apply_chat_template(messages, tokenize=False)
    return with_template


def send_to_deployment(messages: MESSAGE_TYPE):
    templated = apply_chat_template(messages)
    response = requests.post(
        headers=HEADERS,
        url=f"https://model-{DEPLOYMENT_ID}.api.baseten.co/environments/production/sync/predict",
        json={
            "inputs": templated,
            "raw_scores": True,
            "truncate": True,
            "truncation_direction": "Right",
        },
    )
    if response.status_code != 200:
        raise Exception(f"Failed to send to deployment: {response.text}")
    return response.json()


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    send_to_deployment(messages)
