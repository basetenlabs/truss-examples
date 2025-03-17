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
    prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
    response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
    response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

    for response in [response1, response2]:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        print(send_to_deployment(messages))
