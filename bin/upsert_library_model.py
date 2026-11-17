import json
import sys

import requests
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(wait=wait_fixed(20), stop=stop_after_attempt(3), reraise=True)
def upsert_pretrained_model(api_key, model_slug, commit_sha, baseten_url):
    url = f"{baseten_url}/upsert_pretrained_model"
    headers = {"Authorization": f"Api-Key {api_key}"}
    with open("library.config.yaml", "r") as f:
        library_contents = f.read()
        library_models = yaml.safe_load(library_contents)

    model_to_deploy = library_models[model_slug]

    data = {
        "name": model_to_deploy["name"],
        "slug": model_slug,
        "repo_url": "https://github.com/basetenlabs/truss-examples",
        "sha": commit_sha,
        "path": model_to_deploy["path"],
    }

    requests.post(url, headers=headers, json=data)


if __name__ == "__main__":
    api_key = sys.argv[1]
    model_to_deploy = sys.argv[2]
    commit_sha = sys.argv[3]
    baseten_url = sys.argv[4]

    upsert_pretrained_model(api_key, model_to_deploy, commit_sha, baseten_url)
