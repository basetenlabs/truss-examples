import os

import requests


def delete_all_truss_example_deployments():

    API_KEY = os.environ["BASETEN_API_KEY"]
    url = "https://api.baseten.co/v1/models"

    headers = {"Authorization": f"Api-Key {API_KEY}"}

    response = requests.request("GET", url, headers=headers, json={}).json()

    to_delete = []
    for model in response["models"]:
        if model["name"].startswith("BEI") and "-truss-example" in model["name"]:
            to_delete.append(model["id"])

    for delete_id in to_delete:
        url_delete = f"https://api.baseten.co/v1/models/{delete_id}"

        headers = {"Authorization": f"Api-Key {API_KEY}"}

        response_delete = requests.request(
            "DELETE", url_delete, headers=headers, json={}
        )

        print(response_delete.text)
    print("Deleted all truss example deployments")


def deploy_all():
    # This function is not implemented in the snippet
    # for DIR in ./truss-examples/11-embeddings-reranker-classification-tensorrt/*/; do [ -f "$DIR/config.yaml" ] && (cd "$DIR" && truss push --publish --remote baseten --promote); done
    pass


if __name__ == "__main__":
    delete_all_truss_example_deployments()
