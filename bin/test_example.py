import json
import sys
from typing import Any

import packaging.version
import requests
import truss
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential
from truss.cli.cli import _get_truss_from_directory
from truss.remote.baseten import BasetenRemote
from truss.remote.baseten.api import BasetenApi
from truss.remote.remote_factory import RemoteConfig, RemoteFactory

REMOTE_NAME = "ci"
BASETEN_HOST = "https://app.staging.baseten.co"


def write_trussrc_file(api_key: str):
    ci_user = RemoteConfig(
        name=REMOTE_NAME,
        configs={
            "api_key": api_key,
            "remote_url": BASETEN_HOST,
            "remote_provider": "baseten",
        },
    )
    RemoteFactory.update_remote_config(ci_user)


@retry(wait=wait_fixed(30), stop=stop_after_attempt(8), reraise=True)
def attempt_inference(truss_handle, model_id, model_version_id, api_key):
    """
    Retry every 20 seconds to call inference on the example, using the `example_model_input`
    from the Truss config to invoke the model. We return success if there is a 200 response,
    and if not retry, ultimately throwing an exception if we don't get a response after 200
    seconds.
    """
    print("Started attempt inference")

    if "example_model_input" in truss_handle.spec.config.model_metadata:
        example_model_input = truss_handle.spec.config.model_metadata[
            "example_model_input"
        ]
    elif "example_model_input_file" in truss_handle.spec.config.model_metadata:
        example_model_input = json.loads(
            (
                truss_handle._truss_dir
                / truss_handle.spec.config.model_metadata["example_model_input_file"]
            ).read_text()
        )
    else:
        raise Exception("No example_model_input defined in Truss config")

    url = f"https://model-{model_id}.api.staging.baseten.co/deployment/{model_version_id}/predict"

    headers = {"Authorization": f"Api-Key {api_key}"}
    response = requests.post(url, headers=headers, json=example_model_input, timeout=30)

    print(response.content)
    response.raise_for_status()


@retry(
    wait=wait_random_exponential(multiplier=1, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
def deploy_truss(target_directory: str) -> str:
    model_deployment = truss.push(
        target_directory, remote=REMOTE_NAME, trusted=True, publish=True
    )
    model_deployment.wait_for_active(timeout_seconds=30 * 60)
    return model_deployment.model_id, model_deployment.model_deployment_id


def delete_model_deployment(api: BasetenApi, model_id: str, deployment_id: str) -> Any:
    url = f"{api._rest_api_url}/v1/models/{model_id}/deployments/{deployment_id}"
    headers = api._auth_token.header()
    resp = requests.delete(url, headers=headers)
    if not resp.ok:
        resp.raise_for_status()

    deployment = resp.json()
    return deployment


def clean_up_deployments(model_name: str, model_id: str):
    remote: BasetenRemote = RemoteFactory.create(REMOTE_NAME)
    model = remote.api.get_model(model_name)
    versions = [
        (version["id"], version["semver"])
        for version in model["model"]["versions"]
        if not version["is_primary"]
    ]
    versions = sorted(versions, key=lambda x: packaging.version.Version(x[1]))
    to_delete = versions[:-5]  # Keep the last 5 versions
    for deployment_id, _ in to_delete:
        print(f"Deleting {model_id}:{deployment_id}")
        delete_model_deployment(remote.api, model_id, deployment_id)


def main(api_key: str, target_directory: str):
    write_trussrc_file(api_key)
    truss_handle = _get_truss_from_directory(target_directory)
    model_id, model_version_id = deploy_truss(target_directory)
    print(f"Deployed Truss {model_version_id}")
    try:
        attempt_inference(truss_handle, model_id, model_version_id, api_key)
    finally:  # If inference fails, still the last N versions are left deployed.
        try:
            clean_up_deployments(truss_handle.spec.config.model_name, model_id)
        except Exception as e:
            print(f"Error when deleting old deployments: {e}")


if __name__ == "__main__":
    main(api_key=sys.argv[1], target_directory=sys.argv[2])
