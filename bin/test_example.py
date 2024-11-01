import json
import sys
import time

import requests
import truss
from tenacity import (
    Retrying,
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random_exponential,
)
from truss.cli.cli import _get_truss_from_directory
from truss.remote.remote_factory import RemoteConfig, RemoteFactory
from truss.truss_handle import TrussHandle

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


@retry(wait=wait_fixed(30), stop=stop_after_attempt(3), reraise=True)
def attempt_inference(truss_handle, model_version_id, api_key):
    """
    Retry every 20 seconds to call inference on the example, using the `example_model_input`
    from the Truss config to invoke the model. We return success if there is a 200 response,
    and if not retry, ultimately throwing an exception if we don't get a response after 200
    seconds.
    """
    print("Started attempt inference")
    try:
        if "example_model_input" in truss_handle.spec.config.model_metadata:
            example_model_input = truss_handle.spec.config.model_metadata[
                "example_model_input"
            ]
        else:
            example_model_input = json.loads(
                (
                    truss_handle._truss_dir
                    / truss_handle.spec.config.model_metadata[
                        "example_model_input_file"
                    ]
                ).read_text()
            )
    except KeyError:
        raise Exception("No example_model_input defined in Truss config")

    url = f"{BASETEN_HOST}/model_versions/{model_version_id}/predict"
    headers = {"Authorization": f"Api-Key {api_key}"}
    response = requests.post(url, headers=headers, json=example_model_input, timeout=30)

    print(response.content)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}")


def deploy_truss(target_directory: str) -> str:
    for _ in Retrying(
        wait=wait_random_exponential(multiplier=1, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    ):
        model_deployment = truss.push(target_directory, remote=REMOTE_NAME)
    model_deployment.wait_for_active()
    return model_deployment.model_deployment_id


def main(api_key: str, target_directory: str):
    write_trussrc_file(api_key)
    truss_handle = _get_truss_from_directory(target_directory)
    model_version_id = deploy_truss(target_directory)
    print(f"Deployed Truss {model_version_id}")
    attempt_inference(truss_handle, model_version_id, api_key)


if __name__ == "__main__":
    api_key = sys.argv[1]
    target_directory = sys.argv[2]
    main(api_key, target_directory)
