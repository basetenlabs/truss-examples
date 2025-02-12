import argparse
import functools
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import requests


def retry(max_retries=3, delay=2):
    """
    A simple retry decorator.

    Parameters:
        max_retries (int): Maximum number of attempts.
        delay (int|float): Delay between attempts in seconds.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(
                        f"Error in {func.__name__}: {e}. "
                        f"Retrying in {delay} seconds (attempt {attempt}/{max_retries})"
                    )
                    time.sleep(delay + max_retries * delay)
            # If all attempts fail, raise the last exception.
            raise last_exception

        return wrapper

    return decorator


def matches_name(model: dict, key: str = "name") -> bool:
    return model[key].startswith("BEI") and "-truss-example" in model[key]


@retry(max_retries=3, delay=2)
def delete_model(delete_id: str, api_key: str) -> None:
    url_delete = f"https://api.baseten.co/v1/models/{delete_id}"
    headers = {"Authorization": f"Api-Key {api_key}"}
    response = requests.request("DELETE", url_delete, headers=headers, json={})
    # Raise an exception if the deletion was unsuccessful.
    if response.status_code not in (200, 204):
        raise Exception(f"Failed to delete {delete_id}: {response.text}")
    print(f"Deleted {delete_id}: {response.text}")


def delete_all_truss_example_deployments():
    API_KEY = os.environ["BASETEN_API_KEY"]
    url = "https://api.baseten.co/v1/models"
    headers = {"Authorization": f"Api-Key {API_KEY}"}

    response = requests.request("GET", url, headers=headers, json={}).json()
    to_delete = [
        model["id"] for model in response.get("models", []) if matches_name(model)
    ]

    # Use a thread pool and map to delete concurrently.
    with ThreadPoolExecutor(8) as executor:
        delete_func = partial(delete_model, api_key=API_KEY)
        list(executor.map(delete_func, to_delete))

    print("Deleted all truss example deployments")


@retry(max_retries=3, delay=5)
def deploy_config(path: Path) -> None:
    import truss
    import yaml

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    # Check using key "model_name" for deployment
    if matches_name(config, key="model_name"):
        print("Deploying", path)
        truss.push(
            path.parent.as_posix(),
            remote="baseten",
            publish=True,
            promote=True,
        )


def deploy_all():
    paths = list(Path(__file__).parent.parent.rglob("config.yaml"))
    # Use a thread pool and map to deploy concurrently.
    with ThreadPoolExecutor(4) as executor:
        list(executor.map(deploy_config, paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy or delete Truss example deployments."
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["deploy", "delete"],
        required=True,
        help="Action to perform: deploy or delete.",
    )

    args = parser.parse_args()

    if args.action == "deploy":
        deploy_all()
    elif args.action == "delete":
        delete_all_truss_example_deployments()
