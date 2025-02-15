import argparse
import functools
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import requests

REMOTE = "baseten"
FILTER = "-"
API_KEY = os.environ["BASETEN_API_KEY"]


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


def matches_name(model: dict, key: str = "name", filtered: str = "-") -> bool:
    return (
        model[key].endswith("truss-example")
        and filtered in model[key]
        and (not "405b" in model[key])
    )


@retry(max_retries=3, delay=2)
def delete_model(delete_id: str) -> None:
    url_delete = f"https://api.baseten.co/v1/models/{delete_id}"
    headers = {"Authorization": f"Api-Key {API_KEY}"}
    response = requests.request("DELETE", url_delete, headers=headers, json={})
    # Raise an exception if the deletion was unsuccessful.
    if response.status_code not in (200, 204):
        raise Exception(f"Failed to delete {delete_id}: {response.text}")
    print(f"Deleted {delete_id}: {response.text}")


def list_all_models(filtered: str) -> list[str]:
    url = "https://api.baseten.co/v1/models"
    headers = {"Authorization": f"Api-Key {API_KEY}"}

    response = requests.request("GET", url, headers=headers, json={}).json()
    return [
        model["id"]
        for model in response.get("models", [])
        if matches_name(model, filtered=filtered)
    ]


def delete_all_truss_example_deployments():
    to_delete = list_all_models(FILTER)

    # Use a thread pool and map to delete concurrently.
    with ThreadPoolExecutor(16) as executor:
        list(executor.map(delete_model, to_delete))

    print("Deleted all truss example deployments")


def test_deploy(deploy_id: str = "03ykpnkw"):
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"],
        base_url=f"https://model-{deploy_id}.api.baseten.co/environments/production/sync/v1",
    )

    # Default completion
    response_completion = client.completions.create(
        model="not_required",
        prompt="Q: Tell me everything about Baseten.co! A:",
        temperature=0.3,
        max_tokens=100,
    )
    assert any(
        m in response_completion.choices[0].text.lower()
        for m in ["baseten", "sorry", "deepseek"]
    ), f"Completion response: {response_completion.choices[0].text}"

    # Chat completion
    response_chat = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": "Tell me everything about Baseten.co!"}],
        temperature=0.3,
        max_tokens=100,
    )
    assert any(
        m in response_chat.choices[0].message.content.lower()
        for m in ["baseten", "sorry", "deepseek"]
    ), f"Chat response: {response_chat.choices[0].message.content}"
    # Structured output
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    completion = client.beta.chat.completions.parse(
        model="not_required",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Give the name of the event, date, and participants.",
            },
        ],
        response_format=CalendarEvent,
    )

    event = completion.choices[0].message.parsed
    assert len(event.name), f"Event name: {event.name} was wrong"
    print(f"✅ All tests passed for deployment {deploy_id}")
    return deploy_id


def test_all_deployments_briton():
    to_test = list_all_models("Briton")
    # Repeat each test 16x
    to_test = to_test * 16

    errors = {}  # dictionary to collect exceptions per deployment id
    results = []  # to collect successful results

    # Use a thread pool and submit tasks to test concurrently.
    with ThreadPoolExecutor(128) as executor:
        future_to_deployment = {
            executor.submit(test_deploy, deploy_id): deploy_id for deploy_id in to_test
        }

        for future in as_completed(future_to_deployment):
            deploy_id = future_to_deployment[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                errors.setdefault(deploy_id, []).append(e)

    if errors:
        print("Some deployments failed:")
        for deploy_id, exc_list in errors.items():
            for exc in exc_list:
                print(f"Deployment {deploy_id} failed with error: {exc}")
        # Raise an aggregated exception after all tests are complete.
        raise Exception(
            f"{len(errors)} deployments failed during briton tests: {errors}"
        )

    print("All deployments passed tests")
    print("Results:", results)
    return results


@retry(max_retries=3, delay=5)
def deploy_config(path: Path) -> None:
    import truss
    import yaml

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    # Check using key "model_name" for deployment
    if matches_name(config, key="model_name", filtered=FILTER):
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
    with ThreadPoolExecutor(8) as executor:
        list(executor.map(deploy_config, paths))


if __name__ == "__main__":
    """Example usage:

    ```briton
    python deploy_all.py --action deploy --filter "Briton"
    python deploy_all.py --action delete --filter "Briton"
    python deploy_all.py --action britontest
    ```

    ```bash
    python deploy_all.py --action deploy --filter "BEI"
    python deploy_all.py --action delete --filter "BEI"
    ```
    # Deploy all
    ```bash
    python deploy_all.py --action deploy
    ```
    """
    parser = argparse.ArgumentParser(
        description="Deploy or delete Truss example deployments."
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["deploy", "delete", "britontest"],
        required=True,
        help="Action to perform: deploy or delete.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="?",
        default="-",
        help="Filter names with `x in name` logic.",
    )

    args = parser.parse_args()
    if args.filter:
        FILTER = args.filter

    if args.action == "deploy":
        deploy_all()
    elif args.action == "britontest":
        test_all_deployments_briton()
    elif args.action == "delete":
        delete_all_truss_example_deployments()
