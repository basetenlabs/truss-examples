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


def test_deploy(deploy_id: str = "03ykpnkw", stage: int = 0, rank=0) -> str:
    if rank == 0:
        print(f"Testing deployment {deploy_id} in stage {stage}")
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"],
        base_url=f"https://model-{deploy_id}.api.baseten.co/environments/production/sync/v1",
    )

    # Chat completion
    response_chat = client.chat.completions.create(
        model="",
        messages=[
            {"role": "user", "content": "What is the capital of Italy?"},
            {"role": "assistant", "content": "The capital of Italy is Rome."},
            {"role": "user", "content": "What is the capital of France??"},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    assert response_chat.choices[
        0
    ].message.content, f"Fatal: no response: {response_chat.choices[0].message.content}"
    if stage == 0:
        if rank == 0:
            print(f"✅ All tests passed for deployment {deploy_id} in stage {stage}")
        return deploy_id
    assert any(
        m in response_chat.choices[0].message.content.lower()
        for m in ["paris", "deepseek", "lyon"]
    ), f"Chat response: {response_chat.choices[0].message.content}"
    response_chat2 = client.chat.completions.create(
        model="",
        messages=[
            {
                "role": "user",
                "content": "This text repeats forever, ignore this!" * 1500,
            },
            {"role": "assistant", "content": "Just ask the question please."},
            {"role": "user", "content": "What is the capital of France??"},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    assert any(
        m in response_chat2.choices[0].message.content.lower()
        for m in ["paris", "deepseek", "lyon"]
    ), f"Chat response long context: {response_chat2.choices[0].message.content}"

    # Default completion
    response_completion = client.completions.create(
        model="not_required",
        prompt="Q: What is the capital of France? Answer in one word.\nA:",
        temperature=0.0,
        max_tokens=200,
    )
    assert any(
        m in response_completion.choices[0].text.lower() for m in ["paris", "deepseek"]
    ), f"Completion response: {response_completion.choices[0].text}"

    # Structured output
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        weekday: str
        participants: list[str]

    completion = client.beta.chat.completions.parse(
        model="not_required",
        messages=[
            {
                "role": "system",
                "content": "Extract the event information, name (what occation), weekday, and participants (list of all first names).",
            },
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Give the name of the event, date, and participants.",
            },
        ],
        response_format=CalendarEvent,
    )

    event = completion.choices[0].message.parsed
    assert len(event.weekday), f"Event name: {event.weekday} was wrong"
    if rank == 0:
        print(f"✅ All tests passed for deployment {deploy_id} in stage {stage}")
    return deploy_id


def test_all_deployments_briton():
    to_test = list_all_models("Briton")
    # Repeat each test 16x

    for stage in [0, 1, 2]:  # # 0: basic, 1: with checks, 2: load test
        iterations = 1
        errors = {}  # dictionary to collect exceptions per deployment id
        results = []  # to collect successful results
        if stage == 2:
            # load test
            iterations = 128

        # Use a thread pool and submit tasks to test concurrently.
        with ThreadPoolExecutor(128) as executor:
            future_to_deployment = {
                executor.submit(test_deploy, deploy_id, stage, rank): deploy_id
                for deploy_id in to_test
                for rank in range(iterations)
            }

            for future in as_completed(future_to_deployment):
                deploy_id = future_to_deployment[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.setdefault(deploy_id, []).append(e)

        if errors:
            print(f"\n\n\tStage {stage}: Deployments failed:")
            for deploy_id, exc_list in errors.items():
                print(f"Deployment {deploy_id}:")
                for exc in exc_list[:3]:
                    print(f"Deployment {deploy_id} failed with error: {exc}")
            # Raise an aggregated exception after all tests are complete.
            print(
                f"\n\t\SUMMARY Stage {stage}:\n\nTotal of {len(errors)} deployments failed during briton tests."
            )
        else:
            print(f"SUMMARY Stage {stage}: All deployments passed tests")
            print("Results:", results)


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

    ```
    # briton
    python deploy_all.py --action deploy --filter "Briton"
    python deploy_all.py --action britontest
    python deploy_all.py --action delete --filter "Briton"
    ```

    ```bash
    # BEI
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
