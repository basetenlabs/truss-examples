import os
import re
import subprocess
import time
from datetime import datetime

import requests
import yaml

API_KEY = os.environ["BASETEN_API_KEY"]


def get_model_dir():
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main", "HEAD"], capture_output=True
    )
    changed_files = result.stdout.decode().split("\n")
    print(changed_files)
    for file in changed_files:
        if re.match(r".*/model/model\.py", file):
            return os.path.dirname(os.path.dirname(file))
    raise Exception("No model file found")


def get_example_input(image_str):
    ls = os.listdir(".")
    if "config.yaml" not in ls:
        raise Exception("No config.yaml found. You must implement a config.yaml file.")
    with open("config.yaml", "r") as f:
        try:
            loaded_config = yaml.safe_load(f.read())
        except yaml.YAMLError as e:
            raise Exception(f"Invalid config.yaml: {e}")

    if "model_metadata" not in loaded_config:
        raise Exception(
            "No model_metadata found in config.yaml. Config must include model_metadata with an example_model_input value."
        )

    if "example_model_input" not in loaded_config["model_metadata"]:
        raise Exception("No example_model_input found in model_metadata")

    if "model_name" not in loaded_config:
        loaded_config["model_name"] = "model"
        with open("config.yaml", "w") as f:
            f.write(yaml.safe_dump(loaded_config))

    example_input = loaded_config["model_metadata"]["example_model_input"]
    for key in ["image", "b64_image", "base64_image", "base_image", "input_image"]:
        if key in example_input:
            example_input[key] = image_str
    return example_input


def truss_push():
    print("Pushing model...")
    with open("/home/runner/.trussrc", "w") as config_file:
        config_file.write(
            f"""[baseten]
remote_provider = baseten
api_key = {API_KEY}
remote_url = https://app.staging.baseten.co"""
        )

    result = subprocess.run(["truss", "push", "--trusted"], capture_output=True)
    match = re.search(
        r"View logs for your deployment at \n?https://app\.staging\.baseten\.co/models/(\w+)/logs/(\w+)",
        result.stdout.decode(),
    )
    if not match:
        raise Exception(
            f"Failed to push model:\n\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
    model_id = str(match.group(1))  # Ensure model_id is a string
    deployment_id = str(match.group(2))  # Ensure deployment_id is a string
    print(
        f"Model pushed successfully. model-id: {model_id}. deployment-id: {deployment_id}"
    )
    return model_id, deployment_id


def truss_predict(model_id, input):
    result = {"error": "Model is not ready, it is still building or deploying"}
    seconds_remaining = 60 * 30  # Wait for 30 minutes
    while (
        "error" in result
        and result["error"] == "Model is not ready, it is still building or deploying"
        and seconds_remaining > 0
    ):
        print(f"{round(seconds_remaining / 60, 2)} minutes remaining")
        result = requests.post(
            f"https://model-{model_id}.api.baseten.co/development/predict",
            headers={"Authorization": f"Api-Key {API_KEY}"},
            json=input,
        )

        try:
            result = result.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            print(result.text)
            return

        if not isinstance(result, dict):
            return result

        seconds_remaining -= 30
        print("Waiting for model to be ready...")
        time.sleep(30)

    return result


def get_truss_logs(deployment_id, start_time):
    result = requests.post(
        "https://app.staging.baseten.co/logs",
        headers={"Authorization": f"Api-Key {API_KEY}"},
        json={
            "type": "MODEL",
            "start": start_time,
            "end": get_time_in_ms(),
            "levels": [],
            "regex": "",
            "limit": 500,
            "entity_id": deployment_id,
            "direction": "backward",
        },
    )
    # {'success': False, 'message': 'Failed to load logs'}
    return result.json()


def deactivate_truss(model_id):
    print("Deactivating model...")
    result = requests.post(
        f"https://api.baseten.co/v1/models/{model_id}/deployments/production/deactivate",
        headers={"Authorization": f"Api-Key {API_KEY}"},
    )
    print("Model deactivated successfully")
    print(result)


def print_formatted_logs(logs):
    for log in reversed(logs):
        ts = datetime.fromtimestamp(int(log["ts"]) // 1_000_000_000).ctime()
        print(f"{ts} - {log['level']} - {log['msg']}")


def get_time_in_ms():
    return time.time_ns() // 1_000_000


if __name__ == "__main__":
    model_dir = get_model_dir()

    image_str = open("bin/image.txt", "r").read()

    os.chdir(model_dir)
    example_input = get_example_input(image_str)
    model_id, deployment_id = truss_push()
    start_time = get_time_in_ms()
    result = truss_predict(model_id, example_input)
    print(f"Model prediction result: {result}")
    if "error" in result:
        logs = get_truss_logs(deployment_id, start_time)
        if logs["success"]:
            print_formatted_logs(logs["logs"])
            print(
                f"Failed to make prediction. Received error from model {result['error']}"
            )
            exit(1)
        else:
            print("Failed to load logs")

    deactivate_truss(model_id)
