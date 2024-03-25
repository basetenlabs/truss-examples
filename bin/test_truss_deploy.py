import yaml
import subprocess
import re
import os
import requests
import time

API_KEY = os.environ["BASETEN_API_KEY"]


def get_model_dir():
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main", "HEAD"], capture_output=True
    )
    changed_files = result.stdout.decode().split("\n")
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
        raise Exception("No model_metadata found in config.yaml")

    if "example_model_input" not in loaded_config["model_metadata"]:
        raise Exception("No example_model_input found in model_metadata")

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
remote_url = https://app.baseten.co"""
        )

    result = subprocess.run(["truss", "push", "--trusted"], capture_output=True)
    match = re.search(
        r"View logs for your deployment at \n?https://app\.baseten\.co/models/(\w+)/logs/\w+",
        result.stdout.decode(),
    )
    if not match:
        raise Exception(
            f"Failed to push model:\n\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
    print(f"Model pushed successfully. model-id: {match.group(1)}")
    return match.group(1)


def truss_predict(model_id, input):
    result = "Model is not ready, it is still building or deploying"
    seconds_remaining = 60 * 15  # Wait for 15 minutes
    while (
        result == "Model is not ready, it is still building or deploying"
        and seconds_remaining > 0
    ):
        print(f"{round(seconds_remaining / 60, 2)} minutes remaining")
        result = requests.post(
            f"https://model-{model_id}.api.baseten.co/development/predict",
            headers={"Authorization": f"Api-Key {API_KEY}"},
            json=input,
        )
        output = result.json()
        if "error" in output:
            result = output["error"]

        seconds_remaining -= 30
        print("Waiting for model to be ready...")
        time.sleep(30)

    return result


# def get_truss_logs(model_id, start_time):
#     baseten_api_key = os.environ["BASETEN_API_KEY"]
#     result = requests.post(
#         "https://app.baseten.co/logs",
#         headers={"Authorization": f"Api-Key {baseten_api_key}"},
#         json={
#             "type": "MODEL",
#             "start": 1711145005117,
#             "end": 1711145305117,
#             "levels": [],
#             "regex": "",
#             "limit": 500,
#             "entity_id": "e3mzx4zw",
#             "direction": "backward",
#         },
#     )
#     # {'success': False, 'message': 'Failed to load logs'}
#     return result


def deactivate_truss(model_id):
    print("Deactivating model...")
    result = requests.post(
        f"https://api.baseten.co/v1/models/{model_id}/deployments/production/deactivate",
        headers={"Authorization": f"Api-Key {API_KEY}"},
    )
    print("Model deactivated successfully")
    print(result)


# def get_time_in_ms():
#     time.time_ns() // 1_000_000


if __name__ == "__main__":
    model_dir = get_model_dir()

    image_str = open("bin/image.txt", "r").read()

    os.chdir(model_dir)
    example_input = get_example_input(image_str)
    model_id = truss_push()
    deactivate_truss(model_id)

    # TODO
    # result = truss_predict(model_id, example_input)
    # if "error" in result:
    #     logs = get_truss_logs(model_id, start_time)
    #     print(logs)
