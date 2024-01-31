import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

import requests

COMFYUI_DIR = "/app/ComfyUI"


def download_model(model_url, destination_path):
    logging.info(f"Downloading model {model_url} ...")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        logging.debug("download response: ", response)

        # Open the destination file and write the content in chunks
        logging.debug("opening: ", destination_path)
        with open(destination_path, "wb") as file:
            logging.debug("writing chunks...")
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)

            logging.debug("done writing chunks!")

        logging.info(f"Downloaded file to: {destination_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed: {e}")


def download_tempfile(file_url, filename):
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        filetype = filename.split(".")[-1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{filetype}")
        temp_file.write(response.content)
        return temp_file.name, temp_file
    except Exception as e:
        logging.error("Error downloading and saving image:", e)
        return None


def setup_comfyui(original_working_directory, data_dir):

    try:
        model_json = os.path.join(original_working_directory, data_dir, "model.json")
        with open(model_json, "r") as file:
            data = json.load(file)

        logging.debug(f"model json file: {data}")

        if data and len(data) > 0:
            for model in data:
                download_model(
                    model_url=model.get("url"),
                    destination_path=os.path.join(
                        COMFYUI_DIR, "models", model.get("path")
                    ),
                )

        logging.debug("Finished downloading models!")

        # run the comfy-ui server
        subprocess.run([sys.executable, "main.py"], cwd=COMFYUI_DIR, check=True)

    except Exception as e:
        logging.error(e)
        raise Exception("Error setting up comfy UI repo")


def queue_prompt(prompt, client_id, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "http://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


def get_history(prompt_id, server_address):
    with urllib.request.urlopen(
        "http://{}/history/{}".format(server_address, prompt_id)
    ) as response:
        return json.loads(response.read())


def get_images(ws, prompt, client_id, server_address):
    prompt_id = queue_prompt(prompt, client_id, server_address)["prompt_id"]
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history(prompt_id, server_address)[prompt_id]
    for o in history["outputs"]:
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            if "images" in node_output:
                images_output = []
                for image in node_output["images"]:
                    image_data = get_image(
                        image["filename"],
                        image["subfolder"],
                        image["type"],
                        server_address,
                    )
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def fill_template(workflow, template_values):
    if isinstance(workflow, dict):
        # If it's a dictionary, recursively process its keys and values
        for key, value in workflow.items():
            workflow[key] = fill_template(value, template_values)
        return workflow
    elif isinstance(workflow, list):
        # If it's a list, recursively process its elements
        return [fill_template(item, template_values) for item in workflow]
    elif (
        isinstance(workflow, str)
        and workflow.startswith("{{")
        and workflow.endswith("}}")
    ):
        # If it's a placeholder, replace it with the corresponding value
        placeholder = workflow[2:-2]
        if placeholder in template_values:
            return template_values[placeholder]
        else:
            return workflow  # Placeholder not found in values
    else:
        # If it's neither a dictionary, list, nor a placeholder, leave it unchanged
        return workflow


def convert_request_file_url_to_path(template_values):
    tempfiles = []
    new_template_values = template_values.copy()
    for key, value in template_values.items():
        if isinstance(value, str) and (
            value.startswith("https://") or value.startswith("http://")
        ):
            if value[-1] == "/":
                value = value[:-1]
            filename = value.split("/")[-1]

            file_destination_path, file_object = download_tempfile(
                file_url=value, filename=filename
            )
            tempfiles.append(file_object)
            new_template_values[key] = file_destination_path

    return new_template_values, tempfiles
