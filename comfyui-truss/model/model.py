import os
import subprocess
import sys
import websocket
import uuid
import json
import random
from io import BytesIO
import base64
import urllib.request
import urllib.parse
from typing import Dict
from multiprocessing import Process
import requests
import tempfile

side_process = None
original_working_directory = os.getcwd()


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self.ws = None
        self.server_address = "127.0.0.1:8188"
        self.client_id = str(uuid.uuid4())


        def download_model(self, model_url, destination_path):
            print(f"Downloading model {model_url} ...")
            try:
                response = requests.get(model_url, stream=True)
                response.raise_for_status()

                # Open the destination file and write the content in chunks
                with open(destination_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)

                print(f"Downloaded file to: {destination_path}")
            except requests.exceptions.RequestException as e:
                print(f"Download failed: {e}")

    def download_tempfile(self, file_url, filename):
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            filetype = filename.split(".")[-1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{filetype}")
            temp_file.write(response.content)
            return temp_file.name, temp_file
        except Exception as e:
            print("Error downloading and saving image:", e)
            return None

    def fill_template(self, workflow, template_values):
        if isinstance(workflow, dict):
            # If it's a dictionary, recursively process its keys and values
            for key, value in workflow.items():
                workflow[key] = self.fill_template(value, template_values)
            return workflow
        elif isinstance(workflow, list):
            # If it's a list, recursively process its elements
            return [self.fill_template(item, template_values) for item in workflow]
        elif isinstance(workflow, str) and workflow.startswith("{{") and workflow.endswith("}}"):
            # If it's a placeholder, replace it with the corresponding value
            placeholder = workflow[2:-2]
            if placeholder in template_values:
                return template_values[placeholder]
            else:
                return workflow  # Placeholder not found in values
        else:
            # If it's neither a dictionary, list, nor a placeholder, leave it unchanged
            return workflow

    def setup_comfyui(self):
        git_repo_url = "https://github.com/comfyanonymous/ComfyUI.git"
        git_clone_command = ["git", "clone", git_repo_url]

        try:
            # clone the repo
            subprocess.run(git_clone_command, check=True)
            print("Git repository cloned successfully!")

            # copy checkpoints from data directory
            os.chdir(os.path.join(original_working_directory, "ComfyUI"))

            model_json = os.path.join(original_working_directory, self._data_dir, "model.json")
            with open(model_json, "r") as file:
                data = json.load(file)

            print(f"model json file: {data}")

            if data and len(data) > 0:
                for model in data:
                    self.download_model(
                        model_url=model.get("url"),
                        destination_path=os.path.join(os.getcwd(), "models", model.get("path"))
                    )

            # run the comfy-ui server
            subprocess.run([sys.executable, "main.py"], check=True)

        except Exception as e:
            print(e)
            raise Exception("Error setting up comfy UI repo")

    def load(self):
        global side_process
        if side_process is None:
            side_process = Process(target=self.setup_comfyui)
            side_process.start()

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())

    def get_images(self, ws, prompt):
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history['outputs']:
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                output_images[node_id] = images_output

        return output_images

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def predict(self, request: Dict) -> Dict:
        if not self.ws:
            self.ws = websocket.WebSocket()
            self.ws.connect("ws://{}/ws?clientId={}".format(self.server_address, self.client_id))

        json_workflow = request.pop("json_workflow")
        template_values = request.pop("values")

        tempfiles = []
        for key, value in template_values.items():
            if value.startswith("https://") or value.startswith("http://"):
                if value[-1] == "/":
                    value = value[:-1]
                filename = value.split("/")[-1]
                file_destination_path, file_object = self.download_tempfile(file_url=value, filename=filename)
                tempfiles.append(file_object)
                template_values[key] = file_destination_path

        json_workflow = self.fill_template(json_workflow, template_values)

        try:
            images = self.get_images(self.ws, json_workflow)
        except Exception as e:
            print("SOMETHING BROKE!")
            print(e)

        for file in tempfiles:
            file.close()

        result = []
        for node_id in images:
            for image_data in images[node_id]:
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                b64_img = self.pil_to_b64(image)
                result.append(b64_img)

        return {"images": result}
