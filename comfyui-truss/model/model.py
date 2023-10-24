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
import shutil

side_process = None
original_working_directory = os.getcwd()


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self.ws = None
        self.server_address = "127.0.0.1:8188"
        self.client_id = str(uuid.uuid4())

    def setup_comfyui(self):
        git_repo_url = "https://github.com/comfyanonymous/ComfyUI.git"
        git_clone_command = ["git", "clone", git_repo_url]

        try:
            # clone the repo
            subprocess.run(git_clone_command, check=True)
            print("Git repository cloned successfully!")

            # copy checkpoints from data directory
            os.chdir(os.path.join(original_working_directory, "ComfyUI"))
            
            for file in os.listdir(os.path.join(original_working_directory, self._data_dir, "checkpoints")):
                filename = os.fsdecode(file)
                print("found file: ", filename)
                print("copy command cwd: ", os.getcwd())
                for item in os.listdir(os.getcwd()):
                    print(item)
                current_path = os.path.join(original_working_directory, self._data_dir, "checkpoints", filename)
                destination_path = os.path.join(os.getcwd(), "models", "checkpoints")
                print(f"copying model from: {current_path} to {destination_path}")
                subprocess.run(["cp", current_path, destination_path])

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

        positive_prompt = request.pop("positive_prompt")
        negative_prompt = request.pop("negative_prompt")
        json_workflow = request.pop("json_workflow")
        json_workflow["6"]["inputs"]["text"] = positive_prompt
        json_workflow["7"]["inputs"]["text"] = negative_prompt
        json_workflow["3"]["inputs"]["seed"] = random.randint(1, 10000)

        try:
            images = self.get_images(self.ws, json_workflow)
        except Exception as e:
            print("SOMETHING BROKE!")
            print(e)

        result = []
        for node_id in images:
            for image_data in images[node_id]:
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                b64_img = self.pil_to_b64(image)
                result.append(b64_img)

        return {"images": result}
