import base64
import copy
import io
import json
import os
import uuid
from io import BytesIO
from multiprocessing import Process
from typing import Dict

import websocket
from model.helpers import (
    convert_request_file_url_to_path,
    fill_template,
    get_images,
    setup_comfyui,
)
from PIL import Image

side_process = None
original_working_directory = os.getcwd()


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self.ws = None
        self.json_workflow = None
        self.server_address = "127.0.0.1:8188"
        self.client_id = str(uuid.uuid4())

    def load(self):
        global side_process
        if side_process is None:
            side_process = Process(
                target=setup_comfyui,
                kwargs=dict(
                    original_working_directory=original_working_directory,
                    data_dir=self._data_dir,
                ),
            )
            side_process.start()

        with open(
            os.path.join(self._data_dir, "comfy_ui_workflow.json"), "r"
        ) as json_file:
            self.json_workflow = json.load(json_file)

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def predict(self, request: Dict) -> Dict:
        if not self.ws:
            self.ws = websocket.WebSocket()
            self.ws.connect(
                "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
            )

        template_values = request.pop("workflow_values")

        template_values, tempfiles = convert_request_file_url_to_path(template_values)
        json_workflow = copy.deepcopy(self.json_workflow)
        json_workflow = fill_template(json_workflow, template_values)
        print(json_workflow)

        try:
            images = get_images(
                self.ws, json_workflow, self.client_id, self.server_address
            )
        except Exception as e:
            print("Error occurred while running Comfy workflow: ", e)

        for file in tempfiles:
            file.close()

        result = []

        for node_id in images:
            for image_data in images[node_id]:
                image = Image.open(io.BytesIO(image_data))
                b64_img = self.pil_to_b64(image)
                result.append({"node_id": node_id, "image": b64_img})

        return {"result": result}
