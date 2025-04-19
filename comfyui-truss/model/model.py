import copy
import json
import os
import time
import uuid
from multiprocessing import Process
from typing import Dict

import websocket
from model.helpers import (
    convert_outputs_to_base64,
    convert_request_file_url_to_path,
    fill_template,
    get_images,
    setup_comfyui,
)

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
        # Start the ComfyUI server
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

        # Load the workflow file as a python dictionary
        with open(
            os.path.join(self._data_dir, "comfy_ui_workflow.json"), "r"
        ) as json_file:
            self.json_workflow = json.load(json_file)

        # Connect to the ComfyUI server via websockets
        socket_connected = False
        while not socket_connected:
            try:
                self.ws = websocket.WebSocket()
                self.ws.connect(
                    "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
                )
                socket_connected = True
            except Exception:
                print("Could not connect to comfyUI server. Trying again...")
                time.sleep(5)

        print("Truss has successfully connected to the ComfyUI server!")

    def predict(self, request: Dict) -> Dict:
        template_values = request.pop("workflow_values")

        template_values, tempfiles = convert_request_file_url_to_path(template_values)
        json_workflow = copy.deepcopy(self.json_workflow)
        json_workflow = fill_template(json_workflow, template_values)
        print(json_workflow)

        try:
            outputs = get_images(
                self.ws, json_workflow, self.client_id, self.server_address
            )

        except Exception as e:
            print("Error occurred while running Comfy workflow: ", e)

        for file in tempfiles:
            file.close()

        result = []

        for node_id in outputs:
            for item in outputs[node_id]:
                file_name = item.get("filename")
                file_data = item.get("data")
                output = convert_outputs_to_base64(
                    node_id=node_id, file_name=file_name, file_data=file_data
                )
                result.append(output)

        return {"result": result}
