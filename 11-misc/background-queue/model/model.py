from queue import Queue
import uuid
import threading
import requests

class BackgroundQueueProcessor:
    def __init__(self, loaded_model, callback_url):
        self.queue = Queue()
        self.loaded_model = loaded_model
        self.callback_url = callback_url

    def add_to_queue(self, data):
        item_id = uuid.uuid4()
        item = {"id": str(item_id), "data": data}
        self.queue.put(item)
        return str(item_id)

    def process_queue(self):
        while True:
            print("Waiting for tasks")
            item = self.queue.get()
            print("got a task")
            # process item here
            item_id = item["id"]
            result = self.loaded_model(item["data"])
            self.queue.task_done()
            payload = {"result": result, "item_id": item_id}
            response = requests.post(self.callback_url, json=payload)
            print(f"POST: {response.status_code}, text: {response.text}")

import time

class Sleeper:
    def __call__(self, data):
        for i in range(6):
            print(f"task: {data}, processing {i+1}/6")
            time.sleep(60)  # sleep for 1 minute
        print("Done with data.")


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None
        self._model = Sleeper()
        self._queue_processor = BackgroundQueueProcessor(self._model, "https://1948-2600-8802-5502-f700-10dd-d39e-b6c-9e36.ngrok-free.app")
        
        self._thread = threading.Thread(target=self._queue_processor.process_queue)
        self._thread.start()


    def load(self):
        # Load model here and assign to self._model.
        pass

    def predict(self, model_input):
        return {
            "task_id": self._queue_processor.add_to_queue(model_input)
        }
        
        