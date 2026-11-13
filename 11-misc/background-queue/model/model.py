from queue import Queue
import uuid
import threading
import requests

class BackgroundQueueProcessor:
    def __init__(self, loaded_model):
        self.queue = Queue()
        self.loaded_model = loaded_model
        self.results = {}

    def add_to_queue(self, data):
        item_id = uuid.uuid4()
        item = {"id": str(item_id), "data": data}
        self.queue.put(item)
        return str(item_id)

    def process_queue(self):
        while True:
            item = self.queue.get()
            # process item here
            item_id = item["id"]
            result = self.loaded_model(item["data"])
            self.queue.task_done()
            self.results[item_id] = result
    
    def get_is_done(self, item_id) -> bool:
        if item_id in self.results:
            return True
        return False
    
    def get_result(self, item_id):
        return self.results.pop(item_id)
            
import time

class Sleeper:
    def __call__(self, data):
        for i in range(6):
            print(f"task: {data}, processing {i+1}/6")
            time.sleep(60)  # sleep for 1 minute
        print("Done with data.")
        return f"result for {data}"


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None
        self._model = Sleeper()
        self._queue_processor = BackgroundQueueProcessor(self._model)
        
        self._thread = threading.Thread(target=self._queue_processor.process_queue)
        self._thread.start()


    def load(self):
        # Load model here and assign to self._model.
        pass

    def predict(self, model_input: dict) -> dict:
        request_type = model_input.get("type", "queue")
        if request_type == "queue":
            return {
                "task_id": self._queue_processor.add_to_queue(model_input["data"])
            }
            
        if request_type == "status":
            return {
                "done": self._queue_processor.get_is_done(model_input["task_id"])
            }
        if request_type == "result":
            return self._queue_processor.get_result(model_input["task_id"])
        
        