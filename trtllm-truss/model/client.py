import subprocess
import time
from functools import partial
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from queue import Queue
from utils import prepare_model_repository
from tritonclient.utils import InferenceServerException

class UserData:
    def __init__(self):
        self._completed_requests = Queue()

def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

class TritonClient:
    def __init__(self, data_dir, model_repository_dir, tensort_parallel_count=1):
        self._data_dir = data_dir
        self._model_repository_dir = model_repository_dir
        self._tensor_parallel_count = tensort_parallel_count
        self._http_client = None
        self._grpc_client_map = {}

    def start_grpc_stream(self, user_data, model_name, inputs, stream_uuid):
        """Starts a GRPC stream and sends a request to the Triton server."""
        grpc_client_instance = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
        self._grpc_client_map[stream_uuid] = grpc_client_instance
        grpc_client_instance.start_stream(callback=partial(callback, user_data))
        grpc_client_instance.async_stream_infer(
            model_name,
            inputs,
            request_id=stream_uuid,
            enable_empty_final_response=True,
        )
    
    def start_server(
        self,
        mpi: int = 1,
    ):
        """Triton Inference Server has different startup commands depending on
        whether it is running in a TP=1 or TP>1 configuration. This function
        starts the server with the appropriate command."""
        if mpi == 1:
            return subprocess.Popen([
                "tritonserver",
                "--model-repository", self._model_repository_dir,
                "--grpc-port", "8001",
                "--http-port", "8003"
            ])
        command = [
            "mpirun",
            "--allow-run-as-root",
        ]
        for i in range(mpi):
            command += [
                "-n",
                "1",
                "tritonserver",
                "--model-repository", self._model_repository_dir,
                "--grpc-port", "8001",
                "--http-port", "8003",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{i}_",
                ":"
            ]
        return subprocess.Popen(command)    

    def load_server_and_model(self):
        """Loads the Triton server and the model."""
        prepare_model_repository(self._data_dir)
        self.start_server(mpi=self._tensor_parallel_count)
        
        self._http_client = httpclient.InferenceServerClient(url="localhost:8003", verbose=False)
        is_server_up = False
        while not is_server_up:
            try:
                is_server_up = self._http_client.is_server_live()
            except ConnectionRefusedError:
                time.sleep(2)
                continue

        while self._http_client.is_model_ready(model_name="ensemble") == False:
            time.sleep(2)
            continue

    @staticmethod
    def stream_predict(user_data: UserData):
        """Static method to yield predictions or errors based on input and a streaming user_data queue."""
        
        def _is_final_response(result):
            """Check if the given result is a final response according to Triton's specification."""
            if result:
                final_response_param = result.get_response().parameters.get("triton_final_response")
                return final_response_param.bool_param if final_response_param else False
            return False

        result = None
        
        while not _is_final_response(result):
            try:
                result = user_data._completed_requests.get()
                if not isinstance(result, InferenceServerException):
                    res = result.as_numpy('text_output')
                    yield res[0].decode("utf-8")
                else:
                    yield {
                        "status": "error",
                        "message": result.message()
                    }
            except Exception as e:
                yield {
                    "status": "error",
                    "message": str(e)
                }
                break
