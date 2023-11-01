import numpy as np
from client import UserData, TritonClient
from threading import Thread
from utils import prepare_grpc_tensor

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._tensor_parallel_count = None
        self._request_id_counter = 0
        self.triton_client = None
    
    def load(self):
        self._tensor_parallel_count = self._config["model_metadata"].get("tensor_parallelism", 1)
        self.triton_client = TritonClient(self._data_dir, self._tensor_parallel_count)
        self.triton_client.load_server_and_model()

    def predict(self, model_input):
        model_name = "ensemble"
        user_data = UserData()
        self._request_id_counter += 1
        
        prompt = model_input.get("text_input")
        output_len = model_input.get("output_len", 256)
        beam_width = model_input.get("beam_width", 1)
        bad_words_list = model_input.get("bad_words_list", [""])
        stop_words_list = model_input.get("stop_words_list", [""])
        
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * output_len
        bad_words_list = np.array([bad_words_list], dtype=object)
        stop_words_list = np.array([stop_words_list], dtype=object)
        streaming = [[True]]
        streaming_data = np.array(streaming, dtype=bool)
        beam_width = [[beam_width]]
        beam_width_data = np.array(beam_width, dtype=np.uint32)

        inputs = [
            prepare_grpc_tensor("text_input", input0_data),
            prepare_grpc_tensor("max_tokens", output0_len),
            prepare_grpc_tensor("bad_words", bad_words_list),
            prepare_grpc_tensor("stop_words", stop_words_list),
            prepare_grpc_tensor("stream", streaming_data),
            prepare_grpc_tensor("beam_width", beam_width_data),
        ]

        # Start GRPC stream in a separate thread
        stream_uuid = str(self._request_id_counter)
        stream_thread = Thread(
            target=self.triton_client.start_grpc_stream,
            args=(user_data, model_name, inputs, stream_uuid)
        )
        stream_thread.start()

        # Yield results from the queue
        for i in TritonClient.stream_predict(user_data):
            yield i

        # Clean up GRPC stream and thread
        triton_grpc_stream = self._triton_grpc_client_map[stream_uuid]
        triton_grpc_stream.stop_stream()
        stream_thread.join()
        del self._triton_grpc_client_map[stream_uuid]