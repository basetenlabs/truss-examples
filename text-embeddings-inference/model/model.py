import subprocess
import grpc
from tei_pb import tei_pb2
from tei_pb import tei_pb2_grpc
from tei_pb.text_embeddings_router_config import Config
from typing import Any, Dict, List
import time



class Model:
    MAX_FAILED_SECONDS = 600 # 10 minutes; the reason this would take this long is mostly if we download a large model
        
    def __init__(self, data_dir, config, secrets):
        self._secrets = secrets
        self._config = config
        

    def load(self):
        config = Config(self._config['build']['arguments'])
        config.run_router()

        # Health check loop
        channel = grpc.insecure_channel('localhost:80')
        stub = tei_pb2_grpc.InfoStub(channel)
        healthy = False
        failed_seconds = 0
        print(f"Waiting for model to be ready for up to {self.MAX_FAILED_SECONDS} seconds")
        while not healthy and failed_seconds < self.MAX_FAILED_SECONDS:
            try:
                response = stub.Info(tei_pb2.InfoRequest())
                if response.model_id:  # Assuming a valid model_id indicates the service is serving
                    healthy = True
                    print("Model is ready")
                else:
                    failed_seconds += 1
                    time.sleep(1)  # wait for a second before retrying
            except grpc.RpcError:
                failed_seconds += 1
                time.sleep(1)  # wait and retry if server is not up yet

    async def predict(self, model_input):
        texts = model_input.pop("texts")
        stream = model_input.pop("stream", False)

        if isinstance(texts, str):
            texts = [texts]
        
        print(f"Starting to embed {len(texts)} texts")
        requests = [tei_pb2.EmbedRequest(inputs=text) for text in texts]
        async def generator():
            with grpc.insecure_channel('localhost:80') as channel:
                stub = tei_pb2_grpc.EmbedStub(channel)
                responses = stub.EmbedStream(iter(requests))
                for response in responses:
                    yield list(response.embeddings)

        if stream:
            return generator()
        else:
            embeddings = []
            async for embedding in generator():
                embeddings.append(embedding)
            
            return {"embeddings": embeddings}


if __name__ == "__main__":
    model = Model()
    model.load()
    print(model.predict("What is Deep Learning"))