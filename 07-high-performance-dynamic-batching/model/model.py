
from async_batcher.batcher import AsyncBatcher

import gc

class MlBatcher(AsyncBatcher[list[float], list[float]]):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model: keras.src.engine.sequential.Sequential = model

    def process_batch(self, batch: list[list[float]]) -> list[float]:
        # return self.model.predict(batch, verbose=0).tolist()
        print(batch)
        return batch


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None
        self._batcher = None
        gc.freeze()

    def load(self):
        # Load model here and assign to self._model.
        self._batcher = MlBatcher(model=None, max_queue_time=0.01)

        pass

    async def predict(self, model_input):
        # Run model inference here
        return (await self._batcher.process(item=model_input))
