import asyncio
import multiprocessing
from concurrent import futures
from typing import Dict, List

import numpy as np
import torch


class _PreprocessHelper:
    # If preprocessing needs some setup, wrap it into a helper class. Otherwise, a simple function
    # also works.
    # If preprocessing needs objects that cannot be pickled or are inefficient to pickle, the overall
    # design must be changed to using processes that have a persistent process-local setup of these objects.

    def preprocess(self, request: dict):
        # Dummy CPU load.
        n = int(request["n"])
        m = np.random.rand(n, n).astype(np.float32) / n
        for i in range(30):
            m = np.dot(m, m)

        return {"inputs": m}


class Model:
    def __init__(self, **kwargs) -> None:
        self._preprocess_helper = _PreprocessHelper()
        self._process_pool = futures.ProcessPoolExecutor(multiprocessing.cpu_count())

    def load(self):
        pass

    async def preprocess(self, request: Dict) -> Dict:
        # Caveat: the function passed to `submit` must be pickleable. Therefore it cannot be
        # `self`, because `self`` contains `_process_pool` which is not pickleable.
        return await asyncio.wrap_future(
            self._process_pool.submit(self._preprocess_helper.preprocess, request)
        )

    async def predict(self, request: Dict) -> Dict[str, List]:
        # Dummy GPU load.
        inputs = request["inputs"]
        m = torch.from_numpy(inputs).to("cuda")

        for i in range(30):
            m = torch.matmul(m, m)

        result = m.sum().numpy(force=True)
        response = {"predictions": result}
        return response

    def postprocess(self, request: Dict) -> Dict:
        return request
