import base64
import multiprocessing
import os
import subprocess
from typing import Dict

import httpx
import requests
from huggingface_hub import snapshot_download

TORCHSERVE_ENDPOINT = "http://0.0.0.0:8888/predictions/whisper_base"
TORCHSERVE_HEALTH_ENDPOINT = "http://0.0.0.0:8888/ping"


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self.torchserver_ready = False

    def start_tochserver(self):
        subprocess.run(
            [
                "torchserve",
                "--start",
                "--model-store",
                f"{self._data_dir}/model_store",
                "--models",
                "whisper_base.mar",
                "--foreground",
                "--no-config-snapshots",
                "--ts-config",
                f"{self._data_dir}/config.properties",
            ],
            check=True,
        )

    def load(self):
        snapshot_download(
            "htrivedi99/whisper-torchserve",
            local_dir=os.path.join(self._data_dir, "model_store"),
            max_workers=4,
        )
        logging.info("⚡️ Weights Downloaded Successfully!")

        process = multiprocessing.Process(target=self.start_tochserver)
        process.start()

        # Need to wait for the torchserve server to start up
        while not self.torchserver_ready:
            try:
                res = requests.get(TORCHSERVE_HEALTH_ENDPOINT)
                if res.status_code == 200:
                    self.torchserver_ready = True
                    logging.info("🔥Torchserve is ready!")
            except Exception as e:
                logging.info("⏳Torchserve is loading...")
                time.sleep(5)

    async def predict(self, request: Dict):
        audio_base64 = request.get("audio")
        audio_bytes = base64.b64decode(audio_base64)

        async with httpx.AsyncClient() as client:
            res = await client.post(
                TORCHSERVE_ENDPOINT, files={"data": (None, audio_bytes)}, timeout=120
            )
            transcription = res.text
        return {"output": transcription}
