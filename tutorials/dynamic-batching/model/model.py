import base64
import gc
import re
from tempfile import NamedTemporaryFile

import torch
from async_batcher.batcher import AsyncBatcher
from run import WhisperTRTLLM
from torch import Tensor
from whisper_utils import log_mel_spectrogram

TEXT_PREFIX = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# Num beams is the number of paths the model traverses before transcribing the text
NUM_BEAMS = 3

# Max queue time is the amount of time in seconds to wait to fill the batch
MAX_QUEUE_TIME = 0.25

# Maximum size of the batch. This is dictated by the compiled engine.
MAX_BATCH_SIZE = 8


class MlBatcher(AsyncBatcher[list[Tensor], list[str]]):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: WhisperTRTLLM = model

    def process_batch(self, batch: list[Tensor]) -> list[float]:
        # Need to pad the batch up to the maximum batch size
        features = torch.cat(batch, dim=0).type(torch.float16)
        return self.model.process_batch(features, TEXT_PREFIX, NUM_BEAMS)


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self._batcher = None
        gc.freeze()

    def load(self):
        self._model = WhisperTRTLLM("/app/model_cache/trtllm-whisper-a10g-large-v2-1")
        self._batcher = MlBatcher(
            model=self._model,
            max_batch_size=MAX_BATCH_SIZE,
            max_queue_time=MAX_QUEUE_TIME,
        )

    def base64_to_wav(self, base64_string, output_file_path):
        binary_data = base64.b64decode(base64_string)
        with open(output_file_path, "wb") as wav_file:
            wav_file.write(binary_data)
        return output_file_path

    async def predict(self, model_input: dict):
        # TODO: figure out what the normalizer is for
        normalizer = None
        with NamedTemporaryFile() as fp:
            self.base64_to_wav(model_input["audio"], fp.name)
            mel, total_duration = log_mel_spectrogram(
                fp.name,
                self._model.n_mels,
                device="cuda",
                return_duration=True,
                mel_filters_dir=f"{self._data_dir}/assets",
            )
            mel = mel.type(torch.float16)
            mel = mel.unsqueeze(0)
            prediction = await self._batcher.process(item=mel)

            # remove all special tokens in the prediction
            prediction = re.sub(r"<\|.*?\|>", "", prediction)
            if normalizer:
                prediction = normalizer(prediction)
            return {"text": prediction.strip(), "duration": total_duration}
