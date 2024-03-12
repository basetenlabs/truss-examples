import base64
from tempfile import NamedTemporaryFile
from async_batcher.batcher import AsyncBatcher
from huggingface_hub import snapshot_download
from run import WhisperTRTLLM
import gc
import re
import os
from whisper_utils import (
    log_mel_spectrogram,
)

import torch
from torch import Tensor


class MlBatcher(AsyncBatcher[list[Tensor], list[str]]):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: WhisperTRTLLM = model

    def process_batch(self, batch: list[Tensor]) -> list[float]:
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        num_beams=1
        # Need to pad the batch up to the maximum batch size
        features = torch.cat(batch, dim=0).type(torch.float16)
        return self.model.process_batch(features, text_prefix, num_beams)


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self._batcher = None
        gc.freeze()

    def load(self):
        # Load model here and assign to self._model.
        snapshot_download(
            "baseten/trtllm-whisper-a10g-1", local_dir=self._data_dir, max_workers=4
        )
        os.system(
            "wget --directory-prefix=/packages/assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken"
        )
        os.system(
            "wget --directory-prefix=/packages/assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
        )
        self._model = WhisperTRTLLM(f"{self._data_dir}")
        self._batcher = MlBatcher(
            model=self._model, max_batch_size=8, max_queue_time=0.5
        )

    def base64_to_wav(self, base64_string, output_file_path):
        binary_data = base64.b64decode(base64_string)
        with open(output_file_path, "wb") as wav_file:
            wav_file.write(binary_data)
        return output_file_path
    

    async def predict(self, model_input: dict):
        # TODO: figure out what the normalizer is for
        normalizer=None
        with NamedTemporaryFile() as fp:
            self.base64_to_wav(model_input["audio"], fp.name)
            mel, total_duration = log_mel_spectrogram(
                fp.name,
                self._model.n_mels,
                device="cuda",
                return_duration=True,
                mel_filters_dir="/packages/assets",
            )
            mel = mel.type(torch.float16)
            mel = mel.unsqueeze(0)
            prediction = await self._batcher.process(item=mel)
            
            # remove all special tokens in the prediction
            prediction = re.sub(r"<\|.*?\|>", "", prediction)
            if normalizer:
                prediction = normalizer(prediction)
            results = [(0, [""], prediction.split())]
            return results, total_duration

