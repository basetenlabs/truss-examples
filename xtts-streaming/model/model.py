import base64
import io
import os
import wave
import torch
import numpy as np
import logging

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

# This is one of the speaker voices that comes with xtts
SPEAKER_NAME = "Claribel Dervla"


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.speaker = None

    def load(self):
        device = "cuda"
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logging.info("⏳Downloading model")
        ModelManager().download_model(model_name)
        model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        self.model.to(device)

        self.speaker = {
            "speaker_embedding": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "speaker_embedding"].cpu().squeeze().half().tolist(),
            "gpt_cond_latent": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "gpt_cond_latent"].cpu().squeeze().half().tolist()
        }
        logging.info("🔥Model Loaded")

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def predict(self, model_input):
        text = model_input.get("text")
        language = model_input.get("language", "en")
        chunk_size = int(model_input.get("chunk_size", 150))  # Ensure chunk_size is an integer
        add_wav_header = False

        speaker_embedding = torch.tensor(self.speaker.get("speaker_embedding")).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(self.speaker.get("gpt_cond_latent")).reshape((-1, 1024)).unsqueeze(0)

        streamer = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=chunk_size,
            enable_text_splitting=True
        )

        for chunk in streamer:
            print(type(chunk))
            processed_chunk = self.wav_postprocess(chunk)
            processed_bytes = processed_chunk.tobytes()
            yield processed_bytes

        # def inner():
        #     for chunk in streamer:
        #         print(type(chunk))
        #         processed_chunk = self.wav_postprocess(chunk)
        #         processed_bytes = processed_chunk.tobytes()
        #         base64_bytes = base64.b64encode(processed_bytes)
        #         base64_string = base64_bytes.decode('utf-8')
        #         yield base64_string
        #
        # return inner()
