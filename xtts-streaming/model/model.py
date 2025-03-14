import base64
import io
import logging
import os
import time
import wave
import json

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import fastapi

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
        model_path = os.path.join(
            get_user_data_dir("tts"), model_name.replace("/", "--")
        )

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, checkpoint_dir=model_path, eval=True, use_deepspeed=True
        )
        self.model.to(device)

        self.speaker = {
            "speaker_embedding": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "speaker_embedding"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
            "gpt_cond_latent": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "gpt_cond_latent"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
        }

        self.speaker_embedding = (
            torch.tensor(self.speaker.get("speaker_embedding"))
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.gpt_cond_latent = (
            torch.tensor(self.speaker.get("gpt_cond_latent"))
            .reshape((-1, 1024))
            .unsqueeze(0)
        )
        logging.info("🔥Model Loaded")

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav


    async def websocket(self, websocket: fastapi.WebSocket):
        """Handle WebSocket connections for text-to-speech requests"""
        print("WebSocket connected")
        try:
            while True:
                data = await websocket.receive_text()
                
                try:
                    # Parse JSON input if provided
                    input_data = json.loads(data)
                except json.JSONDecodeError:
                    # If not JSON, assume it's just text
                    input_data = {"text": data, "language": "en", "chunk_size": 20}
                
                text = input_data.get("text")
                language = input_data.get("language", "en")
                chunk_size = int(input_data.get("chunk_size", 20))
                
                # Process the text to speech using the logic from the original predict method
                streamer = self.model.inference_stream(
                    text,
                    language,
                    self.gpt_cond_latent,
                    self.speaker_embedding,
                    stream_chunk_size=chunk_size,
                    enable_text_splitting=True,
                    temperature=0.2,
                )

                for chunk in streamer:
                    processed_chunk = self.wav_postprocess(chunk)
                    processed_bytes = processed_chunk.tobytes()
                    encoded_chunk = base64.b64encode(processed_bytes).decode('utf-8')
                    await websocket.send_json({
                        "type": "chunk",
                        "data": encoded_chunk
                    })
                
                await websocket.send_json({
                    "type": "complete",
                    "message": f"Processed '{text}'"
                })
                
        except fastapi.WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass