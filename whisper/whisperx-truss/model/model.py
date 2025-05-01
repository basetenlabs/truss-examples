import base64
import logging
import os
import tempfile
from io import BytesIO
from typing import Dict

import requests
import whisperx
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)


class Model:
    device = "cuda"
    batch_size = 16
    compute_type = "float16"

    asr_config = {
        "word_timestamps": True,
        "prepend_punctuations": '"\'"¿([{-',
        "append_punctuations": '"\'.。,，!！?？:：")]}、',
    }

    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._secrets = kwargs["secrets"]
        self.model = None
        self.diarize_model = None

    def load(self):
        self.model = whisperx.load_model(
            "large-v3",
            device=self.device,
            compute_type=self.compute_type,
            asr_options=self.asr_config,
        )
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self._secrets["hf_access_token"], device=self.device
        )

    def predict(self, request: Dict) -> Dict:
        file = request.get("audio_file", None)
        audio_base64 = request.get("audio_b64", None)

        if not file and not audio_base64:
            raise Exception(
                "An audio file or base64 audio string is required for this model"
            )

        headers = request.get("headers", {})

        if file:
            res = requests.get(file, headers=headers)
            logging.info(f"Response status code: {res.status_code}")
            logging.info(f"Response content length: {len(res.content)} bytes")

        if audio_base64:
            audio_data = base64.b64decode(audio_base64)
            audio_segment = AudioSegment.from_file(BytesIO(audio_data))
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_audio_file:
            if file:
                temp_audio_file.write(res.content)
                temp_audio_file.seek(0)

            if audio_base64:
                audio_segment.export(temp_audio_file.name, format="mp3")
                temp_audio_file.flush()

            audio = whisperx.load_audio(temp_audio_file.name)

            result = self.model.transcribe(audio, batch_size=self.batch_size)

            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            diarize_segments = self.diarize_model(audio)
            diarization_output = whisperx.assign_word_speakers(diarize_segments, result)

            result_segments = diarization_output.get("segments")
            word_seg = diarization_output.get("word_segments")

            results_segments_with_speakers = []
            for result_segment in result_segments:
                results_segments_with_speakers.append(
                    {
                        "start": result_segment["start"],
                        "end": result_segment["end"],
                        "text": result_segment["text"],
                        "speaker": result_segment["speaker"],
                    }
                )

            return {"model_output": results_segments_with_speakers}
