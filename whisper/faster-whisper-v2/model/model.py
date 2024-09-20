import os

from faster_whisper import WhisperModel, BatchedInferencePipeline
import httpx
import tempfile
import time
from typing import Dict
import base64
import logging

DEFAULT_BATCH_SIZE = 8
DEFAULT_WORD_LEVEL_TIMESTAMPS = False
DEFAULT_PROMPT = None
DEFAULT_TEMPERATURE = 0
DEFAULT_BEAM_SIZE = 5
DEFAULT_BEST_OF = 5
DEFAULT_LANGUAGE = None
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = False


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.batched_model = None

    def load(self):
        self.model = WhisperModel("large-v2", device="cuda", compute_type="float16")
        self.batched_model = BatchedInferencePipeline(model=self.model)

    def base64_to_wav(self, base64_string):
        binary_data = base64.b64decode(base64_string)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file_path:
            output_file_path.write(binary_data)
            output_file_path.flush()
        return output_file_path.name

    def download_file(self, url):
        with httpx.Client() as client:
            response = client.get(url, timeout=500)
            if response.status_code == 200:
                # Save the file to a local file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file_path:
                    output_file_path.write(response.content)
                    output_file_path.flush()
                logging.info("File downloaded successfully.")
                return output_file_path.name
            else:
                logging.info(f"Failed to download file. Status code: {response.status_code}")
                return None

    def preprocess(self, request: Dict) -> Dict:
        audio_base64 = request.get("audio")
        url = request.get("url")
        word_level_timestamps = request.get("word_timestamps", DEFAULT_WORD_LEVEL_TIMESTAMPS)
        prompt = request.get("prompt", DEFAULT_PROMPT)
        temperature = request.get("temperature", DEFAULT_TEMPERATURE)
        batch_size = request.get("batch_size", DEFAULT_BATCH_SIZE)
        beam_size = request.get("beam_size", DEFAULT_BEAM_SIZE)
        best_of = request.get("best_of", DEFAULT_BEST_OF)
        language = request.get("language", DEFAULT_LANGUAGE)

        response = {}

        if audio_base64 and url:
            return {
                "error": "Only a base64 audio file OR a URL can be passed to the API, not both of them.",
            }
        if not audio_base64 and not url:
            return {
                "error": "Please provide either an audio file in base64 string format or a URL to an audio file.",
            }

        if audio_base64:
            file_name = self.base64_to_wav(audio_base64)
            response['audio'] = file_name

        elif url:
            start = time.time()
            file_name = self.download_file(url)
            logging.info(f"url download time: {time.time() - start}",)
            response['audio'] = file_name

        response['word_timestamps'] = word_level_timestamps
        response['initial_prompt'] = prompt
        response['temperature'] = temperature
        response['batch_size'] = batch_size
        response['beam_size'] = beam_size
        response['best_of'] = best_of
        response['language'] = language

        return response

    def predict(self, model_input: Dict):
        start = time.time()

        all_segments = []
        full_transcript = ""
        audio = model_input.pop("audio")

        if audio:
            segments, info = self.batched_model.transcribe(audio, **model_input)
            for segment in segments:
                segment_information = {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                }

                words = []
                if segment.words:
                    for word in segment.words:
                        words.append(
                            {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word
                            }
                        )

                    segment_information['words'] = words

                all_segments.append(segment_information)
                full_transcript += segment.text

            language = info.language
            end = time.time()
            transcription_time = end - start

            return {
                        "segments": all_segments,
                        "language": language,
                        "transcript": full_transcript,
                        "transcription_time": transcription_time
                    }

        return model_input

