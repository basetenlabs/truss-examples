import base64
import tempfile
import time

from whisper_streaming.whisper_online import *

SAMPLING_RATE = 16000


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self.model = None

    def load(self):
        whisper_model = self._config["model_metadata"]["whisper_model"]
        valid_whisper_models = {
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
        }
        if whisper_model not in valid_whisper_models:
            raise ValueError(
                f"Whisper model must be one of the following: {valid_whisper_models}"
            )

        language = "en"
        asr = FasterWhisperASR(language, whisper_model)
        self.model = OnlineASRProcessor(asr)

    def base64_to_wav(self, base64_string, output_file_path):
        binary_data = base64.b64decode(base64_string)
        with open(output_file_path, "wb") as wav_file:
            wav_file.write(binary_data)

    def stream_transcriptions(self, audio_file_path, min_chunk, duration):
        beginning = 0.0
        end = 0.0
        start = time.time() - beginning

        while True:
            now = time.time() - start

            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)

            end = time.time() - start
            a = load_audio_chunk(audio_file_path, beginning, end)
            beginning = end

            self.model.insert_audio_chunk(a)

            try:
                (
                    start_timestamp,
                    end_timestamp,
                    transcription,
                ) = self.model.process_iter()
                yield transcription
            except AssertionError:
                print("assertion error")

            now = time.time() - start

            if end >= duration:
                break

        o = self.model.finish()
        self.model.init()

    def predict(self, request):
        audio = request.pop("audio")
        min_chunk = request.pop("chunk_size", 1.0)

        audio_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_file_path = audio_tempfile.name
        self.base64_to_wav(audio, audio_file_path)

        duration = len(load_audio(audio_file_path)) / SAMPLING_RATE
        streamer = self.stream_transcriptions(audio_file_path, min_chunk, duration)

        # Yield generated text as it becomes available
        def inner():
            for text in streamer:
                yield text

        return inner()
