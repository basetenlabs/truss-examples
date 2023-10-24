import whisperx
import requests
import os
import tempfile
from typing import Dict


class Model:
    device = "cuda"
    batch_size = 16
    compute_type = "float16"

    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._secrets = kwargs["secrets"]
        self.model = None
        self.diarize_model = None

    def load(self):
        # Need to manually download vad model
        vad_model_path = os.path.join(self._data_dir, "models", "pytorch_model.bin")
        self.model = whisperx.load_model(
            "medium",
            self.device,
            language="en",
            compute_type=self.compute_type,
            vad_options={"model_fp": vad_model_path},
        )
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self._secrets["hf_access_token"], device=self.device
        )

    def predict(self, request: Dict) -> Dict:
        file = request.get("audio_file", None)
        if not file:
            raise Exception("An audio file is required for this model")

        res = requests.get(file)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_audio_file:
            temp_audio_file.write(res.content)
            temp_audio_file.seek(0)
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
