import tempfile

import nemo.collections.asr as nemo_asr
import requests
import torch


def json_serialize_recursive(obj):
    if isinstance(obj, dict):
        return {k: json_serialize_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(json_serialize_recursive(v) for v in obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif hasattr(obj, "__dict__"):
        return json_serialize_recursive(obj.__dict__)
    else:
        return str(obj)


class Model:
    def __init__(self, **kwargs) -> None:
        # self.tokenizer = None
        self.model = None
        self._hf_access_token = kwargs["secrets"]["hf_access_token"]

    def load(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )

    def predict(self, request: dict):
        print(f"{request=}")
        audio_url = request.get("audio_url")
        print(f"{audio_url=}")

        is_timestamps = request.get("timestamps", False)

        # download audio temp file
        try:
            response = requests.get(audio_url)

            url_without_query = audio_url.split("?")[0]

            # if not .wav or .flac, reject for now
            if not url_without_query.endswith(
                ".wav"
            ) and not url_without_query.endswith(".flac"):
                return "Error: Only .wav and .flac files are supported"

            url_filetype = url_without_query.split(".")[-1]

            with tempfile.NamedTemporaryFile(
                delete=True, suffix=f".{url_filetype}"
            ) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

                # transcribe audio
                transcripts = self.model.transcribe(
                    [temp_file_path], timestamps=is_timestamps
                )
                print(f"{transcripts=}")

                if not is_timestamps:
                    return {"transcript": transcripts[0][0]}

                # some latency penalty for json serialization
                transcripts_copy_json = json_serialize_recursive(
                    transcripts[0][0].__dict__.copy()
                )

                return {"transcript": transcripts_copy_json}
        except Exception as e:
            print(e)
            return "Error transcribing audio"
