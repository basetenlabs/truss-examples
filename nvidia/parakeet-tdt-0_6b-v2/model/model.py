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
        self.model.change_attention_model("rel_pos_local_attn", [256, 256])
        self.model.change_subsampling_conv_chunking_factor(1)
        self.model.to(torch.bfloat16)

    def predict(self, request: dict):
        print(f"{request=}")
        audio_url = request.get("audio_url")
        print(f"{audio_url=}")

        is_timestamps = request.get("timestamps", False)

        # download audio temp file
        try:
            with requests.get(audio_url, stream=True) as response:
                response.raise_for_status()  # ensure HTTP errors are caught early

                url_without_query = audio_url.split("?")[0]

                if not url_without_query.endswith(
                    ".wav"
                ) and not url_without_query.endswith(".flac"):
                    return "Error: Only .wav and .flac files are supported"

                url_filetype = url_without_query.split(".")[-1]

                with tempfile.NamedTemporaryFile(
                    delete=True, suffix=f".{url_filetype}"
                ) as temp_file:
                    print("Downloading audio file...")
                    downloaded_bytes = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            downloaded_bytes += len(chunk)
                            temp_file.write(chunk)
                    temp_file.flush()
                    print(f"Download successful: {downloaded_bytes} bytes")

                    print("Transcribing audio...")
                    # transcribe audio
                    transcripts = self.model.transcribe(
                        [temp_file.name], timestamps=is_timestamps
                    )
                    print("Transcription successful")

                    # print(f"{transcripts=}")
                    if not is_timestamps:
                        return {"transcript": transcripts[0][0]}

                    # some latency penalty for json serialization
                    transcripts_json = json_serialize_recursive(
                        transcripts[0][0].__dict__
                    )

                    return {"transcript": transcripts_json}
        except Exception as e:
            print(e)
            return f"Error transcribing audio for {audio_url}"
