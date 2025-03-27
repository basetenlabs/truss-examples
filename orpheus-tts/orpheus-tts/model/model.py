from huggingface_hub import login
from orpheus_tts import OrpheusModel


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs["secrets"]
        self.hf_access_token = self._secrets["hf_access_token"]
        assert self.hf_access_token, "missing huggingface access token"
        login(token=self.hf_access_token)

        self.model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

    def predict(self, model_input):
        text = model_input.get("text", "Hello from Orpheus.")
        audio_generator = self.model.generate_speech(prompt=text, voice="tara")
        for chunk in audio_generator:
            yield chunk
