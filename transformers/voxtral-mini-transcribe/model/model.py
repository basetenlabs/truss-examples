import torch

from transformers import AutoProcessor, VoxtralForConditionalGeneration

repo_id = "mistralai/Voxtral-Mini-3B-2507"


class Model:
    def __init__(self, **kwargs):
        self._device = "cuda"
        self._model = None
        self._processor = None
        self._hf_access_token = kwargs["secrets"]["hf_access_token"]

    def load(self):

        self._model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
            token=self._hf_access_token,
        )
        self._processor = AutoProcessor.from_pretrained(
            repo_id, token=self._hf_access_token
        )

    def predict(self, model_input):
        print(model_input)
        inputs = self._processor.apply_transcrition_request(
            language="en",
            audio=model_input["audio"],
            model_id=repo_id,
        )
        inputs = inputs.to(self._device, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=500)
            decoded_outputs = self._processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            return decoded_outputs
