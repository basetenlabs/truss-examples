import torch
from enc_dec.enc_dec_model import TRTLLMEncDecModel
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

HF_MODEL_NAME = "google-t5/t5-large"
DEFAULT_MAX_NEW_TOKENS = 20


class Model:
    def __init__(self, **kwargs):
        self._engine_dir = str(kwargs["data_dir"])
        model_metadata = kwargs["config"]["model_metadata"]
        self._engine_repo = model_metadata["engine_repository"]
        self._engine_name = model_metadata["engine_name"]
        self._beam_width = model_metadata["beam_width"]

    def load(self):
        snapshot_download(repo_id=self._engine_repo, local_dir=self._engine_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model_config = AutoConfig.from_pretrained(HF_MODEL_NAME)
        self._decoder_start_token_id = model_config.decoder_start_token_id
        self._tllm_model = TRTLLMEncDecModel.from_engine(
            self._engine_name, self._engine_dir
        )

    def predict(self, model_input):
        try:
            input_text = model_input.pop("prompt")
            max_new_tokens = model_input.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

            tokenized_inputs = self._tokenizer(
                input_text, return_tensors="pt", padding=True
            )
            input_ids = tokenized_inputs.input_ids.type(torch.IntTensor).to("cuda")
            decoder_input_ids = torch.IntTensor([[self._decoder_start_token_id]]).to(
                "cuda"
            )
            decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

            tllm_output = self._tllm_model.generate(
                encoder_input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=self._beam_width,
                bos_token_id=self._tokenizer.bos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                return_dict=True,
                attention_mask=tokenized_inputs.attention_mask,
            )
            tllm_output_ids = tllm_output["output_ids"]
            decoded_output = []
            for i in range(self._beam_width):
                output_ids = tllm_output_ids[:, i, :]
                output_text = self._tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                decoded_output.append(output_text)
            return {"status": "success", "data": decoded_output}
        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}
