import os
import random

import torch
from huggingface_hub import snapshot_download
from model.helper import TRTLLMEncDecModel
from transformers import AutoConfig, AutoTokenizer, set_seed


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self._data_dir = kwargs["data_dir"]
        self._secrets = kwargs["secrets"]
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.hf_token = self._secrets["hf_access_token"]

    def load(self):
        hugging_face_repo = self._config["model_metadata"]["repo_id"]
        snapshot_download(
            hugging_face_repo,
            local_dir=os.path.join(self._data_dir, "weights"),
            max_workers=4,
            use_auth_token=self.hf_token,
        )
        print("Downloaded weights succesfully!")

        self.model = TRTLLMEncDecModel.from_engine(
            "flan-t5-xl", os.path.join(self._data_dir, "weights")
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        self.model_config = AutoConfig.from_pretrained("google/flan-t5-xl")

    def preprocess(self, request: dict):
        if "bad_words" in request:
            bad_words = request.pop("bad_words")
            bad_word_ids = self.tokenizer(
                bad_words, add_prefix_space=True, add_special_tokens=False
            ).input_ids

            request["bad_words_ids"] = bad_word_ids
        if "seed" in request:
            set_seed(request.pop("seed"))
        else:
            set_seed(random.randint(0, 4294967294))
        return request

    def predict(self, request: dict):
        prompt = request.pop("prompt")
        tokenized_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        max_new_tokens = request.pop("max_new_tokens", 70)
        input_ids = tokenized_inputs.input_ids.type(torch.IntTensor).to("cuda")

        decoder_input_ids = torch.IntTensor(
            [[self.model_config.decoder_start_token_id]]
        ).to("cuda")
        decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

        tllm_output_ids = self.model.generate(
            encoder_input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_ids = tllm_output_ids[:, 0, :]
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return {"output": output_text}
