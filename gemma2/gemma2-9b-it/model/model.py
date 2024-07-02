from threading import Thread
from typing import Dict
import subprocess

import torch
from local_gemma import LocalGemma2ForCausalLM
from transformers import (
    AutoTokenizer,
    TextIteratorStreamer,
)

CHECKPOINT = "google/gemma-2-9b-it"
class Model:
    def __init__(self, **kwargs) -> None:
        self.tokenizer = None
        self.model = None
        if "secrets" in kwargs:
            self._secrets = kwargs["secrets"]
        else:
            raise ValueError("Missing secrets")
    def load(self):
        # make sure token exists
        if not self._secrets["hf_access_token"]:
            raise ValueError("Missing hf_access_token")
        # huggingface auth is done in the local-gemma script, so we just need to call it
        command = ["local-gemma", "--token", self._secrets["hf_access_token"], "What is the capital of France?"]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with code {e.returncode}: {e.stderr}")
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, token=self._secrets["hf_access_token"])
        self.model = LocalGemma2ForCausalLM.from_pretrained(CHECKPOINT, preset="auto", token=self._secrets["hf_access_token"])

    def predict(self, request: Dict) -> Dict:
        prompt = request.pop("prompt")
        # Instantiate the Streamer object, which we'll later use for
        # returning the output to users.
        streamer = TextIteratorStreamer(self.tokenizer)
        messages = [
        {"role": "user", "content": prompt}]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.model.device)
        # When creating the generation parameters, ensure to pass the `streamer` object
        # that we created previously.
        with torch.no_grad():
            generation_kwargs = {
                "input_ids": model_inputs,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": 1000,
                "streamer": streamer,
            }
            # Spawn a thread to run the generation, so that it does not block the main
            # thread.
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            def inner():
                for text in streamer:
                    yield text
                thread.join()

            return inner()
