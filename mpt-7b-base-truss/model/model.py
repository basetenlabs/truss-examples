from typing import Dict, List

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Model:
    def __init__(self, **kwargs) -> None:

        # Model name (uncomment/ comment for the different MPT flavors)
        self.model_name = 'mosaicml/mpt-7b'
        #self.model_name = 'mosaicml/mpt-7b-instruct'
        #self.model_name = 'mosaicml/mpt-7b-storywriter'
        #self.model_name = 'mosaicml/mpt-7b-chat'

        # Device
        self.device='cuda:0'
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # Attention implementation
        # config = transformers.AutoConfig.from_pretrained(
        #   model_name,
        #   trust_remote_code=True
        # )
        # config.attn_config['attn_impl'] = 'triton'

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            #config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model.to(device=self.device)
        self.model.eval()

    def preprocess(self, request: dict):
        generate_args = {
            'max_new_tokens': 100,
            'temperature': 1.0,
            'top_p': 1.0,
            'top_k': 50,
            'repetition_penalty': 1.0,
            'no_repeat_ngram_size': 0,
            'use_cache': True,
            'do_sample': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        if 'max_tokens' in request.keys():
            generate_args['max_new_tokens'] = request['max_tokens']
        if 'temperature' in request.keys():
            generate_args['temperature'] = request['temperature']
        if 'top_p' in request.keys():
            generate_args['top_p'] = request['top_p']
        if 'top_k' in request.keys():
            generate_args['top_k'] = request['top_k']
        request['generate_args'] = generate_args
        return request
    
    def generate(self, prompt, generate_args):
        encoded_inp = self.tokenizer(prompt, return_tensors='pt', padding=True)
        for key, value in encoded_inp.items():
            encoded_inp[key] = value.to(self.device)
        with torch.no_grad():
            encoded_gen = self.model.generate(
                input_ids=encoded_inp['input_ids'],
                attention_mask=encoded_inp['attention_mask'],
                **generate_args,
            )
        decoded_gen = self.tokenizer.batch_decode(encoded_gen,
                                  skip_special_tokens=True)
        continuation = decoded_gen[0][len(prompt):]
        return continuation

    def predict(self, request: Dict) -> Dict[str, List]:
        try:
            prompt = request.pop("prompt")
            completion = self.generate(prompt, request['generate_args'])
        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": completion, "message": None}
