from typing import Dict, List
import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        self._tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self._model = LlamaForCausalLM.from_pretrained(
            str(self._data_dir),
            torch_dtype=torch.float16,
            device_map="auto",
        ) 
        self._model = PeftModel.from_pretrained(
            self._model, "tloen/alpaca-lora-7b",
            torch_dtype=torch.float16
        )
        self._model.eval()

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request
    
    
    def generate_prompt(self, instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """
    
    def forward(self, instruction, temperature=0.1, top_p=0.75, top_k=40, num_beams=2, **kwargs):
        prompt = self.generate_prompt(instruction)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1024,
            )
        
        s = generation_output.sequences[0]
        output = self._tokenizer.decode(s)
        return output.split("### Response:")[1].strip()



    def predict(self, request: Dict) -> Dict[str, List]:
        prompt = request.pop("prompt")
        completion = self.forward(prompt, **request)
        return {"completion" : completion}
