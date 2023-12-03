import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image
from io import BytesIO
import base64

BASE64_PREAMBLE = "data:image/png;base64,"
def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto'
        ).to('cuda').eval()

    def predict(self, model_input):
        query = model_input["query"]
        image = model_input["image"]
        image = b64_to_pil(image)

        # See code here for more details: https://huggingface.co/THUDM/cogvlm-chat-hf/blob/main/modeling_cogvlm.py#L738
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            result = self.tokenizer.decode(outputs[0])

        print(result)
        return {"result": result}
