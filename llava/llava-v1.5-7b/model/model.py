import base64
from io import BytesIO

import torch
from llava.conversation import SeparatorStyle, conv_templates

# All of these files (in the package directory) are from the
# original LLaVA repository: https://github.com/haotian-liu/LLaVA/
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from PIL import Image

model_path = "liuhaotian/llava-v1.5-7b"
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
BASE64_PREAMBLE = "data:image/png;base64,"


def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None

    def load(self):
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
        )

    # inference code from: https://github.com/haotian-liu/LLaVA/blob/82fc5e0e5f4393a4c26851fa32c69ab37ea3b146/predict.py#L87
    def predict(self, model_input):
        query = model_input["query"]
        image = model_input["image"]

        if image[:5] == "https":
            image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
        else:
            image = b64_to_pil(image)

        top_p = model_input.get("top_p", 1.0)
        temperature = model_input.get("temperature", 0.2)
        max_tokens = model_input.get("max_tokens", 1000)

        # Run model inference here
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_tensor = (
            self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )

        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + "\n" + query
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        output = self.tokenizer.decode(
            output[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        print(output)

        return {"result": output}
