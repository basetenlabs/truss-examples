# In this example, we go through a Truss that serves an LLM. We
# use the model Mistral-7B, which is a general-purpose LLM that
# can used for a variety of tasks, like summarization, question-answering,
# translation, and others.
#
# # Set up the imports and key constants
#
# In this example, we use the Huggingface transformers library to build a text generation model.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# We use the 7B version of the Mistral model.
CHECKPOINT = "mistralai/Mistral-7B-v0.1"

# # Define the `Model` class and load function
#
# In the `load` function of the Truss, we implement logic involved in
# downloading and setting up the model. For this LLM, we use the `Auto`
# classes in `transformers` to instantiate our Mistral model.
class Model:
    def __init__(self, **kwargs) -> None:
        self.tokenizer = None
        self.model = None

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, torch_dtype=torch.float16, device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT,
        )

    # # Define the `predict` function
    #
    # In the predict function, we implement the actual inference logic. The steps
    # here are:
    #   * Set up the generation params. We have defaults for both of these, but
    # adjusting the values will have an impact on the model output
    #   * Tokenize the input
    #   * Generate the output
    #   * Use tokenizer to decode the output
    def predict(self, request: dict):
        prompt = request.pop("prompt")
        generate_args = {
            "max_new_tokens": request.get("max_new_tokens", 128),
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 0.95),
            "top_k": request.get("top_k", 50),
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            output = self.model.generate(inputs=input_ids, **generate_args)
            return self.tokenizer.decode(output[0])
