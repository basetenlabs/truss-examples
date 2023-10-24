from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class Model:
    def __init__(self, **kwargs) -> None:
        self.model = None
        self.tokenizer = None

    def load(self):
        # Load model here and assign to self._model.
        base_model = "PygmalionAI/pygmalion-6b"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model = model
        self.tokenizer = tokenizer

    def predict(self, request) -> Any:

        char_name = request.pop("character_name")
        char_persona = request.pop("persona", "")
        scenario = request.pop("scenario", "")
        greeting = request.pop("greeting", "")
        dialogue_history = request.pop("dialogue_history", "")
        new_message = request.pop("new_message")
        example_dialogues = request.pop("example_dialogue", [])

        _output = evaluate(
            self.model,
            self.tokenizer,
            char_name,
            char_persona,
            scenario,
            greeting,
            dialogue_history,
            new_message,
            example_dialogues=example_dialogues,
            **request,
        )
        return _output


from transformers import StoppingCriteria


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt, tokenizer):
        self.target_sequence = target_sequence
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt, "")
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def evaluate(
    model,
    tokenizer,
    char_name,
    char_persona,
    scenario,
    greeting,
    dialogue_history,
    new_message,
    example_dialogues=[],
    temperature=1,
    top_p=0.9,
    top_k=40,
    num_beams=1,
    max_new_tokens=2048,
    **kwargs,
):
    prompts = generate_prompt(
        char_name,
        char_persona,
        scenario,
        greeting,
        dialogue_history,
        new_message,
        example_dialogues=example_dialogues,
    )
    inputs = tokenizer(
        prompts, return_tensors="pt", max_length=1024, truncation=True, padding=True
    )
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=MyStoppingCriteria("\nYou:", prompts, tokenizer),
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output


def generate_prompt(
    char_name,
    char_persona,
    scenario,
    greeting,
    dialogue_history,
    new_message,
    example_dialogues=[],
):

    examples = ""
    for dialogue in example_dialogues:
        examples += f"<START>\n{dialogue}\n"

    prompt = f"""{char_name}'s Persona: {char_persona}
Scenario: {scenario}\n
{examples}
<START>
{greeting}
{dialogue_history}
You: {new_message}
"""

    return prompt
