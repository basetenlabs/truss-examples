from transformers import pipeline

MODEL_NAME = "Nexusflow/NexusRaven-V2-13B"


class Model:
    def __init__(self) -> None:
        self.pipeline = None

    def load(self):
        self.generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
        self.eos_token_id = self.generator.tokenizer.encode(
            "<bot_end>", add_special_tokens=False
        )[0]

    def predict(self, request: dict):
        prompt = request.pop("prompt")
        result = self.generator(
            prompt,
            max_new_tokens=request.get("max_new_tokens", 2048),
            temperature=0.001,
            do_sample=False,
            return_full_text=False,
            eos_token_id=self.eos_token_id,
        )
        function_call = result[0]["generated_text"].replace("Call:", "").strip()

        return function_call
