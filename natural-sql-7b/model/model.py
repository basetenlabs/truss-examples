import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "chatdb/natural-sql-7b", device_map="auto", torch_dtype=torch.float16
        )
        # Ensure the model is optimized for CUDA if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def predict(self, input_dict):
        if "question" not in input_dict or "schema" not in input_dict:
            raise ValueError(
                "Input dictionary must contain 'question' and 'schema' keys."
            )

        question = input_dict["question"]
        schema = input_dict["schema"]
        prompt = f"""
        ### Task
        Generate a SQL query to answer the following question: `{question}`

        ### PostgreSQL Database Schema
        {schema}

        ### Answer
        Here is the SQL query that answers the question: `{question}`
        """
        # Ensure the tokenizer generates tensors directly on the model's device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
        )
        result = (
            self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            .split("### Answer")[1]
            .strip()
        )
        return {"question": question, "sql_query": result}
