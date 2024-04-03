import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "chatdb/natural-sql-7b",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def predict(self, input_dict):
        # Ensure that the input_dict contains the requisite keys for 'questions'
        if "questions" not in input_dict:
            raise ValueError("Input dictionary must contain a 'questions' key.")

        questions = input_dict["questions"]

        outputs = []
        for question in questions:
            prompt = f"""
            ### Task

            Generate a SQL query to answer the following question: `{question}`

            ### Database Schema
            The query will run on a database with the following schema:
            ```schema
            # ... include the schema here ...
            ```

            ### Answer
            Here is the SQL query that answers the question: `{question}`
            """
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )

            # Check if CUDA is available and move tensors accordingly
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
                self.model.to("cuda")

            generated_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=5,
                early_stopping=True,
            )

            # Post-process the output to extract the SQL query
            output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            sql_query = output.split("```sql")[-1].strip()
            outputs.append(sql_query)

        return outputs
