import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self):
        """Loads the chatdb/natural-sql-7b model along with its tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "chatdb/natural-sql-7b",
            torch_dtype=torch.float16,
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, input_dict):
        """Generates SQL queries from a list of natural language SQL questions.
        Args:
            input_dict (dict): A dictionary containing a list of SQL questions under the key 'questions'.
        Returns:
            list: A list of generated SQL queries that are JSON serializable.
        """
        questions = input_dict.get("questions", [])
        sql_queries = []

        for question in questions:
            prompt = f"""
            ### Task
            
            Generate a SQL query to answer the following question: `{question}`
            
            ### PostgreSQL Database Schema
            The query will run on a database with the following schema:
            ```
            CREATE TABLE users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE projects (
                project_id SERIAL PRIMARY KEY,
                project_name VARCHAR(100) NOT NULL,
                description TEXT,
                start_date DATE,
                end_date DATE,
                owner_id INTEGER REFERENCES users(user_id)
            );
            
            CREATE TABLE tasks (
                task_id SERIAL PRIMARY KEY,
                task_name VARCHAR(100) NOT NULL,
                description TEXT,
                due_date DATE,
                status VARCHAR(50),
                project_id INTEGER REFERENCES projects(project_id)
            );
            
            CREATE TABLE taskassignments (
                assignment_id SERIAL PRIMARY KEY,
                task_id INTEGER REFERENCES tasks(task_id),
                user_id INTEGER REFERENCES users(user_id),
                assigned_date DATE NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE comments (
                comment_id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                task_id INTEGER REFERENCES tasks(task_id),
                user_id INTEGER REFERENCES users(user_id)
            );
            ```
            
            ### Answer
            Here is the SQL query that answers the question: `{question}`
            ```sql
            """
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            generated_ids = self.model.generate(
                **inputs,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=256,  # Adjusted for controlled response length
                do_sample=False,
                num_beams=5,  # Using beam search for better accuracy
            )
            sql_query = (
                self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                .split("```sql")[-1]
                .strip()
            )
            sql_queries.append(sql_query)

        return sql_queries
