# Natural SQL 7B Truss

Natural SQL 7B, developed by ChatDB and benchmarked using the SQL-Eval framework by the Defog team, is a groundbreaking model for converting natural language queries into sequential query language (SQL) with unparalleled accuracy. This model represents a significant advancement in natural language processing for databases. For more detailed information and to reference the source, visit the [Natural SQL 7B on Hugging Face](https://huggingface.co/chatdb/natural-sql-7b).

## Deploying Natural SQL 7B on Baseten

To deploy the Natural SQL 7B model on Baseten:

1. Ensure you have cloned the truss-examples repository:
    ```
    git clone https://github.com/basetenlabs/truss-examples/
    ```
2. Navigate to the `natural-sql-7b` directory within the cloned repository.
3. Deploy the truss to Baseten using the following command:
    ```
    truss push natural-sql-7b
    ```

## Input

The model expects a JSON formatted input specifying 'question' and 'schema'. 'question' should be a natural language query, and 'schema' should describe the relevant database schema in string format. For example:
```json
{
  "question": "Show me the day with the most users joining",
  "schema": "CREATE TABLE users (user_id SERIAL PRIMARY KEY, username VARCHAR(50) NOT NULL, email VARCHAR(100) NOT NULL, password_hash TEXT NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP);"
}
```

## Output

The output is provided in JSON format. It includes the 'question' and a 'sql_query' – the SQL query representing the answer to the natural language question. For instance:
```json
{
  "sql_query": "SELECT created_at::date AS join_date, COUNT(*) AS user_count FROM users GROUP BY join_date ORDER BY user_count DESC LIMIT 1;"
}
```

## Example Usage

An example command to use the model with Baseten is provided below:
```shell
truss predict --input '{"question": "Show me the day with the most users joining", "schema": "CREATE TABLE users (user_id SERIAL PRIMARY KEY, username VARCHAR(50) NOT NULL, email VARCHAR(100) NOT NULL, password_hash TEXT NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP);"}'
```

Expected output from the above command:
```json
{
  "question": "Show me the day with the most users joining",
  "sql_query": "SELECT created_at::date AS join_date, COUNT(*) AS user_count FROM users GROUP BY join_date ORDER BY user_count DESC LIMIT 1;"
}
```

This quick example underlines the model's capability to accurately transform compendious natural language questions into executable SQL queries.

## License

This model and its underlying codebase are made available under the Apache 2.0 license. The model itself is distributed under the CC BY-SA 4.0 license, ensuring that it can be freely used and shared within the community under the same license terms.
