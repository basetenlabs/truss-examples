# Natural SQL 7B Truss

NaturalSQL-7B, created by ChatDB, is a state-of-the-art model designed to translate natural language questions into SQL queries efficiently. Thanks to the collaborative effort of researchers and developers at ChatDB and contributions from the wider community such as [Defog](https://github.com/defog-ai/sql-eval) for the SQL-Eval benchmarking framework, this model stands out in the realm of structured query language generation. For a deeper dive into its capabilities, refer to the [model's page on HuggingFace](https://huggingface.co/chatdb/natural-sql-7b) and the official [NaturalSQL Github repository](https://github.com/cfahlgren1/natural-sql).

This model, including its code, is licensed under `apache-2.0`, and the model's data is under `CC BY-SA 4.0`.

## Deploying Natural SQL 7B on Baseten

To use Natural SQL 7B, begin by cloning the truss-examples repository using the command below:

```shell
git clone https://github.com/basetenlabs/truss-examples/
```

Within the `natural-sql-7b` directory, deploy the model onto the Baseten platform by:

```shell
truss push
```

Ensure the Baseten CLI is installed and you are signed into your Baseten account before executing the above command.

## Input

Natural SQL 7B expects a JSON formatted input under the key 'questions:', comprising an array of human language questions. Here’s an example of what the input looks like:

```json
{
  "questions": [
    "Show me the day with the most users joining",
    "What is the ratio of users with gmail addresses vs without?"
  ]
}
```

## Output

The model responds with a JSON array containing SQL queries corresponding to each natural language question from the input. Example of an output:

```json
[
  "SELECT created_at::date AS join_date, COUNT(*) AS user_count FROM users GROUP BY join_date ORDER BY user_count DESC LIMIT 1;",
  "SELECT SUM(CASE WHEN email LIKE '%@gmail.com%' THEN 1 ELSE 0 END) AS gmail_users, SUM(CASE WHEN email NOT LIKE '%@gmail.com%' THEN 1 ELSE 0 END) AS non_gmail_users, (SUM(CASE WHEN email LIKE '%@gmail.com%' THEN 1 ELSE 0 END)::FLOAT / NULLIF(SUM(CASE WHEN email NOT LIKE '%@gmail.com%' THEN 1 ELSE 0 END), 0)) AS gmail_ratio FROM users;"
]

```

## Example Usage

To issue a prediction from the deployed Natural SQL 7B model on Baseten, utilize:

```shell
truss predict --input '{"questions": ["List all projects starting next month"]}'
```

This command will return a SQL query that retrieves projects based on the specified criteria, as demonstrated in the above examples, highlighting the model's proficiency in handling various query complexities.

This guide aims to facilitate your understanding of deploying and engaging with the Natural SQL 7B Model within the Baseten environment, offering insights into its operational dynamics through detailed examples and structured information.
`
