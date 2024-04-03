# Natural SQL 7B Truss

Created by the innovative minds at ChatDB, Natural SQL 7B represents a premier solution for converting natural language queries into SQL commands seamlessly. This model highlights the collective genius of the ChatDB team and contributions from wider community projects, including the SQL-Eval framework by Defog. Dive deeper into its capabilities at [HuggingFace](https://huggingface.co/chatdb/natural-sql-7b) and explore the repository on [GitHub](https://github.com/cfahlgren1/natural-sql).

This model and its code are under an `apache-2.0` license, with the dataset licensed under `CC BY-SA 4.0`, promoting open and flexible usage.

## Deploying Natural SQL 7B on Baseten

Deploying this model with Baseten takes just a few steps. Start by cloning the truss-examples repository:

```shell
git clone https://github.com/basetenlabs/truss-examples/
```

Next, position yourself within the `natural-sql-7b` directory. With the Baseten CLI installed and after you have logged into your Baseten account, execute:

```shell
truss push
```

This command deploys the model to Baseten, making it ready for use.

## Input

Expecting JSON formatted input, Natural SQL 7B looks for the key 'questions' containing an array of your queries. An input example:

```json
{
  "questions": [
    "Which day saw the most user registrations?",
    "Ratio of users with vs without gmail accounts?"
  ]
}
```

## Output

Corresponding SQL queries for each question are provided in a JSON array, showcasing the model's translation excellence.

```json
[
  "SELECT created_at::date AS join_date, COUNT(*) AS user_count FROM users GROUP BY join_date ORDER BY user_count DESC LIMIT 1;",
  "SELECT SUM(CASE WHEN email LIKE '%@gmail.com%' THEN 1 ELSE 0 END) AS gmail_users, SUM(CASE WHEN email NOT LIKE '%@gmail.com%' THEN 1 ELSE 0 END) AS non_gmail_users, (SUM(CASE WHEN email LIKE '%@gmail.com%' THEN 1 ELSE 0 END)::FLOAT / NULLIF(SUM(CASE WHEN email NOT LIKE '%@gmail.com%' THEN 1 ELSE 0 END), 0)) AS gmail_ratio FROM users;"
]

## Example Usage

To obtain SQL queries from your natural language questions:

```shell
truss predict --input '{"questions": ["Identify all active projects for the next month."]}'
```

This command interacts with the model to provide a relevant SQL query, demonstrating the model's capacity to address a variety of queries.

For further exploration and more detailed operation instructions, refer to the documentation and examples on the Baseten documentation site and the Truss example GitHub repository.
