# Natural-SQL-7b Truss README

## Introduction to Natural-SQL-7b Truss
Natural-SQL-7b, developed by ChatDB, is a state-of-the-art machine learning model designed for transforming natural language queries into formal SQL commands. This revolutionary model demonstrates exemplary performance in generating accurate SQL statements for complex queries, setting a benchmark in the SQL-Eval framework.

We express our profound appreciation to ChatDB for their groundbreaking work on the Natural-SQL-7b model, significantly advancing the field of natural language processing. For additional details, please visit [HuggingFace Natural-SQL-7b](https://huggingface.co/chatdb/natural-sql-7b), which provides comprehensive insights into the model.

## Deploying Natural-SQL-7b on Baseten
Begin by cloning the Truss examples repository to gain access to a myriad of deployable model examples, including Natural-SQL-7b:

```
git clone https://github.com/basetenlabs/truss-examples/
```

After cloning, deploy Natural-SQL-7b onto Baseten by navigating to the model's directory and executing:

```
truss push natural-sql-7b
```

This command initiates the deployment process, leveraging Truss to seamlessly integrate the model into the Baseten platform.

## Input
The model expects a JSON object comprising a 'questions' key, under which a list of natural language questions aimed at a SQL database is provided:

```json
{
  "questions": ["What is the total revenue for the last quarter?", "Which department has the highest number of employees?"]
}
```

Each question in the list is transformed into an SQL query by the model, demonstrating its capability to understand and interpret complex queries.

## Output
Upon processing the input, Natural-SQL-7b outputs a JSON object containing corresponding SQL queries for each input question:

```json
{
  "queries": [
    "SELECT SUM(revenue) FROM sales WHERE date BETWEEN '2022-01-01' AND '2022-03-31';",
    "SELECT department, COUNT(*) FROM employees GROUP BY department ORDER BY COUNT(*) DESC LIMIT 1;"
  ]
}
```

## Example Usage
To utilize the model on the Baseten platform, execute the model using the `truss predict` command, specifying the model ID and input JSON object:

```
truss predict --model_id=<model_id> --input='{"questions": ["Identify the products with the highest sales volume last month."]}'
```

This example demonstrates querying the model to generate an SQL command that identifies high-volume products, showcasing the model's utility in translating natural language to SQL queries.

For insightful details on leveraging other models within Truss and additional deployment strategies, visit the [Truss Examples root README](https://github.com/basetenlabs/truss-examples).
