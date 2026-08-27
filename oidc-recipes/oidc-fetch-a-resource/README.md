# OIDC: Fetch a Resource
_Application: Fetching text from an object in an S3 bucket_

This recipe demonstrates a basic and flexible pattern: fetch a resource from a remote host _without_ storing any long-term credentials. Instead, configure trust in the B10 IdP and establish a session by sending the OIDC token.

## Setup

1. Run `truss whoami --show-oidc` to get your Baseten organization and team IDs.
2. Fill in the values at the top of [`setup.sh`](setup.sh).
3. Run `./setup.sh`.
4. Copy the printed role, region, bucket, and key into [`standard-truss/config.yaml`](standard-truss/config.yaml).
5. Deploy with `truss push ./standard-truss`.

At runtime, `Model.load()` maps `B10_OIDC_TOKEN_PATH` to boto3's web-identity token variable. boto3 exchanges the token for short-lived AWS credentials. Each prediction reads the configured S3 object and returns its text:

```json
{"text": "Contents of the S3 object"}
```

## Other uses

The resource contents can support many use cases. The snippets below assume `read_text()` and `write_text()` access the authenticated remote resource:

- use extra context in an inference request (e.g. company report documents)

  ```python
  def predict(self, model_input):
	  report = self.read_text("reports/q2-2026.txt")
	  return self.llm.generate(f"Report:\n{report}\n\nQuestion: {model_input['question']}")
  ```

- write results of an inference request to the remote resource

  ```python
  def predict(self, model_input):
	  result = self.llm.generate(model_input["prompt"])
	  self.write_text(f"results/{model_input['request_id']}.txt", result)
	  return {"result": result}
  ```

- pull initialization configuration without hardcoding it in your Truss (configuration can live remotely in one place and be read on every `load()` or even every `predict()`)

  ```python
  def load(self):
	  self.settings = json.loads(self.read_text("config/production.json"))
	  self.model = load_model(self.settings["model_id"])
  ```

- load a tenant-specific prompt template from `s3://company-prompts/acme/support-agent.txt`

  ```python
  def predict(self, model_input):
	  prompt = self.read_text(f"prompts/{model_input['tenant_id']}/support-agent.txt")
	  return self.llm.generate(f"{prompt}\n\nCustomer: {model_input['message']}")
  ```

- fetch the latest product catalog before answering a recommendation request

  ```python
  def predict(self, model_input):
	  catalog = self.read_text("catalog/current.json")
	  return self.llm.generate(f"Catalog: {catalog}\nRecommend: {model_input['request']}")
  ```

- read a JSON feature-flag file that changes model behavior without redeploying the model

  ```python
  def predict(self, model_input):
	  flags = json.loads(self.read_text("config/feature-flags.json"))
	  model = self.fast_model if flags["use_fast_model"] else self.quality_model
	  return model.generate(model_input["prompt"])
  ```

- retrieve a customer-specific glossary before translating industry-specific documents

  ```python
  def predict(self, model_input):
	  glossary = self.read_text(f"glossaries/{model_input['customer_id']}.txt")
	  return self.translator.translate(model_input["text"], glossary=glossary)
  ```

- load a list of blocked terms or compliance rules before generating a response

  ```python
  def predict(self, model_input):
	  blocked_terms = self.read_text("compliance/blocked-terms.txt").splitlines()
	  response = self.llm.generate(model_input["prompt"])
	  return {"response": redact(response, blocked_terms)}
  ```

- fetch a small set of few-shot examples selected for a particular workflow

  ```python
  def predict(self, model_input):
	  examples = self.read_text(f"examples/{model_input['workflow']}.jsonl")
	  return self.llm.generate(f"Examples:\n{examples}\n\nInput: {model_input['text']}")
  ```

- read model routing configuration that selects an upstream model or endpoint

  ```python
  def predict(self, model_input):
	  routes = json.loads(self.read_text("routing/models.json"))
	  model = self.clients[routes[model_input["task"]]]
	  return model.generate(model_input["prompt"])
  ```

- retrieve private certificates, public keys, or trust bundles needed to call another internal service

  ```python
  def load(self):
	  trust_bundle = self.read_text("certificates/internal-ca.pem")
	  self.internal_client = InternalClient(ca_certificate=trust_bundle)
  ```

- write generated transcripts, summaries, or embeddings back to a customer-owned bucket

  ```python
  def predict(self, model_input):
	  transcript = self.transcriber.transcribe(model_input["audio"])
	  self.write_text(f"transcripts/{model_input['customer_id']}.txt", transcript)
	  return {"transcript": transcript}
  ```

- check for a kill-switch file that disables a workflow without requiring a new deployment

  ```python
  def predict(self, model_input):
	  if self.read_text("controls/summarization-enabled.txt").strip() != "true":
		  return {"error": "Summarization is temporarily disabled"}
	  return {"summary": self.summarizer(model_input["text"])}
  ```
