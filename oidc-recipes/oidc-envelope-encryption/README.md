# OIDC: Envelope Encryption
_Use customer-owned AWS KMS keys to protect model weights and inference payloads._

These recipes use a Baseten OIDC token to obtain short-lived AWS credentials. The running model can then ask AWS KMS to unwrap a data key without storing a long-lived AWS access key in the Truss.

Two flows are included:

- [Weight encryption](#weight-encryption): Decrypt model weights during startup.
- [Payload encryption](#payload-encryption): Decrypt requests and encrypt responses at inference time.

## Terminology

- **Root Key:** An optional top-level AWS KMS key that protects a KEK in a deeper key hierarchy.
- **KEK (Key Encryption Key):** The customer-managed AWS KMS key that wraps and unwraps the DEK.
- **DEK (Data Encryption Key):** The key that encrypts and decrypts the data.

These recipes do not use a separate Root Key. The customer owns the KEK in their AWS account, and its key material never leaves AWS KMS.

## How envelope encryption works

AWS KMS is designed to protect keys, not to encrypt large data directly. Envelope encryption separates the work:

- A generated DEK encrypts the data.
- The KEK wraps the DEK.
- The encrypted data and wrapped DEK can be stored or sent together.
- An authorized workload asks KMS to unwrap the DEK.

Only the wrapped DEK and its encryption context are sent to KMS. The encrypted data is not.

Both examples use a 64-byte DEK:

- 32 bytes for AES-256-CBC encryption.
- 32 bytes for HMAC-SHA256 integrity protection.

The HMAC covers `IV || ciphertext` and is verified before decryption.

## Weight encryption

This flow keeps model weights encrypted in a customer-owned S3 bucket, through Baseten Data Network (BDN), and in the model mount. The model decrypts them during `load()`.

### Flow

```text
Setup:
  KMS GenerateDataKey
    -> plaintext DEK encrypts weights.json
    -> wrapped DEK is stored with weights.enc in S3

Runtime:
  BDN mounts the encrypted bundle at /models/custom
    -> Baseten OIDC token is exchanged with AWS STS
    -> short-lived credentials call KMS Decrypt
    -> KMS unwraps the DEK
    -> the model verifies and decrypts weights.enc
    -> /tmp/decrypted-weights/weights.json
```

The setup script uploads:

```text
s3://<bucket>/models/custom-weights/
├── weights.enc
├── encrypted-data-key
└── envelope.json
```

The companion `envelope.json` stores the algorithm, IV, HMAC, and KMS encryption context. S3 metadata is not used because the BDN mount exposes files rather than the original S3 response metadata.

### Setup

Prerequisites:

- AWS CLI v2
- `jq`
- OpenSSL
- A Baseten organization and team with OIDC enabled

Get the Baseten OIDC identifiers:

```bash
truss whoami --show-oidc
```

Fill in the values at the top of [`weight-encryption/setup.sh`](weight-encryption/setup.sh), then run:

```bash
cd weight-encryption
./setup.sh
```

The script creates or configures the S3 bucket, KMS key, OIDC provider, IAM role, and encrypted example weights.

### Configure and deploy

Copy the encrypted S3 source, role ARN, and region printed by the setup script into [`weight-encryption/standard-truss/config.yaml`](weight-encryption/standard-truss/config.yaml), then deploy:

```bash
truss push ./weight-encryption/standard-truss
```

For the fictional linear weights, an input value of `4` produces `11`:

```json
{
  "value": 4
}
```

```json
{
  "value": 11.0
}
```

### Custom base image

For vLLM, SGLang, TensorRT-LLM, Triton, or another custom server, run [`weight-encryption/custom-base-image/packages/decrypt.py`](weight-encryption/custom-base-image/packages/decrypt.py) before the server command:

```yaml
docker_server:
  start_command: >-
    sh -c 'python3 /packages/decrypt.py &&
    exec <server-command> /tmp/decrypted-weights'
```

The included custom-base-image config shows the startup hook with vLLM. Its fictional JSON weights cannot be loaded by vLLM; replace them with a real encrypted model directory.

### Performance cost

This flow adds startup time for STS and KMS calls and for local decryption. Decryption time grows with the weight size. It adds no per-request encryption cost after startup.

## Payload encryption

This flow protects inference data beyond transport encryption. The client encrypts each request before sending it to Baseten. Customer-owned code in the model pod decrypts the request, runs inference, and encrypts the response before returning it.

### Flow

```text
Client:
  KMS GenerateDataKey
    -> plaintext DEK encrypts the request
    -> wrapped DEK travels with the encrypted request

Runtime:
  Baseten OIDC token is exchanged with AWS STS
    -> short-lived credentials call KMS Decrypt
    -> KMS unwraps the DEK
    -> the model verifies and decrypts the request
    -> the model encrypts the response with the same DEK and a new IV

Client:
  retained plaintext DEK verifies and decrypts the response
```

The client sends a versioned JSON envelope containing the wrapped DEK, encryption context, IV, HMAC, and ciphertext. The response contains a new IV, HMAC, and ciphertext. Plaintext request and response data only exist in the client and the customer-owned model code.

### Setup

Prerequisites:

- AWS CLI v2
- `jq`
- Python with `boto3` and `cryptography`
- A Baseten organization and team with OIDC enabled

Fill in the values at the top of [`payload-encryption/setup.sh`](payload-encryption/setup.sh), then run:

```bash
cd payload-encryption
./setup.sh
```

The script creates or configures the KMS key, OIDC provider, and runtime IAM role. The runtime role can only call `kms:Decrypt` with the expected encryption context. The AWS identity running the client needs `kms:GenerateDataKey` permission for the same key.

### Configure and deploy

Copy the role ARN and region printed by the setup script into [`payload-encryption/standard-truss/config.yaml`](payload-encryption/standard-truss/config.yaml), then deploy:

```bash
truss push ./payload-encryption/standard-truss
```

On the machine that sends inference requests, authenticate to AWS with an identity that can call `kms:GenerateDataKey` on the KMS key. Then, from the `oidc-envelope-encryption` directory, install the client dependencies and set the deployed model details:

```bash
python -m pip install boto3 cryptography
export AWS_REGION=us-west-2
export KMS_KEY_ARN=arn:aws:kms:us-west-2:123456789012:key/example
export BASETEN_API_KEY=YOUR_API_KEY
export MODEL_URL=https://model-example.api.baseten.co/production/predict
python ./payload-encryption/client.py
```

The example encrypts `{"value": 4}`. The model returns an encrypted response that the client decrypts to:

```json
{
  "value": 9.0
}
```

### Custom base image

The custom-base-image example runs vLLM on an internal port and exposes an encryption proxy on port `8000`. The proxy decrypts each request, forwards the plaintext OpenAI request to vLLM over loopback, and encrypts the response.

Copy the role ARN and region printed by the setup script into [`payload-encryption/custom-base-image/config.yaml`](payload-encryption/custom-base-image/config.yaml), then deploy:

```bash
truss push ./payload-encryption/custom-base-image
```

Set the deployed model's predict URL and an OpenAI-compatible request before running the same client. Baseten forwards the request to the proxy's `/v1/completions` endpoint.

```bash
export MODEL_URL=https://model-example.api.baseten.co/production/predict
export PAYLOAD_JSON='{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Hello","max_tokens":16}'
python ./payload-encryption/client.py
```

Replace the example model and request with the vLLM model you deploy.

### Performance cost

This flow adds a client-side KMS `GenerateDataKey` call and a model-side KMS `Decrypt` call to every request. It also adds request and response encryption time. The cost grows with payload size and affects request latency.

## Security boundary

The OIDC token is exchanged for temporary AWS credentials; it is not sent in the KMS request body. The KEK never leaves AWS KMS. Only the plaintext DEK leaves KMS after an authorized call.

The code running in the pod is customer-owned. After KMS unwraps a DEK, that code is responsible for protecting it and discarding it after use. The client has the same responsibility for DEKs returned by `GenerateDataKey`.

Do not log plaintext DEKs, decrypted data, OIDC tokens, or temporary AWS credentials. Use a separate KMS key or encryption context per environment, tenant, or model family when stronger isolation is needed. AWS CloudTrail records KMS operations.

## Other use cases

The same pattern can protect other model assets:

- Load a customer-specific LoRA adapter only after KMS authorizes the replica.
- Decrypt a private tokenizer, prompt library, or retrieval index during startup.
- Revoke a deployed model's access without rebuilding or deleting its encrypted artifacts.
