# OIDC: Envelope Encryption
_Load encrypted weights from S3 and decrypt them by authenticating to AWS KMS with OIDC._

This recipe demonstrates how to keep model weights encrypted until a Baseten model replica starts. The encrypted artifacts are stored in a customer-owned S3 bucket and mirrored through Baseten Data Network (BDN). The running model uses its Baseten OIDC token to obtain short-lived AWS credentials, asks KMS to decrypt the envelope data key, and decrypts the weights locally during `load()`.

No long-lived AWS access key is stored in the Truss or injected into the model container.

## Why envelope encryption?

AWS KMS is designed to protect encryption keys, not to encrypt large model files directly. KMS `Encrypt` and `Decrypt` operations have small payload limits, while model weights may be many gigabytes.

Envelope encryption separates the work:

- A random **data key** encrypts the large payload locally.
- A KMS **key-encryption key** encrypts, or wraps, that data key.
- The encrypted payload and encrypted data key can be stored together safely.
- A workload must be authorized by KMS before it can recover the plaintext data key.

This gives you centralized access control and audit logs in KMS without sending the full weights through KMS.

## Flow

```text
Setup:
  KMS GenerateDataKey
    -> plaintext data key: encrypts weights.json locally
    -> encrypted data key: stored with weights.enc in S3

Runtime:
  BDN mounts the encrypted bundle at /models/custom
    -> Baseten OIDC token is exchanged with AWS STS
    -> STS returns short-lived AWS credentials
    -> credentials sign a KMS Decrypt request containing:
         encrypted-data-key + encryption context
    -> KMS returns the plaintext data key
    -> verify the HMAC and decrypt weights.enc locally
    -> /tmp/decrypted-weights/weights.json
```

The OIDC token is used to obtain temporary AWS credentials; it is not sent in the KMS request body. The encrypted model weights are never sent to KMS. Only the small encrypted data key and its encryption context are sent to KMS, and the returned plaintext data key is used inside the model container.

## Encrypted bundle

The setup script uploads three files:

```text
s3://<bucket>/models/custom-weights/
├── weights.enc          # Encrypted weight payload
├── encrypted-data-key   # 64-byte data key wrapped by KMS
└── envelope.json        # Algorithm, IV, HMAC, and KMS encryption context
```

S3 object metadata is not used because the BDN mount exposes files, not the original S3 `GetObject` response metadata. Keeping the envelope information in a companion file also makes the bundle portable between object stores and local test environments.

The sample splits the 64-byte plaintext data key into:

- 32 bytes for AES-256-CBC encryption.
- 32 bytes for HMAC-SHA256 integrity protection.

The model verifies the HMAC over `IV || ciphertext` before decrypting. This is important: encryption without integrity protection would not detect a modified payload.

## What the recipe creates

[`setup.sh`](setup.sh) creates or configures:

- A customer-owned S3 bucket and encrypted weight prefix.
- A customer-owned KMS key and alias.
- A small fictional `weights.json` containing `scale` and `bias` values.
- An OIDC provider for `oidc.baseten.co`.
- An IAM role trusted by the configured Baseten organization and team.
- Prefix-scoped S3 read permissions and key-scoped KMS decrypt permission.

The fictional weights keep the example runnable and make it easy to verify decryption. For a real model, replace `weights.json` with a model archive, SafeTensors file, adapter, tokenizer bundle, or other artifact and load the decrypted output with the appropriate framework.

## Setup

Prerequisites:

- AWS CLI v2
- `jq`
- OpenSSL
- A Baseten organization and team with OIDC enabled

Get the Baseten OIDC identifiers:

```bash
truss whoami --show-oidc
```

Fill in the values at the top of [`setup.sh`](setup.sh), then run:

```bash
./setup.sh
```

The script prints the S3 source, KMS key ARN, and IAM role ARN when it finishes.

## Configure and deploy

Use the encrypted S3 prefix as the weight source. OIDC authentication is needed here because BDN mirrors the encrypted files before the model container starts:

```yaml
runtime:
  oidc:
    enabled: true

weights:
  - source: "s3://my-bucket/models/custom-weights"
    mount_location: "/models/custom"
    auth:
      auth_method: AWS_OIDC
      aws_oidc_role_arn: "arn:aws:iam::123456789012:role/BasetenOIDCEnvelopeRole"
      aws_oidc_region: us-west-2
```

Set the same weight and runtime values in the config you want to deploy:

```yaml
environment_variables:
  AWS_ROLE_ARN: "arn:aws:iam::123456789012:role/BasetenOIDCEnvelopeRole"
  AWS_REGION: us-west-2
  DECRYPTED_WEIGHTS_PATH: /tmp/decrypted-weights/weights.json
```

Deploy the standard Truss:

```bash
truss push ./standard-truss
```

At startup, `Model.load()`:

1. Reads the mounted envelope and encrypted data key.
2. Uses `B10_OIDC_TOKEN_PATH` with `AssumeRoleWithWebIdentity` through boto3.
3. Calls KMS `Decrypt` with the recorded encryption context.
4. Verifies the ciphertext HMAC.
5. Decrypts the weights into `/tmp/decrypted-weights/weights.json` with mode `0600`.
6. Loads the decrypted values before the replica becomes ready.

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

## Custom base image

For a custom base image such as vLLM, SGLang, TensorRT-LLM, or Triton, run [`custom-base-image/packages/decrypt.py`](custom-base-image/packages/decrypt.py) before the server command. Truss copies it to `/packages/decrypt.py`.

```yaml
requirements:
  - boto3
  - cryptography
```

Then chain decryption and server startup through `docker_server.start_command`:

```yaml
docker_server:
  start_command: >-
    sh -c 'python3 /packages/decrypt.py &&
    exec <server-command> /tmp/decrypted-weights/weights.json'
```

Replace `<server-command>` with the image's normal vLLM, SGLang, TensorRT-LLM, Triton, or other launch command. The `&&` prevents the server from starting if decryption fails, and `exec` ensures the server receives container signals directly.

The included [`custom-base-image/config.yaml`](custom-base-image/config.yaml) uses vLLM. Deploy it with:

```bash
truss push ./custom-base-image
```

Both implementations read `AWS_ROLE_ARN`, `AWS_REGION`, `DECRYPTED_WEIGHTS_PATH`, and the Baseten-provided `B10_OIDC_TOKEN_PATH` from the environment. They read encrypted weights from `/models/custom`.

The custom-base-image config demonstrates the startup-hook structure. Its fictional JSON weights cannot actually be loaded by vLLM; replace the payload and model argument with a real decrypted model directory for an inference deployment.

## Security boundary

This recipe protects weights while they are stored in S3, mirrored by BDN, and mounted as encrypted files. Plaintext exists only after the running model has successfully authenticated to KMS and decrypted the payload into its writable container filesystem.

The sample uses one IAM role for both BDN S3 access and runtime KMS access. This keeps the recipe approachable. In a stricter production design, use separate roles:

- A `model_build` role that can only read the encrypted S3 prefix.
- A `model_container` role that can only call `kms:Decrypt` for the required KMS key.

Also consider one KMS key or encryption context per environment, tenant, or model family. AWS CloudTrail can then provide an audit record of every data-key unwrap.

Do not log the plaintext data key, decrypted weights, OIDC token, or temporary AWS credentials. Delete decrypted artifacts when they are no longer needed, and prefer an in-memory loader when the model framework supports one.

## Other use cases

The same pattern applies whenever a model must process sensitive payloads but should not carry a long-lived decryption credential. The snippets below focus on business logic and assume `decrypt_envelope()` and `encrypt_envelope()` implement the same authenticated envelope format.

- Decrypt an inference request supplied by a customer application

  ```python
  def predict(self, model_input):
      request = json.loads(self.decrypt_envelope(model_input["encrypted_request"]))
      return self.model.generate(request["prompt"])
  ```

- Return an encrypted inference response so only the customer can read it

  ```python
  def predict(self, model_input):
      response = self.model.generate(model_input["prompt"])
      return {"encrypted_response": self.encrypt_envelope(response.encode())}
  ```

- Decrypt both the request and response for end-to-end application-level protection

  ```python
  def predict(self, model_input):
      request = json.loads(self.decrypt_envelope(model_input["payload"]))
      response = self.model.generate(request["prompt"])
      return {"payload": self.encrypt_envelope(response.encode())}
  ```

- Load a customer-specific LoRA adapter only after KMS authorizes the replica

  ```python
  def predict(self, model_input):
      adapter = self.decrypt_envelope(self.read_adapter(model_input["tenant_id"]))
      self.model.load_adapter(adapter)
      return self.model.generate(model_input["prompt"])
  ```

- Decrypt a private tokenizer, prompt library, or retrieval index during startup

  ```python
  def load(self):
      index_bytes = self.decrypt_envelope(Path("/models/private/index.enc").read_bytes())
      self.index = VectorIndex.from_bytes(index_bytes)
  ```

- Encrypt transcripts, embeddings, or generated documents before writing them to storage

  ```python
  def predict(self, model_input):
      transcript = self.transcriber(model_input["audio"])
      encrypted = self.encrypt_envelope(transcript.encode())
      self.write_result(model_input["request_id"], encrypted)
      return {"stored": True}
  ```

- Use a tenant-specific KMS key to enforce cryptographic tenant isolation

  ```python
  def predict(self, model_input):
      kms_key = self.tenant_keys[model_input["tenant_id"]]
      document = self.decrypt_envelope(model_input["document"], kms_key=kms_key)
      return self.classifier(document)
  ```

- Revoke a deployed model's access without rebuilding or deleting its encrypted artifacts

  ```python
  def load(self):
      # Startup fails closed after kms:Decrypt is removed from the runtime role.
      self.weights = self.decrypt_envelope(Path("/models/custom/weights.enc").read_bytes())
  ```
