#!/usr/bin/env bash
set -euo pipefail

# FILL ME
AWS_ACCOUNT_ID="863478709086" # Existing AWS account ID
AWS_REGION="us-west-2"     # e.g. us-west-2
S3_BUCKET="863478709086-envelope-test"      # Globally unique bucket name to create
S3_PREFIX="models/custom-weights"
KMS_ALIAS="alias/baseten-oidc-envelope"
ROLE_NAME="BasetenOIDCEnvelopeRole"
BASETEN_ORG_ID="qWGxoxq"  # From `truss whoami --show-oidc`
BASETEN_TEAM_ID="wlpny2q" # From `truss whoami --show-oidc`

# ──────────────────────────────────────────────
# Helpers and configuration validation
# ──────────────────────────────────────────────
require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' was not found" >&2
    exit 1
  fi
}

require_value() {
  if [[ -z "$2" ]]; then
    echo "error: $1 must be set" >&2
    exit 1
  fi
}

file_hex() {
  od -An -tx1 "$1" | tr -d ' \n'
}

for command in aws jq od openssl tr; do
  require_command "${command}"
done

require_value AWS_ACCOUNT_ID "${AWS_ACCOUNT_ID}"
require_value AWS_REGION "${AWS_REGION}"
require_value S3_BUCKET "${S3_BUCKET}"
require_value S3_PREFIX "${S3_PREFIX}"
require_value KMS_ALIAS "${KMS_ALIAS}"
require_value ROLE_NAME "${ROLE_NAME}"
require_value BASETEN_ORG_ID "${BASETEN_ORG_ID}"
require_value BASETEN_TEAM_ID "${BASETEN_TEAM_ID}"

# ──────────────────────────────────────────────
# 1. Authenticate the AWS CLI and verify the account
# ──────────────────────────────────────────────
if ! aws sts get-caller-identity --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "No valid AWS CLI credentials were found."
  echo "  1) Log in with AWS Console credentials (recommended)"
  echo "  2) Configure an access key"
  read -r -p "Choose an authentication method [1]: " auth_method

  case "${auth_method:-1}" in
    1) aws login --region "${AWS_REGION}" ;;
    2) aws configure ;;
    *)
      echo "error: invalid authentication method: ${auth_method}" >&2
      exit 1
      ;;
  esac
fi

caller_account=$(aws sts get-caller-identity \
  --region "${AWS_REGION}" \
  --query Account \
  --output text)
if [[ "${caller_account}" != "${AWS_ACCOUNT_ID}" ]]; then
  echo "error: authenticated to AWS account ${caller_account}, expected ${AWS_ACCOUNT_ID}" >&2
  exit 1
fi

# ──────────────────────────────────────────────
# 2. Create the bucket that will hold encrypted weights
# ──────────────────────────────────────────────
echo "Creating S3 bucket ${S3_BUCKET} in ${AWS_REGION}..."
create_bucket_args=(--bucket "${S3_BUCKET}" --region "${AWS_REGION}")
if [[ "${AWS_REGION}" != "us-east-1" ]]; then
  create_bucket_args+=(
    --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
  )
fi

if ! output=$(aws s3api create-bucket "${create_bucket_args[@]}" 2>&1); then
  if [[ "${output}" == *"BucketAlreadyOwnedByYou"* ]]; then
    echo "S3 bucket already exists and is owned by you, continuing..."
  else
    echo "${output}" >&2
    exit 1
  fi
fi

# ──────────────────────────────────────────────
# 3. Create or reuse the KMS key-encryption key
#
# KMS protects the generated data key. It does not encrypt the model weights
# directly, which avoids KMS payload-size limits.
# ──────────────────────────────────────────────
if KMS_KEY_ARN=$(aws kms describe-key \
  --key-id "${KMS_ALIAS}" \
  --region "${AWS_REGION}" \
  --query KeyMetadata.Arn \
  --output text 2>/dev/null); then
  echo "Using existing KMS key ${KMS_KEY_ARN}."
else
  echo "Creating KMS key ${KMS_ALIAS}..."
  KMS_KEY_ARN=$(aws kms create-key \
    --region "${AWS_REGION}" \
    --description "Envelope encryption key for Baseten model weights" \
    --query KeyMetadata.Arn \
    --output text)
  aws kms create-alias \
    --region "${AWS_REGION}" \
    --alias-name "${KMS_ALIAS}" \
    --target-key-id "${KMS_KEY_ARN}"
fi

# ──────────────────────────────────────────────
# 4. Create small fictional model weights
#
# All plaintext keys and weights remain in a temporary directory that is
# deleted when this script exits.
# ──────────────────────────────────────────────
work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' EXIT

cat >"${work_dir}/weights.json" <<'EOF'
{
  "scale": 2.5,
  "bias": 1.0,
  "description": "Fictional linear-model weights for the OIDC envelope-encryption recipe"
}
EOF

# ──────────────────────────────────────────────
# 5. Ask KMS for an envelope data key
#
# KMS returns the same 64-byte key in two forms: plaintext for this one-time
# encryption operation and encrypted for storage alongside the ciphertext.
# The encryption context must match when the model later calls KMS Decrypt.
# ──────────────────────────────────────────────
encryption_context="purpose=baseten-model-weights,bucket=${S3_BUCKET}"
data_key_response=$(aws kms generate-data-key \
  --key-id "${KMS_KEY_ARN}" \
  --number-of-bytes 64 \
  --encryption-context "${encryption_context}" \
  --region "${AWS_REGION}" \
  --output json)

jq -r '.Plaintext' <<<"${data_key_response}" \
  | tr -d '\n' \
  | openssl base64 -d -A >"${work_dir}/data-key"
jq -r '.CiphertextBlob' <<<"${data_key_response}" \
  | tr -d '\n' \
  | openssl base64 -d -A >"${work_dir}/encrypted-data-key"

# ──────────────────────────────────────────────
# 6. Encrypt and authenticate the weights
#
# Split the data key into independent encryption and MAC keys. The HMAC covers
# IV || ciphertext, so the model verifies integrity before decrypting.
# ──────────────────────────────────────────────
dd if="${work_dir}/data-key" of="${work_dir}/encryption-key" bs=1 count=32 2>/dev/null
dd if="${work_dir}/data-key" of="${work_dir}/mac-key" bs=1 skip=32 count=32 2>/dev/null
openssl rand 16 >"${work_dir}/iv"

openssl enc -aes-256-cbc \
  -K "$(file_hex "${work_dir}/encryption-key")" \
  -iv "$(file_hex "${work_dir}/iv")" \
  -in "${work_dir}/weights.json" \
  -out "${work_dir}/weights.enc"

cat "${work_dir}/iv" "${work_dir}/weights.enc" >"${work_dir}/authenticated-data"
openssl dgst -sha256 \
  -mac HMAC \
  -macopt "hexkey:$(file_hex "${work_dir}/mac-key")" \
  -binary "${work_dir}/authenticated-data" >"${work_dir}/weights.hmac"

# ──────────────────────────────────────────────
# 7. Build the envelope manifest
#
# S3 metadata is not preserved in the mounted filesystem, so the IV, HMAC,
# encryption context, and encrypted-key filename travel in this companion file.
# ──────────────────────────────────────────────
jq --null-input \
  --arg iv "$(openssl base64 -A -in "${work_dir}/iv")" \
  --arg hmac "$(openssl base64 -A -in "${work_dir}/weights.hmac")" \
  --arg bucket "${S3_BUCKET}" \
  '{
    version: 1,
    algorithm: "AES-256-CBC-HMAC-SHA256",
    iv: $iv,
    hmac: $hmac,
    encryption_context: {
      purpose: "baseten-model-weights",
      bucket: $bucket
    }
  }' >"${work_dir}/envelope.json"

# ──────────────────────────────────────────────
# 8. Upload the encrypted payload and its companion files
#
# BDN mounts these three files at /models/custom. No plaintext weights or data
# keys are uploaded.
# ──────────────────────────────────────────────
echo "Uploading encrypted weights to s3://${S3_BUCKET}/${S3_PREFIX}/..."
for artifact in weights.enc encrypted-data-key envelope.json; do
  aws s3 cp "${work_dir}/${artifact}" \
    "s3://${S3_BUCKET}/${S3_PREFIX}/${artifact}" \
    --region "${AWS_REGION}"
done

# ──────────────────────────────────────────────
# 9. Register the Baseten OIDC identity provider
# ──────────────────────────────────────────────
OIDC_ISSUER="oidc.baseten.co"
if ! output=$(aws iam create-open-id-connect-provider \
  --url "https://${OIDC_ISSUER}" 2>&1); then
  if [[ "${output}" != *"EntityAlreadyExists"* ]]; then
    echo "${output}" >&2
    exit 1
  fi
fi

# ──────────────────────────────────────────────
# 10. Create the OIDC role and restrict who may assume it
#
# BDN uses a model_build token to mirror the encrypted S3 objects. The model
# container uses a model_container token to unwrap the data key through KMS.
# ──────────────────────────────────────────────
trust_policy=$(jq --null-input \
  --arg provider "arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_ISSUER}" \
  --arg audience "${OIDC_ISSUER}" \
  --arg build_subject "v=1:org=${BASETEN_ORG_ID}:team=${BASETEN_TEAM_ID}:*:type=model_build" \
  --arg runtime_subject "v=1:org=${BASETEN_ORG_ID}:team=${BASETEN_TEAM_ID}:*:type=model_container" \
  --arg audience_key "${OIDC_ISSUER}:aud" \
  --arg subject_key "${OIDC_ISSUER}:sub" \
  '{
    Version: "2012-10-17",
    Statement: [{
      Effect: "Allow",
      Principal: {Federated: $provider},
      Action: "sts:AssumeRoleWithWebIdentity",
      Condition: {
        StringEquals: {($audience_key): $audience},
        StringLike: {($subject_key): [$build_subject, $runtime_subject]}
      }
    }]
  }')

if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  aws iam update-assume-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-document "${trust_policy}"
else
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${trust_policy}" \
    --description "Baseten OIDC role for decrypting envelope-encrypted weights"
fi

# ──────────────────────────────────────────────
# 11. Grant build-time S3 read and runtime KMS decrypt permissions
#
# S3 access is limited to the encrypted weight prefix. KMS access is limited
# to the key that wrapped this envelope's data key.
# ──────────────────────────────────────────────
weights_policy=$(jq --null-input \
  --arg bucket_arn "arn:aws:s3:::${S3_BUCKET}" \
  --arg object_arn "arn:aws:s3:::${S3_BUCKET}/${S3_PREFIX}/*" \
  --arg prefix "${S3_PREFIX}" \
  --arg key_arn "${KMS_KEY_ARN}" \
  '{
    Version: "2012-10-17",
    Statement: [
      {
        Effect: "Allow",
        Action: "s3:ListBucket",
        Resource: $bucket_arn,
        Condition: {
          StringLike: {"s3:prefix": [$prefix, ($prefix + "/*")]}
        }
      },
      {
        Effect: "Allow",
        Action: "s3:GetObject",
        Resource: $object_arn
      },
      {
        Effect: "Allow",
        Action: "kms:Decrypt",
        Resource: $key_arn
      }
    ]
  }')
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "BasetenEncryptedWeightsAccess" \
  --policy-document "${weights_policy}"

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
role_arn="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
echo
echo "Setup complete."
echo "Encrypted weights: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "KMS key: ${KMS_KEY_ARN}"
echo "OIDC role: ${role_arn}"
echo
echo "Set AWS_ROLE_ARN = \"${role_arn}\" and AWS_REGION = \"${AWS_REGION}\""
echo "in standard-truss/config.yaml and custom-base-image/config.yaml."