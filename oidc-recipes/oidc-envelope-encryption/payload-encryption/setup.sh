#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# FILL ME: replace these with your values
# ──────────────────────────────────────────────
AWS_ACCOUNT_ID=""  # Existing AWS account ID
AWS_REGION=""      # AWS region for KMS (e.g. us-west-2)
KMS_ALIAS=""       # KMS alias to create or reuse (must start with alias/)
ROLE_NAME=""       # IAM role name to create or update
BASETEN_ORG_ID=""  # From `truss whoami --show-oidc`
BASETEN_TEAM_ID="" # From `truss whoami --show-oidc`

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

for command in aws jq; do
  require_command "${command}"
done

require_value AWS_ACCOUNT_ID "${AWS_ACCOUNT_ID}"
require_value AWS_REGION "${AWS_REGION}"
require_value KMS_ALIAS "${KMS_ALIAS}"
require_value ROLE_NAME "${ROLE_NAME}"
require_value BASETEN_ORG_ID "${BASETEN_ORG_ID}"
require_value BASETEN_TEAM_ID "${BASETEN_TEAM_ID}"

if [[ "${KMS_ALIAS}" != alias/* ]]; then
  echo "error: KMS_ALIAS must start with alias/" >&2
  exit 1
fi

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
# 2. Create or reuse the KMS key-encryption key
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
    --description "Envelope encryption key for Baseten inference payloads" \
    --query KeyMetadata.Arn \
    --output text)
  aws kms create-alias \
    --region "${AWS_REGION}" \
    --alias-name "${KMS_ALIAS}" \
    --target-key-id "${KMS_KEY_ARN}"
fi

# ──────────────────────────────────────────────
# 3. Register the Baseten OIDC identity provider
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
# 4. Create the runtime role and restrict who may assume it
# ──────────────────────────────────────────────
trust_policy=$(jq --null-input \
  --arg provider "arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_ISSUER}" \
  --arg audience "${OIDC_ISSUER}" \
  --arg subject "v=1:org=${BASETEN_ORG_ID}:team=${BASETEN_TEAM_ID}:*:type=model_container" \
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
        StringLike: {($subject_key): $subject}
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
    --description "Baseten OIDC role for decrypting inference payloads"
fi

# ──────────────────────────────────────────────
# 5. Grant the runtime permission to unwrap payload data keys
# ──────────────────────────────────────────────
payload_policy=$(jq --null-input \
  --arg key_arn "${KMS_KEY_ARN}" \
  '{
    Version: "2012-10-17",
    Statement: [{
      Effect: "Allow",
      Action: "kms:Decrypt",
      Resource: $key_arn,
      Condition: {
        StringEquals: {
          "kms:EncryptionContext:purpose": "baseten-inference-payload"
        }
      }
    }]
  }')
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "BasetenPayloadDecryptAccess" \
  --policy-document "${payload_policy}"

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
role_arn="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
echo
echo "Setup complete."
echo "KMS key: ${KMS_KEY_ARN}"
echo "OIDC role: ${role_arn}"
echo
echo "Set AWS_ROLE_ARN = \"${role_arn}\" and AWS_REGION = \"${AWS_REGION}\""
echo "in standard-truss/config.yaml or custom-base-image/config.yaml."
echo "Set KMS_KEY_ARN = \"${KMS_KEY_ARN}\" and AWS_REGION = \"${AWS_REGION}\""
echo "when running client.py."
