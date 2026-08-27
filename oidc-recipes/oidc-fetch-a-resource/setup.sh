#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# FILL ME: replace these with your values
# ──────────────────────────────────────────────
AWS_ACCOUNT_ID=""             # Your AWS account ID (must exist prior to this script)
AWS_REGION=""                 # AWS region for the S3 bucket (e.g. us-west-2)
S3_BUCKET=""                  # S3 bucket name (to be created)
S3_KEY=""                     # S3 object key (like a file path)
BUCKET_TEXT=""                # Text to store in the S3 object
ROLE_NAME=""                  # IAM role name (to be created)
BASETEN_ORG_ID=""             # From `truss whoami --show-oidc`
BASETEN_TEAM_ID=""            # From `truss whoami --show-oidc`

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
require_non_empty() {
  local _name="$1"
  local _desc="${2:-$1}"
  local _val
  eval "_val=\${${_name}-}"
  if [ -z "$_val" ]; then
    echo "error: ${_desc} is empty; set ${_name} in the configuration section." >&2
    exit 1
  fi
}

require_non_empty AWS_ACCOUNT_ID "AWS account ID"
require_non_empty S3_BUCKET "S3 bucket name"
require_non_empty AWS_REGION "AWS region"
require_non_empty S3_KEY "S3 object key"
require_non_empty ROLE_NAME "IAM role name"
require_non_empty BASETEN_ORG_ID "Baseten organization ID"
require_non_empty BASETEN_TEAM_ID "Baseten team ID"

OIDC_ISSUER="oidc.baseten.co"
OIDC_ISSUER_URL="https://${OIDC_ISSUER}"

# ──────────────────────────────────
# 1. Authenticate the AWS CLI
# ──────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
  echo "error: AWS CLI is not installed or is not on PATH." >&2
  exit 1
fi

if ! aws sts get-caller-identity --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "No valid AWS CLI credentials were found."
  echo "  1) Log in with AWS Console credentials (recommended)"
  echo "  2) Configure an access key"
  read -r -p "Choose an authentication method [1]: " auth_method

  case "${auth_method:-1}" in
    1)
      aws login --region "${AWS_REGION}"
      ;;
    2)
      aws configure
      ;;
    *)
      echo "error: invalid authentication method: ${auth_method}" >&2
      exit 1
      ;;
  esac
fi

if ! caller_account=$(aws sts get-caller-identity \
  --region "${AWS_REGION}" \
  --query Account \
  --output text); then
  echo "error: AWS authentication failed." >&2
  exit 1
fi

if [[ "${caller_account}" != "${AWS_ACCOUNT_ID}" ]]; then
  echo "error: authenticated to AWS account ${caller_account}, expected ${AWS_ACCOUNT_ID}." >&2
  exit 1
fi

echo "Authenticated to AWS account ${caller_account}."

# ──────────────────────────────────
# 2. Create the S3 bucket
# ──────────────────────────────────
echo "Creating S3 bucket ${S3_BUCKET} in ${AWS_REGION}..."

create_bucket_args=(
  --bucket "${S3_BUCKET}"
  --region "${AWS_REGION}"
)

if [[ "${AWS_REGION}" != "us-east-1" ]]; then
  create_bucket_args+=(
    --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
  )
fi

if ! output=$(aws s3api create-bucket "${create_bucket_args[@]}" 2>&1); then
  if [[ "$output" == *"BucketAlreadyOwnedByYou"* ]]; then
    echo "S3 bucket already exists and is owned by you, continuing..."
  else
    echo "$output" >&2
    exit 1
  fi
else
  echo "S3 bucket created."
fi

# ──────────────────────────────────
# 3. Upload a text object to S3
# ──────────────────────────────────
echo "Uploading text to s3://${S3_BUCKET}/${S3_KEY}..."
printf '%s' "${BUCKET_TEXT}" | aws s3 cp - "s3://${S3_BUCKET}/${S3_KEY}" \
  --region "${AWS_REGION}" \
  --content-type "text/plain"

# ──────────────────────────────────
# 4. Create the OIDC identity provider
# ──────────────────────────────────
echo "Creating OIDC identity provider..."

if ! output=$(aws iam create-open-id-connect-provider \
  --url "${OIDC_ISSUER_URL}" 2>&1); then
  if [[ "$output" == *"EntityAlreadyExists"* ]]; then
    echo "OIDC provider already exists, continuing..."
  else
    echo "$output" >&2
    exit 1
  fi
else
  echo "OIDC provider created."
fi

# ──────────────────────────────────
# 5. Create the IAM trust policy
# ──────────────────────────────────
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_ISSUER}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${OIDC_ISSUER}:aud": "${OIDC_ISSUER}"
        },
        "StringLike": {
          "${OIDC_ISSUER}:sub": "v=1:org=${BASETEN_ORG_ID}:team=${BASETEN_TEAM_ID}:*:type=model_container"
        }
      }
    }
  ]
}
EOF
)

# ──────────────────────────────────
# 6. Create the IAM role w/ above trust policy
# ──────────────────────────────────
echo "Creating IAM role ${ROLE_NAME}..."
aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document "${TRUST_POLICY}" \
  --description "Baseten OIDC role for model workloads"

echo "IAM role created."

# ──────────────────────────────────
# 7. Attach S3 read policy to the IAM role
# ──────────────────────────────────
S3_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${S3_BUCKET}",
        "arn:aws:s3:::${S3_BUCKET}/*"
      ]
    }
  ]
}
EOF
)

echo "Attaching S3 read policy..."
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "BasetenS3ReadAccess" \
  --policy-document "${S3_POLICY}"

# ──────────────────────────────────
# Done
# ──────────────────────────────────
ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
echo ""
echo "Setup complete. Set these values in standard-truss/config.yaml:"
echo "  AWS_ROLE_ARN: \"${ROLE_ARN}\""
echo "  AWS_REGION: ${AWS_REGION}"
echo "  S3_BUCKET: ${S3_BUCKET}"
echo "  S3_KEY: ${S3_KEY}"
echo ""