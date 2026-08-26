import os
import boto3

# FILL ME
AWS_ROLE_ARN = ""  # e.g. arn:aws:iam::123456789012:role/BasetenOIDCRole
AWS_REGION = ""    # e.g. us-west-2
S3_BUCKET = ""     # e.g. mybucket
S3_KEY = ""        # e.g. inputs/review.txt

class Model:
    def __init__(self, **kwargs):
        self._oidc_token_file = os.environ.get("B10_OIDC_TOKEN_PATH")
        self._s3 = None

    def _get_s3_client(self):
        os.environ["AWS_ROLE_ARN"] = AWS_ROLE_ARN
        os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"] = self._oidc_token_file
        os.environ["AWS_REGION"] = AWS_REGION
        return boto3.client("s3", region_name=AWS_REGION)

    def _read_text_from_s3(self) -> str:
        resp = self._s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        text = resp["Body"].read().decode("utf-8").strip()
        return text

    def load(self):
        self._s3 = self._get_s3_client()

    def predict(self, model_input):
        text = self._read_text_from_s3()
        print(f"Read from s3://{S3_BUCKET}/{S3_KEY}: {text}")
        return { "text": text }