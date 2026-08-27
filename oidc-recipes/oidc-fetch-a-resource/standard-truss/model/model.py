import os
import boto3


class Model:
    def _get_s3_client(self):
        os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"] = os.environ["B10_OIDC_TOKEN_PATH"]
        return boto3.client("s3")

    def _read_text_from_s3(self) -> str:
        resp = self._s3.get_object(Bucket=self._bucket, Key=self._key)
        text = resp["Body"].read().decode("utf-8").strip()
        return text

    def load(self):
        self._bucket = os.environ["S3_BUCKET"]
        self._key = os.environ["S3_KEY"]
        self._s3 = self._get_s3_client()

    def predict(self, model_input):
        text = self._read_text_from_s3()
        print(f"Read from s3://{self._bucket}/{self._key}: {text}")
        return {"text": text}
