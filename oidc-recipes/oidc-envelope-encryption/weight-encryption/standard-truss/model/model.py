import base64
import hashlib
import hmac
import json
import os
from pathlib import Path

import boto3
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ENVELOPE_VERSION = 1
ENVELOPE_ALGORITHM = "AES-256-CBC-HMAC-SHA256"
ENCRYPTED_WEIGHTS_DIR = Path("/models/custom")
ENCRYPTED_WEIGHTS_FILE = ENCRYPTED_WEIGHTS_DIR / "weights.enc"
ENCRYPTED_DATA_KEY_FILE = ENCRYPTED_WEIGHTS_DIR / "encrypted-data-key"
ENVELOPE_FILE = ENCRYPTED_WEIGHTS_DIR / "envelope.json"


def get_kms_client():
    os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"] = os.environ["B10_OIDC_TOKEN_PATH"]
    os.environ["AWS_ROLE_SESSION_NAME"] = "baseten-envelope-decryption"
    return boto3.client("kms", region_name=os.environ["AWS_REGION"])


def read_envelope() -> dict:
    envelope = json.loads(ENVELOPE_FILE.read_text())
    if envelope.get("version") != ENVELOPE_VERSION:
        raise ValueError("Unsupported weights envelope version.")
    if envelope.get("algorithm") != ENVELOPE_ALGORITHM:
        raise ValueError("Unsupported weights envelope algorithm.")
    return envelope


def decrypt_weights(output: Path) -> Path:
    envelope = read_envelope()
    encrypted_data_key = ENCRYPTED_DATA_KEY_FILE.read_bytes()
    ciphertext = ENCRYPTED_WEIGHTS_FILE.read_bytes()

    response = get_kms_client().decrypt(
        CiphertextBlob=encrypted_data_key,
        EncryptionContext=envelope["encryption_context"],
    )
    data_key = response["Plaintext"]
    if len(data_key) != 64:
        raise ValueError("Expected a 64-byte envelope data key from KMS.")

    encryption_key = data_key[:32]
    mac_key = data_key[32:]
    iv = base64.b64decode(envelope["iv"], validate=True)
    expected_mac = base64.b64decode(envelope["hmac"], validate=True)

    actual_mac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(actual_mac, expected_mac):
        raise ValueError("Encrypted weights failed integrity verification.")

    decryptor = Cipher(algorithms.AES(encryption_key), modes.CBC(iv)).decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output.with_name(f"{output.name}.tmp")
    try:
        temporary_path.write_bytes(plaintext)
        temporary_path.chmod(0o600)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


class Model:
    def load(self):
        output = Path(os.environ["DECRYPTED_WEIGHTS_PATH"])
        weights_path = decrypt_weights(output)
        self._weights = json.loads(weights_path.read_text())

    def predict(self, model_input):
        value = float(model_input["value"])
        return {"value": value * self._weights["scale"] + self._weights["bias"]}
