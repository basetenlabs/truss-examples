import base64
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request

import boto3
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ENVELOPE_VERSION = 1
ENVELOPE_ALGORITHM = "AES-256-CBC-HMAC-SHA256"
ENCRYPTION_CONTEXT = {"purpose": "baseten-inference-payload"}


def encode(value: bytes) -> str:
    return base64.b64encode(value).decode()


def decode_field(envelope: dict, field: str) -> bytes:
    try:
        return base64.b64decode(envelope[field], validate=True)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid envelope field: {field}.") from error


def encrypt_request(payload: dict) -> tuple[dict, bytes]:
    kms = boto3.client("kms", region_name=os.environ["AWS_REGION"])
    response = kms.generate_data_key(
        KeyId=os.environ["KMS_KEY_ARN"],
        NumberOfBytes=64,
        EncryptionContext=ENCRYPTION_CONTEXT,
    )
    data_key = response["Plaintext"]
    encryption_key = data_key[:32]
    mac_key = data_key[32:]
    iv = os.urandom(16)
    plaintext = json.dumps(payload, separators=(",", ":")).encode()

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()
    encryptor = Cipher(algorithms.AES(encryption_key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    payload_hmac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()

    return {
        "version": ENVELOPE_VERSION,
        "algorithm": ENVELOPE_ALGORITHM,
        "encrypted_data_key": encode(response["CiphertextBlob"]),
        "encryption_context": ENCRYPTION_CONTEXT,
        "iv": encode(iv),
        "hmac": encode(payload_hmac),
        "ciphertext": encode(ciphertext),
    }, data_key


def call_model(envelope: dict) -> dict:
    request = urllib.request.Request(
        os.environ["MODEL_URL"],
        data=json.dumps(envelope).encode(),
        headers={
            "Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def decrypt_response(envelope: dict, data_key: bytes) -> dict:
    if envelope.get("version") != ENVELOPE_VERSION:
        raise ValueError("Unsupported payload envelope version.")
    if envelope.get("algorithm") != ENVELOPE_ALGORITHM:
        raise ValueError("Unsupported payload envelope algorithm.")

    encryption_key = data_key[:32]
    mac_key = data_key[32:]
    iv = decode_field(envelope, "iv")
    ciphertext = decode_field(envelope, "ciphertext")
    expected_mac = decode_field(envelope, "hmac")

    actual_mac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(actual_mac, expected_mac):
        raise ValueError("Encrypted response failed integrity verification.")

    decryptor = Cipher(algorithms.AES(encryption_key), modes.CBC(iv)).decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return json.loads(plaintext)


def main():
    payload = json.loads(os.environ.get("PAYLOAD_JSON", '{"value":4}'))
    envelope, data_key = encrypt_request(payload)
    try:
        encrypted_response = call_model(envelope)
        print(json.dumps(decrypt_response(encrypted_response, data_key), indent=2))
    finally:
        del data_key


if __name__ == "__main__":
    main()
