import base64
import hashlib
import hmac
import json
import os

import boto3
import httpx
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

ENVELOPE_VERSION = 1
ENVELOPE_ALGORITHM = "AES-256-CBC-HMAC-SHA256"
VLLM_URL = "http://127.0.0.1:8001"

app = FastAPI()


@app.get("/live")
async def live():
    return {"status": "ok"}


def get_kms_client():
    os.environ["AWS_WEB_IDENTITY_TOKEN_FILE"] = os.environ["B10_OIDC_TOKEN_PATH"]
    os.environ["AWS_ROLE_SESSION_NAME"] = "baseten-payload-decryption"
    return boto3.client("kms", region_name=os.environ["AWS_REGION"])


def decode_field(envelope: dict, field: str) -> bytes:
    try:
        return base64.b64decode(envelope[field], validate=True)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid envelope field: {field}.") from error


def decrypt_request(envelope: dict) -> tuple[dict, bytes]:
    if envelope.get("version") != ENVELOPE_VERSION:
        raise ValueError("Unsupported payload envelope version.")
    if envelope.get("algorithm") != ENVELOPE_ALGORITHM:
        raise ValueError("Unsupported payload envelope algorithm.")

    response = get_kms_client().decrypt(
        CiphertextBlob=decode_field(envelope, "encrypted_data_key"),
        EncryptionContext=envelope["encryption_context"],
    )
    data_key = response["Plaintext"]
    if len(data_key) != 64:
        raise ValueError("Expected a 64-byte envelope data key from KMS.")

    encryption_key = data_key[:32]
    mac_key = data_key[32:]
    iv = decode_field(envelope, "iv")
    ciphertext = decode_field(envelope, "ciphertext")
    expected_mac = decode_field(envelope, "hmac")

    actual_mac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(actual_mac, expected_mac):
        raise ValueError("Encrypted request failed integrity verification.")

    decryptor = Cipher(algorithms.AES(encryption_key), modes.CBC(iv)).decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return json.loads(plaintext), data_key


def encrypt_response(payload: dict, data_key: bytes) -> dict:
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
        "iv": base64.b64encode(iv).decode(),
        "hmac": base64.b64encode(payload_hmac).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{VLLM_URL}/health")
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="vLLM is not ready.") from error
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{VLLM_URL}/metrics")
            response.raise_for_status()
    except httpx.HTTPError:
        return Response(
            content="payload_proxy_vllm_ready 0\n",
            media_type="text/plain; version=0.0.4",
        )
    return Response(content=response.content, media_type="text/plain")


@app.post("/v1/completions")
async def completions(envelope: dict):
    try:
        request_payload, data_key = decrypt_request(envelope)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{VLLM_URL}/v1/completions", json=request_payload
            )
            response.raise_for_status()
        return encrypt_response(response.json(), data_key)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="vLLM request failed.") from error
    finally:
        del data_key
