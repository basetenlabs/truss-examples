import asyncio
import base64
import os
from pathlib import Path

import aiohttp

API_BASE = "https://model-xxxx.api.baseten.co/deployment/31d4m09/sync"
MODEL_NAME = "voxcpm2"


def _api_headers() -> dict[str, str]:
    return {
        "Authorization": f"Api-Key {os.getenv('BASETEN_API_KEY')}",
        "Content-Type": "application/json",
    }


def _audio_data_url(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower() or "wav"
    mime = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }.get(ext, "audio/wav")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


async def synthesize(
    session: aiohttp.ClientSession,
    payload: dict,
    out_path: Path,
) -> None:
    async with session.post(
        f"{API_BASE}/v1/audio/speech",
        headers=_api_headers(),
        json=payload,
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"{resp.status}: {text[:500]}")
        with out_path.open("wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                f.write(chunk)
        print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")


async def main() -> None:
    prompt_wav = Path(__file__).with_name("prompt_audio.wav")
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Missing prompt wav: {prompt_wav}")

    ref_audio = _audio_data_url(prompt_wav)
    ref_text = "Hello, this is a test of text to speech."

    jobs = [
        # Zero-shot synthesis (no reference audio).
        (
            {
                "model": MODEL_NAME,
                "input": "Hello from VoxCPM2 running on Baseten.",
                "voice": "default",
                "response_format": "mp3",
            },
            Path("out_zeroshot.mp3"),
        ),
        # Voice cloning with the local prompt_audio.wav.
        (
            {
                "model": MODEL_NAME,
                "input": "Hey this is a cloned voice! How do you think I sound?",
                "voice": "default",
                "response_format": "mp3",
                "ref_audio": ref_audio,
                "ref_text": ref_text,
            },
            Path("out_prompted.mp3"),
        ),
    ]

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *[synthesize(session, payload, out_path) for payload, out_path in jobs]
        )


if __name__ == "__main__":
    asyncio.run(main())
