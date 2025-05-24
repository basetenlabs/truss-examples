import asyncio
import aiohttp
import uuid
import time
import struct
from concurrent.futures import ProcessPoolExecutor

# Configuration
MODEL = "03yxn1lq"
BASETEN_HOST = f"https://model-{MODEL}.api.baseten.co/production/predict"
BASETEN_API_KEY = "zcqOMgeU.xGJlLqKffJHr1ASqJ5qBC9QVP5wLV9Z8"
PAYLOADS_PER_PROCESS = 1
NUM_PROCESSES = 4
MAX_REQUESTS_PER_PROCESS = 8

# Sample prompts
prompts = [
    """Hello there.
Thank you for calling our support line.
My name is Sarah and I'll be helping you today.
Could you please provide your account number and tell me what issue you're experiencing?"""
]
prompt_types = ["short", "medium", "long"]

base_request_payload = {
    "max_tokens": 4096,
    "voice": "tara",
    "stop_token_ids": [128258, 128009],
}


async def stream_to_buffer(
    session: aiohttp.ClientSession, label: str, payload: dict
) -> bytes:
    """Send one streaming request, accumulate into bytes, and log timings."""
    req_id = str(uuid.uuid4())
    payload = {**payload, "request_id": req_id}

    t0 = time.perf_counter()

    try:
        async with session.post(
            BASETEN_HOST,
            json=payload,
            headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
        ) as resp:
            if resp.status != 200:
                print(f"[{label}] ← HTTP {resp.status}")
                return b""

            buf = bytearray()
            idx = 0
            # *** CORRECTED: async for on the AsyncStreamIterator ***
            async for chunk in resp.content.iter_chunked(4_096):
                elapsed_ms = (time.perf_counter() - t0) * 1_000
                if idx in [0, 10]:
                    print(
                        f"[{label}] ← chunk#{idx} ({len(chunk)} B) @ {elapsed_ms:.1f} ms"
                    )
                buf.extend(chunk)
                idx += 1

            total_s = time.perf_counter() - t0
            print(f"[{label}] ← done {len(buf)} B in {total_s:.2f}s")
            return bytes(buf)

    except Exception as e:
        print(f"[{label}] ⚠️ exception: {e!r}")
        return b""


async def run_session(
    session: aiohttp.ClientSession,
    prompt: str,
    ptype: str,
    run_id: int,
    semaphore: asyncio.Semaphore,
) -> None:
    """Wrap a single prompt run in its own error‐safe block."""
    label = f"{ptype}_run{run_id}"
    async with semaphore:
        try:
            payload = {**base_request_payload, "prompt": f"Chapter {run_id}: {prompt}"}
            buf = await stream_to_buffer(session, label, payload)
            if not buf:
                print(f"[{label}] 🛑 no data received")
                return
            elif run_id < 3:
                fn = f"output_{ptype}_run{run_id}.wav"
                # Convert raw PCM data to proper WAV format
                wav_data = pcm_to_wav(buf)
                with open(fn, "wb") as f:
                    f.write(wav_data)
                print(f"[{label}] ➔ saved {fn} (converted to WAV)")

        except Exception as e:
            print(f"[{label}] 🛑 failed: {e!r}")


async def run_with_offset(offset: int) -> None:
    semph = asyncio.Semaphore(MAX_REQUESTS_PER_PROCESS)
    connector = aiohttp.TCPConnector(limit_per_host=128, limit=128)
    async with aiohttp.ClientSession(connector=connector) as session:
        # warmup once per worker
        await run_session(session, "warmup", "warmup", 90 + offset, semph)

        tasks = []
        for i, prompt in enumerate(prompts):
            ptype = prompt_types[i]
            print(f"\nWorker@offset {offset} ▶ {ptype} prompt starts…")
            for run_id in range(offset, offset + PAYLOADS_PER_PROCESS):
                tasks.append(run_session(session, prompt, ptype, run_id, semph))

        await asyncio.gather(*tasks)
        print(f"Worker@offset {offset} ✅ all done.")


def run_with_offset_sync(offset: int) -> None:
    try:
        # create and run a fresh event loop in each process
        asyncio.run(run_with_offset(offset))
    except Exception as e:
        print(f"Worker@offset {offset} ❌ error: {e}")


def main():
    offsets = [i * PAYLOADS_PER_PROCESS for i in range(NUM_PROCESSES)]
    with ProcessPoolExecutor() as exe:
        # map each offset to its own process
        exe.map(run_with_offset_sync, offsets)

    print("🎉 All processes completed.")


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Convert raw PCM data to WAV format with proper headers."""
    # WAV file header
    # RIFF header
    riff_header = b"RIFF"
    # File size (will be calculated)
    file_size = 36 + len(pcm_data)
    file_size_bytes = struct.pack("<I", file_size)
    # WAVE header
    wave_header = b"WAVE"

    # fmt subchunk
    fmt_header = b"fmt "
    fmt_size = struct.pack("<I", 16)  # PCM format size
    audio_format = struct.pack("<H", 1)  # PCM format
    num_channels = struct.pack("<H", channels)
    sample_rate_bytes = struct.pack("<I", sample_rate)
    byte_rate = struct.pack("<I", sample_rate * channels * bits_per_sample // 8)
    block_align = struct.pack("<H", channels * bits_per_sample // 8)
    bits_per_sample_bytes = struct.pack("<H", bits_per_sample)

    # data subchunk
    data_header = b"data"
    data_size = struct.pack("<I", len(pcm_data))

    # Combine all parts
    wav_data = (
        riff_header
        + file_size_bytes
        + wave_header
        + fmt_header
        + fmt_size
        + audio_format
        + num_channels
        + sample_rate_bytes
        + byte_rate
        + block_align
        + bits_per_sample_bytes
        + data_header
        + data_size
        + pcm_data
    )

    return wav_data


if __name__ == "__main__":
    main()
