import asyncio
import aiohttp
import uuid
import time
import os

# Configuration
MODEL = "dq4rlnkw"
WS_URL = f"https://model-{MODEL}.api.baseten.co/environments/production/predict"
BASETEN_API_KEY = os.environ["BASETEN_API_KEY"]

base_request_payload = {
    "max_tokens": 4096,
    "voice": "tara",
    "stop_token_ids": [128258, 128009],
    "attach_wav_header": True
}

async def stream_via_ws(label: str, payload: dict) -> bytes:
    """Send one payload over WS, accumulate audio bytes, and log timings."""
    req_id = str(uuid.uuid4())
    payload = {**payload, "request_id": req_id}

    t0 = time.perf_counter()
    buf = bytearray()

    async with aiohttp.ClientSession() as session:
        # Note: pass your API key in headers
        async with session.ws_connect(
            WS_URL,
            headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"}
        ) as ws:
            # send the JSON payload
            await ws.send_json(payload)

            idx = 0
            async for msg in ws:
                elapsed_ms = (time.perf_counter() - t0) * 1_000
                # binary frames are raw audio chunks
                if msg.type == aiohttp.WSMsgType.BINARY:
                    chunk = msg.data
                    if idx == 0:
                        print(f"[{label}] ← first chunk ({len(chunk)} B) @ {elapsed_ms:.1f} ms")
                    buf.extend(chunk)
                    idx += 1

                # text frames include the final "DONE" (or any debug/info you choose)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == "DONE":
                        total_s = time.perf_counter() - t0
                        print(f"[{label}] ← DONE after {len(buf)} B in {total_s:.2f}s")
                        break
                    else:
                        # could log timing or metadata messages here
                        print(f"[{label}] ← text msg: {msg.data}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[{label}] WS error: {ws.exception()}")
                    break

    return bytes(buf)


async def run_ws_session(prompt: str, run_id: int):
    label = f"run{run_id}"
    payload = {**base_request_payload, "prompt": prompt}
    audio = await stream_via_ws(label, payload)
    # save the first few to disk for inspection
    if run_id < 3 and audio:
        fn = f"output_run{run_id}.wav"
        with open(fn, "wb") as f:
            f.write(audio)
        print(f"[{label}] ➔ saved {fn}")


async def main():
    prompts = ["Hello there...", "How is the weather in Tokyo? I heard it's been raining a lot lately."]  # your list
    tasks = [run_ws_session(p, i) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
