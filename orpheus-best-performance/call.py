import asyncio
import aiohttp
import uuid
import time
import os

MODEL = "5wodgkgq"  # 7qkrmxd3
BASETEN_HOST = f"https://model-{MODEL}.api.baseten.co/environments/production/predict"

BASETEN_API_KEY = os.environ.get("BASETEN_API_KEY")

# Sample prompts of varying lengths
prompts = [
    # Short (1 sentence)
    "Hi, how can I help you today?",
    # Medium (2 sentences)
    "Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever",
]

base_request_payload = {
    "max_tokens": 10000,
    "voice": "tara",
    "stop_token_ids": [128258, 128009],
}


async def stream_to_buffer(session, stream_label, payload):
    """
    Sends a streaming request and accumulates the response in a buffer.
    """
    unique_id = str(uuid.uuid4())
    payload_with_id = payload.copy()
    payload_with_id["request_id"] = unique_id

    print(f"[{stream_label}] Starting request with request_id: {unique_id}")
    start_time = time.time()

    async with session.post(
        BASETEN_HOST,
        json=payload_with_id,
        headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
    ) as resp:
        if resp.status != 200:
            print(f"[{stream_label}] Error: Received status code {resp.status}")
            return b""
        buffer = b""
        chunk_count = 0
        async for chunk in resp.content.iter_chunked(4096):
            chunk_count += 1
            now = time.time()
            execution_time_ms = (now - start_time) * 1000
            print(
                f"[{stream_label}] Received chunk {chunk_count} ({len(chunk)} bytes) after {execution_time_ms:.2f}ms"
            )
            buffer += chunk

        total_time = time.time() - start_time
        print(
            f"[{stream_label}] Completed receiving stream. Total size: {len(buffer)} bytes in {total_time:.2f}s"
        )
        return buffer


async def run_session(session, prompt, prompt_type, run):
    payload = base_request_payload.copy()
    payload["prompt"] = prompt

    stream_label = f"{prompt_type}_run{run}"
    buffer = await stream_to_buffer(session, stream_label, payload)

    filename = f"output_{prompt_type}_run{run}.wav"
    with open(filename, "wb") as f:
        f.write(buffer)
    print(f"Saved {filename}")


async def main():
    async with aiohttp.ClientSession() as session:
        runs = []
        for i, prompt in enumerate(prompts):
            prompt_type = ["short", "medium", "long", "very_long", "super_long"][i]
            print(f"\nProcessing {prompt_type} prompt: {prompt[:50]}...")
            print(f"STOP TOKEN IDS: {base_request_payload['stop_token_ids']}")

            # Run each prompt twice
            for run in range(1, 4):
                print(f"\nRunning {prompt_type} prompt, run {run}...")
                runs.append(run_session(session, prompt, prompt_type, run))
        await asyncio.gather(*runs)
        print("All runs completed.")


if __name__ == "__main__":
    asyncio.run(main())
