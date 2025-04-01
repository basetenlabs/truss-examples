import asyncio
import aiohttp
import uuid
import time
import string

BASETEN_HOST = "YOUR_PREDICT_URL"
BASETEN_API_KEY = "YOUR_API_KEY"

base_request_payload = {
    "prompt": "In today’s fast-paced world, finding balance between work and personal life is more important than ever. With the constant demands of technology, remote communication, and a culture that often praises overworking, it's easy to feel overwhelmed and burned out. Creating healthy boundaries, both physically and mentally, can lead to greater productivity and improved well-being. Taking regular breaks, prioritizing sleep, and making time for activities that bring joy are essential practices. Even small habits, like stepping outside for fresh air or disconnecting from screens for a few minutes, can have a big impact.",
    "max_tokens": 10000,
    "voice": "tara"
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
        headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"}
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
            print(f"[{stream_label}] Received chunk {chunk_count} ({len(chunk)} bytes) after {execution_time_ms:.2f}ms")
            buffer += chunk

        total_time = time.time() - start_time
        print(f"[{stream_label}] Completed receiving stream. Total size: {len(buffer)} bytes in {total_time:.2f}s")
        return buffer

async def main():
    stream_labels = [f"Stream{chr(65+i)}" for i in range(1)]
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_to_buffer(session, label, base_request_payload)
            for label in stream_labels
        ]
        results = await asyncio.gather(*tasks)
        
        for label, buffer in zip(stream_labels, results):
            filename = f"output_{label}.wav"
            with open(filename, "wb") as f:
                f.write(buffer)
            print(f"Saved {filename}")

if __name__ == '__main__':
    asyncio.run(main())
