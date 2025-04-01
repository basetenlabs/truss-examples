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
    Sends a streaming request and measures the response time and size.
    Returns the label, total time taken, and total bytes received.
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
            return stream_label, 0, 0
        total_size = 0
        chunk_count = 0
        async for chunk in resp.content.iter_chunked(4096):
            chunk_count += 1
            total_size += len(chunk)
            now = time.time()
            execution_time_ms = (now - start_time) * 1000
            print(f"[{stream_label}] Received chunk {chunk_count} ({len(chunk)} bytes) after {execution_time_ms:.2f}ms")
        total_time = time.time() - start_time
        print(f"[{stream_label}] Completed receiving stream. Total size: {total_size} bytes in {total_time:.2f}s")
        return stream_label, total_time, total_size

async def main():
    async with aiohttp.ClientSession() as session:
        for concurrency in range(1, 17):
            print(f"\nStarting batch with concurrency {concurrency}")
            # Create tasks for this concurrency level
            tasks = [
                stream_to_buffer(session, f"Batch{concurrency}_Stream{m}", base_request_payload)
                for m in range(1, concurrency + 1)
            ]
            # Execute all tasks concurrently and wait for completion
            results = await asyncio.gather(*tasks)
            
            # Process and log results
            durations = []
            for label, duration, total_size in results:
                if duration > 0:
                    bitrate = total_size / duration
                else:
                    bitrate = 0
                print(f"[{label}] Duration: {duration:.2f}s, Size: {total_size} bytes, Bitrate: {bitrate:.2f} bytes/s")
                durations.append(duration)
            
            # Log batch summary
            if durations:
                avg_duration = sum(durations) / len(durations)
                print(f"Batch {concurrency} average duration: {avg_duration:.2f}s")
            
            # Wait 5 seconds before the next batch, except after the last one
            if concurrency < 16:
                print("Waiting 5 seconds before the next batch...")
                await asyncio.sleep(5)

if __name__ == '__main__':
    asyncio.run(main())