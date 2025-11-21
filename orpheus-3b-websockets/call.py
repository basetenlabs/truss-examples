import asyncio
import base64
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from aiohttp import ClientSession, WSMsgType
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BASETEN_API_KEY")
MODEL_ID = ""

WS_URL = f"wss://model-yqv72oxq.api.baseten.co/deployment/qvvz10x/websocket"

VOICE = "tara"
MAX_TOKENS = 2000
BUFFER_SIZE = 10  # words / chunk
SAMPLE_RATE = 24000
WIDTH = pyaudio.paInt16
CHANNELS = 1


async def stream_tts(text: str, return_latency: bool = False, verbose: bool = True) -> Optional[float]:
    """
    Stream TTS and optionally return the server-side end-to-end latency.
    
    Args:
        text: Text to convert to speech
        return_latency: If True, return the server E2E latency in ms instead of playing audio
        verbose: If True, print progress messages
    
    Returns:
        Server-side end-to-end latency in milliseconds if return_latency is True, None otherwise.
        This measures time from first text received to first audio sent on the server,
        excluding network latency.
    """
    if not return_latency:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    else:
        pa = None
        stream = None

    headers = {"Authorization": f"Api-Key {API_KEY}"}
    if verbose:
        print(f"Connecting to WebSocket: {WS_URL}")
    async with ClientSession(headers=headers) as sess:
        try:
            async with sess.ws_connect(WS_URL) as ws:
                if verbose:
                    print("✅ WS connected")

                # send metadata once
                await ws.send_json(
                    {
                        "voice": VOICE,
                        "max_tokens": MAX_TOKENS,
                        "buffer_size": BUFFER_SIZE,
                    }
                )
                if verbose:
                    print("📤 metadata sent")
                # start audio receiver
                first_word_time = None
                first_byte_time = None
                first_token_gen_time = None
                first_e2e_latency = None
                token_gen_times = []
                latency_result = None
                
                async def receiver():
                    nonlocal first_byte_time, first_word_time, first_token_gen_time, first_e2e_latency, token_gen_times, latency_result
                    async for msg in ws:
                        
                        if msg.type == WSMsgType.TEXT:
                            data = msg.json()
                            audio_bytes = base64.b64decode(data['audio'])
                            
                            # Track end-to-end latency from server (first text received to first audio sent)
                            e2e_latency = data.get('first_chunk_e2e_latency_ms')
                            if e2e_latency is not None and first_e2e_latency is None:
                                first_e2e_latency = e2e_latency
                                latency_result = first_e2e_latency
                                if verbose:
                                    print(f"🎯 Server-side E2E latency (first text → first audio): {first_e2e_latency:.2f} ms")
                            
                            # Track token generation time from server (no network latency)
                            token_gen_time = data.get('token_gen_time')
                            if token_gen_time is not None:
                                token_gen_times.append(token_gen_time * 1000)  # Convert to ms
                                if first_token_gen_time is None:
                                    first_token_gen_time = token_gen_time * 1000
                                    if verbose:
                                        print(f"⏱️  First token generation time: {first_token_gen_time:.2f} ms")
                                        print(f"    (tokens processed: {data.get('tokens_processed', 'N/A')})")
                            
                            if first_byte_time is None:
                                first_byte_time = time.time()
                                if first_word_time is not None:
                                    total_latency = (first_byte_time - first_word_time) * 1000
                                    if verbose:
                                        print(f"📡 Total client latency (first word sent → first audio received): {total_latency:.2f} ms")
                                        if first_e2e_latency:
                                            print(f"    (network overhead: {total_latency - first_e2e_latency:.2f} ms)")
                            
                            if stream:
                                if verbose:
                                    print(f"⏯️  playing {len(audio_bytes)} bytes (gen_time: {token_gen_time*1000:.2f}ms)")
                                stream.write(audio_bytes)
                        elif msg.type == WSMsgType.BINARY:
                            print(f"⚠️  Received binary data, expected JSON: {msg}")
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                            if verbose:
                                print("🔒 server closed")
                                if token_gen_times:
                                    print(f"📊 Token generation stats:")
                                    print(f"    Mean: {np.mean(token_gen_times):.2f} ms")
                                    print(f"    Median: {np.median(token_gen_times):.2f} ms")
                                    print(f"    Min: {min(token_gen_times):.2f} ms")
                                    print(f"    Max: {max(token_gen_times):.2f} ms")
                            return

                recv = asyncio.create_task(receiver())

                # send words
                words = text.strip().split()
                for i, w in enumerate(words):
                    if i == 0:
                        first_word_time = time.time()
                    await ws.send_str(w)
                if verbose:
                    print("📤 words sent")

                # signal end-of-text
                await ws.send_str("__END__")
                if verbose:
                    print("📤 END sentinel sent — waiting for audio")

                # wait until server closes
                await recv

        except Exception as e:
            if verbose:
                print(f"❌ Connection error: {e}")
            return None

    if stream:
        stream.stop_stream()
        stream.close()
    if pa:
        pa.terminate()
    if verbose:
        print("🎉 done")
    
    return latency_result


async def benchmark_concurrency(
    text: str,
    concurrency_levels: list[int],
    runs_per_level: int = 3
) -> dict[int, list[float]]:
    """
    Benchmark server-side end-to-end latency for different concurrency levels.
    
    Args:
        text: Text to use for TTS
        concurrency_levels: List of concurrent connection counts to test
        runs_per_level: Number of runs per concurrency level
    
    Returns:
        Dictionary mapping concurrency level to list of server E2E latencies (ms).
        These timings measure time from first text received to first audio sent on the server,
        excluding network latency to isolate server performance.
    """
    results = {}
    
    for concurrency in concurrency_levels:
        print(f"\n{'='*60}")
        print(f"Testing concurrency level: {concurrency}")
        print(f"{'='*60}")
        latencies = []
        
        for run in range(runs_per_level):
            print(f"\nRun {run + 1}/{runs_per_level}")
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = [
                stream_tts(text, return_latency=True, verbose=True)
                for _ in range(concurrency)
            ]
            
            # Wait for all tasks to complete
            run_latencies = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None values
            valid_latencies = [
                lat for lat in run_latencies
                if isinstance(lat, (int, float)) and lat is not None
            ]
            
            if valid_latencies:
                avg_latency = np.mean(valid_latencies)
                latencies.extend(valid_latencies)
                print(f"  Average latency: {avg_latency:.2f} ms")
                print(f"  Min: {min(valid_latencies):.2f} ms, Max: {max(valid_latencies):.2f} ms")
            else:
                print(f"  ⚠️  No valid latencies recorded")
            
            elapsed = time.time() - start_time
            print(f"  Total time: {elapsed:.2f}s")
        
        results[concurrency] = latencies
        if latencies:
            print(f"\nSummary for concurrency {concurrency}:")
            print(f"  Mean: {np.mean(latencies):.2f} ms")
            print(f"  Median: {np.median(latencies):.2f} ms")
            print(f"  Std Dev: {np.std(latencies):.2f} ms")
            print(f"  Min: {min(latencies):.2f} ms")
            print(f"  Max: {max(latencies):.2f} ms")
    
    return results


def plot_results(results: dict[int, list[float]], output_file: str = f"latency_chart_{time.strftime('%Y%m%d_%H%M%S')}.png"):
    """
    Plot latency results for different concurrency levels.
    
    Args:
        results: Dictionary mapping concurrency level to list of latencies
        output_file: Path to save the chart
    """
    concurrency_levels = sorted(results.keys())
    
    # Prepare data for plotting - calculate percentiles
    p50_values = []
    p90_values = []
    p95_values = []
    
    for level in concurrency_levels:
        latencies = results[level]
        if latencies:
            p50_values.append(np.percentile(latencies, 50))
            p90_values.append(np.percentile(latencies, 90))
            p95_values.append(np.percentile(latencies, 95))
        else:
            p50_values.append(0)
            p90_values.append(0)
            p95_values.append(0)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Percentiles (p50, p90, p95)
    ax1.plot(
        concurrency_levels,
        p50_values,
        marker='o',
        linestyle='-',
        label='p50 (Median)',
        linewidth=2,
        markersize=4,
        color='#2E86AB'
    )
    ax1.plot(
        concurrency_levels,
        p90_values,
        marker='s',
        linestyle='--',
        label='p90',
        linewidth=2,
        markersize=4,
        color='#A23B72'
    )
    ax1.plot(
        concurrency_levels,
        p95_values,
        marker='^',
        linestyle='-.',
        label='p95',
        linewidth=2,
        markersize=4,
        color='#F18F01'
    )
    ax1.set_xlabel('Concurrent Connections', fontsize=12)
    ax1.set_ylabel('Server E2E Latency (ms)', fontsize=12)
    ax1.set_title('Server-Side E2E Latency Percentiles vs Concurrency', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(concurrency_levels)
    
    # Plot 2: Box plot
    data_to_plot = [results[level] for level in concurrency_levels if results[level]]
    labels = [str(level) for level in concurrency_levels if results[level]]
    
    if data_to_plot:
        bp = ax2.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True
        )
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Concurrent Connections', fontsize=12)
        ax2.set_ylabel('Server E2E Latency (ms)', fontsize=12)
        ax2.set_title('Server-Side E2E Latency Distribution by Concurrency', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Chart saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    import sys
    
    sample = (
        "Nothing beside remains. Round the decay of that colossal wreck, "
        "boundless and bare, The lone and level sands stretch far away."
    )
    
    async def main():
        if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
            # Run benchmark
            #concurrency_levels = [23, 24, 25, 26]
            concurrency_levels = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
            if len(sys.argv) > 2:
                concurrency_levels = [int(x) for x in sys.argv[2].split(',')]
            
            runs_per_level = 5
            if len(sys.argv) > 3:
                runs_per_level = int(sys.argv[3])
            
            print(f"Starting benchmark with concurrency levels: {concurrency_levels}")
            print(f"Runs per level: {runs_per_level}")
            
            results = await benchmark_concurrency(sample, concurrency_levels, runs_per_level)

            # INSERT_YOUR_CODE
            import json
            with open(f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                json.dump(results, f, indent=2)
            print("📝 Results saved to benchmark_results.json")
            plot_results(results)
        else:
            # Single stream mode
            await stream_tts(sample)

    asyncio.run(main())
