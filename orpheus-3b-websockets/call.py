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
MODEL_ID = "Orpheus-3b-Websockets"

# Orpheus TTS Websocket Endpoint
WS_URL = f"wss://model-yqv72oxq.api.baseten.co/deployment/qvvz10x/websocket"

VOICE = "tara"
MAX_TOKENS = 2000
BUFFER_SIZE = 10  # words / chunk
SAMPLE_RATE = 24000
WIDTH = pyaudio.paInt16
CHANNELS = 1


async def stream_tts(text: str, return_latency: bool = True, verbose: bool = True, num_loops: int = 1, play_audio: bool = False, stop_event: Optional[asyncio.Event] = None, started_event: Optional[asyncio.Event] = None, connection_id: str = "") -> Optional[dict]:
    """
    Stream TTS and optionally return timing measurements.
    
    Args:
        text: Text to convert to speech
        return_latency: If True, return timing measurements dict
        verbose: If True, print progress messages
        num_loops: Number of times to loop through the text (default: 1, use -1 for infinite)
        play_audio: If True, play the audio (default: False)
        stop_event: Optional event to signal when to stop (for infinite loops)
        started_event: Optional event to signal when connection has started
        connection_id: Optional ID for logging/debugging this specific connection
    
    Returns:
        Dict with timing measurements if return_latency is True, None otherwise:
        - client_ttfb_ms: Client-side time to first byte (includes network)
        - server_e2e_ms: Server-side processing time (text received → audio sent)
        - network_overhead_ms: Network latency overhead (client_ttfb - server_e2e)
        Each connection has its own isolated timing measurements.
    """
    if play_audio:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    else:
        pa = None
        stream = None

    # Each call to stream_tts creates its own isolated websocket connection
    # with separate timing variables - no cross-contamination between connections
    headers = {"Authorization": f"Api-Key {API_KEY}"}
    async with ClientSession(headers=headers) as sess:
        try:
            async with sess.ws_connect(WS_URL) as ws:
                # send metadata once
                await ws.send_json(
                    {
                        "voice": VOICE,
                        "max_tokens": MAX_TOKENS,
                        "buffer_size": BUFFER_SIZE,
                        "include_timing_info": return_latency,  # Only include timing when measuring latency
                    }
                )
                # start audio receiver
                # Each connection has its own isolated timing variables
                first_word_time = None
                first_byte_time = None
                server_e2e_latency = None
                
                async def receiver():
                    nonlocal first_byte_time, server_e2e_latency
                    async for msg in ws:
                        if msg.type == WSMsgType.TEXT:
                            data = msg.json()
                            audio_bytes = base64.b64decode(data['audio'])
                            
                            if first_byte_time is None:
                                first_byte_time = time.time()
                                # Capture server-side E2E latency from first chunk
                                server_e2e_latency = data.get('first_chunk_e2e_latency_ms')
                            
                            if stream:
                                stream.write(audio_bytes)
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                            return

                recv = asyncio.create_task(receiver())

                # send words (loop through text num_loops times, or infinitely if num_loops < 0)
                words = text.strip().split()
                infinite_loop = num_loops < 0
                loop_idx = 0
                
                while True:
                    for i, w in enumerate(words):
                        if loop_idx == 0 and i == 0:
                            first_word_time = time.time()
                        await ws.send_str(w)
                        
                        # Signal that connection has started after sending first word
                        if loop_idx == 0 and i == 0 and started_event:
                            started_event.set()
                        
                        # Check stop event during infinite loop
                        if infinite_loop and stop_event and stop_event.is_set():
                            break
                    
                    loop_idx += 1
                    
                    # Exit condition: break if we've done the specified number of loops or stop event is set
                    if not infinite_loop and loop_idx >= num_loops:
                        break
                    if infinite_loop and stop_event and stop_event.is_set():
                        break

                # signal end-of-text
                await ws.send_str("__END__")

                # wait until server closes
                await recv

        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            return None

    if stream:
        stream.stop_stream()
        stream.close()
    if pa:
        pa.terminate()
    
    # Calculate and return latency if requested
    # This timing is specific to THIS connection only (isolated per stream_tts call)
    if return_latency and first_word_time is not None and first_byte_time is not None:
        client_ttfb_ms = (first_byte_time - first_word_time) * 1000
        
        result = {
            'client_ttfb_ms': client_ttfb_ms,
            'server_e2e_ms': server_e2e_latency,
            'network_overhead_ms': None
        }
        
        # Calculate network overhead if server timing is available
        if server_e2e_latency is not None:
            result['network_overhead_ms'] = client_ttfb_ms - server_e2e_latency
        
        return result
    
    return None

async def benchmark_load_impact(
    text: str,
    load_levels: list[int]
) -> dict[int, dict]:
    """
    Benchmark how existing load affects TTFB for a new connection.
    Starts n active connections running indefinitely, then measures the
    TTFB for the n+1st connection.
    
    Args:
        text: Text to use for TTS
        load_levels: List of background connection counts to test (n values)
    
    Returns:
        Dictionary mapping load level (n) to dict with timing measurements:
        - client_ttfb_ms: Client-side time to first byte
        - server_e2e_ms: Server-side processing time
        - network_overhead_ms: Network latency overhead
    """
    results = {}
    
    for n in load_levels:
        print(f"\n{'='*60}")
        print(f"Load Level: {n} background connections")
        print(f"{'='*60}")
        
        if n > 0:
            # Create stop event and started events for background connections
            stop_event = asyncio.Event()
            started_events = [asyncio.Event() for _ in range(n)]
            
            # Start n background connections with infinite loops
            print(f"Starting {n} background connections...", end=" ", flush=True)
            background_tasks = [
                asyncio.create_task(stream_tts(
                    text, 
                    return_latency=False, 
                    verbose=False, 
                    num_loops=-1, 
                    stop_event=stop_event, 
                    started_event=started_events[i],
                    connection_id=f"bg-{i+1}"
                ))
                for i in range(n)
            ]
            
            # Wait for all connections to start
            await asyncio.gather(*[event.wait() for event in started_events])
            print("✓")
            
            # Brief stabilization period
            await asyncio.sleep(5)
        else:
            background_tasks = []
            stop_event = None
        
        # Measure the n+1st connection (this is the test connection with isolated timing)
        print(f"Testing n+1st connection...", end=" ", flush=True)
        test_result = await stream_tts(
            text, 
            return_latency=True, 
            verbose=False, 
            num_loops=1,
            connection_id="TEST"
        )
        
        if test_result is not None:
            results[n] = test_result
            client_ttfb = test_result['client_ttfb_ms']
            server_e2e = test_result.get('server_e2e_ms')
            network_overhead = test_result.get('network_overhead_ms')
            
            if server_e2e is not None and network_overhead is not None:
                print(f"TTFB: {client_ttfb:.2f}ms (server: {server_e2e:.2f}ms, network: {network_overhead:.2f}ms)")
            else:
                print(f"TTFB: {client_ttfb:.2f}ms (server timing unavailable)")
        else:
            results[n] = None
            print(f"FAILED")
        
        if background_tasks:
            # Signal background connections to stop gracefully
            print(f"Closing {len(background_tasks)} background connections...", end=" ", flush=True)
            stop_event.set()
            
            # Wait for all background connections to close properly
            await asyncio.gather(*background_tasks, return_exceptions=True)
            print("✓")
        
        await asyncio.sleep(1)
    
    return results


def plot_results(results: dict[int, dict], output_file: str = f"latency_chart_{time.strftime('%Y%m%d_%H%M%S')}.png"):
    """
    Plot latency results for different concurrency levels.

    Args:
        results: Dictionary mapping concurrency level to timing dict
        output_file: Path to save the chart
    """
    concurrency_levels = sorted(results.keys())
    
    # Extract timing data
    client_ttfb = []
    server_e2e = []
    network_overhead = []
    labels = []
    
    for level in concurrency_levels:
        if results[level] is not None:
            labels.append(str(level))
            client_ttfb.append(results[level]['client_ttfb_ms'])
            server_e2e.append(results[level].get('server_e2e_ms'))
            network_overhead.append(results[level].get('network_overhead_ms'))

    fig, ax = plt.subplots(figsize=(10, 6))

    if client_ttfb:
        # Plot client TTFB
        ax.plot(labels, client_ttfb, marker='o', linestyle='-', color='tab:blue', label='Client TTFB', linewidth=2)
        
        # Plot server E2E if available
        if any(s is not None for s in server_e2e):
            server_e2e_cleaned = [s if s is not None else np.nan for s in server_e2e]
            ax.plot(labels, server_e2e_cleaned, marker='s', linestyle='--', color='tab:green', label='Server Processing', linewidth=2)
        
        # Plot network overhead if available
        if any(n is not None for n in network_overhead):
            network_cleaned = [n if n is not None else np.nan for n in network_overhead]
            ax.plot(labels, network_cleaned, marker='^', linestyle=':', color='tab:orange', label='Network Overhead', linewidth=2)
        
        ax.set_xlabel('Background Connections', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('TTFB Impact by Load Level', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    # The text to send to each connection and loop through
    sample_text = (
        "Nothing beside remains. Round the decay of that colossal wreck, "
        "boundless and bare, The lone and level sands stretch far away."
    )
    
    async def main():
        load_levels = [i for i in range(1, 41, 5)]
        
        print("\n" + "="*60)
        print("LOAD IMPACT BENCHMARK")
        print(f"Model: {MODEL_ID} | Voice: {VOICE} | Max Tokens: {MAX_TOKENS} | Buffer Size: {BUFFER_SIZE}")
        print("="*60)
        print(f"Testing load levels: {load_levels}")
        print(f"Measuring TTFB impact of n background connections on n+1st connection")
        
        results = await benchmark_load_impact(sample_text, load_levels)
        
        import json
        filename = f"load_impact_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        chart_filename = f"load_impact_chart_{time.strftime('%Y%m%d_%H%M%S')}.png"
        
        # Prepare results with metadata
        output = {
            "metadata": {
                "model_id": MODEL_ID,
                "ws_url": WS_URL,
                "voice": VOICE,
                "max_tokens": MAX_TOKENS,
                "buffer_size": BUFFER_SIZE,
                "sample_rate": SAMPLE_RATE,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "sample_text": sample_text
            },
            "results": {str(k): v for k, v in results.items()}
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        
        plot_results(results, chart_filename)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        print(f"Results: {filename}")
        print(f"Chart:   {chart_filename}")
        print("="*60 + "\n")

    asyncio.run(main())
