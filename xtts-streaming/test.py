import asyncio
import websockets
import json
import base64
import wave
import os

def wav_to_base64(wav_path):
    """Convert a WAV file to base64 encoded string"""
    with open(wav_path, "rb") as wav_file:
        return base64.b64encode(wav_file.read()).decode('utf-8')

async def send_websocket_data():
    # Connection details
    uri = "wss://model-rwn1jgd3.api.baseten.co/v1/websocket"
    headers = {"Authorization": "Api-Key vVolDAU0.Mbynm8M7VGnaGqLbW9pwfWxFePNrGw8G"}
    
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        # For the TTS model, we send text instead of audio
        text_data = {
            "text": "Hello, this is a test of the text to speech websocket API.",
            "language": "en",
            "chunk_size": 20
        }
        
        # Send the text data as JSON
        await websocket.send(json.dumps(text_data))
        print(f"Sent text: {text_data['text']}")
        
        # Collect audio chunks
        audio_chunks = []
        
        # Process responses
        while True:
            try:
                response = await websocket.recv()
                
                # Try to parse as JSON
                try:
                    data = json.loads(response)
                    print(f"Received response: {data.get('type', 'unknown')}")
                    
                    if data.get("type") == "chunk":
                        # Decode and save the audio chunk
                        audio_chunk = base64.b64decode(data["data"])
                        audio_chunks.append(audio_chunk)
                        print("Saved audio chunk")
                    
                    elif data.get("type") == "complete":
                        print(f"Processing complete: {data.get('message')}")
                        break
                    
                    elif data.get("type") == "error":
                        print(f"Error: {data.get('message')}")
                        break
                
                except json.JSONDecodeError:
                    # Not JSON, print the first part
                    print(f"Received non-JSON response: {response[:50]}...")
                    break
                
            except Exception as e:
                print(f"Error receiving data: {str(e)}")
                break
        
        # Save the audio to a WAV file if we received chunks
        if audio_chunks:
            output_file = "tts_output.wav"
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # XTTS default sample rate
                wf.writeframes(b''.join(audio_chunks))
            
            print(f"Audio saved to {output_file}")
            print(f"Full path: {os.path.abspath(output_file)}")
        else:
            print("No audio data received")

async def test_multiple_concurrent_requests():
    """Test sending multiple concurrent requests to the TTS websocket API"""
    
    async def single_request(idx):
        """Handle a single request with unique text and output file"""
        output_file = f"tts_output_{idx}.wav"
        text = f"This is concurrent test number {idx}."
        
        try:
            # Connection details
            uri = "wss://model-rwn1jgd3.api.baseten.co/v1/websocket"
            headers = {"Authorization": "Api-Key vVolDAU0.Mbynm8M7VGnaGqLbW9pwfWxFePNrGw8G"}
            
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                # Send text data as JSON
                text_data = {
                    "text": text,
                    "language": "en",
                    "chunk_size": 20
                }
                
                await websocket.send(json.dumps(text_data))
                print(f"Request {idx}: Sent text: {text}")
                
                # Collect audio chunks
                audio_chunks = []
                
                # Process responses
                while True:
                    try:
                        response = await websocket.recv()
                        
                        # Try to parse as JSON
                        try:
                            data = json.loads(response)
                            
                            if data.get("type") == "chunk":
                                # Decode and save the audio chunk
                                audio_chunk = base64.b64decode(data["data"])
                                audio_chunks.append(audio_chunk)
                            
                            elif data.get("type") == "complete":
                                print(f"Request {idx}: Processing complete")
                                break
                            
                            elif data.get("type") == "error":
                                print(f"Request {idx}: Error: {data.get('message')}")
                                return False
                        
                        except json.JSONDecodeError:
                            print(f"Request {idx}: Received non-JSON response")
                            return False
                        
                    except Exception as e:
                        print(f"Request {idx}: Error receiving data: {str(e)}")
                        return False
                
                # Save the audio to a WAV file if we received chunks
                if audio_chunks:
                    with wave.open(output_file, 'wb') as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(24000)  # XTTS default sample rate
                        wf.writeframes(b''.join(audio_chunks))
                    
                    print(f"Request {idx}: Audio saved to {output_file}")
                    return True
                else:
                    print(f"Request {idx}: No audio data received")
                    return False
                
        except Exception as e:
            print(f"Request {idx}: Failed with exception: {str(e)}")
            return False
    
    num_requests = 4
    
    print(f"Starting {num_requests} concurrent requests...")
    results = await asyncio.gather(*[single_request(i+1) for i in range(num_requests)])
    
    successful = results.count(True)
    print(f"Completed {successful} out of {num_requests} requests successfully")
    return successful == num_requests

# Run the tests
if __name__ == "__main__":
    asyncio.run(send_websocket_data())
    print("\n--- Testing multiple concurrent requests ---\n")
    asyncio.run(test_multiple_concurrent_requests())