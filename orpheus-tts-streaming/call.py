import time

import pyaudio
import requests

BASETEN_HOST = "<ENTER_PREDICT_URL>"
BASETEN_API_KEY = "<ENTER_API_KEY>"
FORMAT = pyaudio.paInt16  # Audio format (e.g., 16-bit PCM)
CHANNELS = 1  # Number of audio channels
RATE = 24000  # Sample rate

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream for audio playback
stream = p.open(
    format=p.get_format_from_width(2), channels=CHANNELS, rate=RATE, output=True
)

# Make a streaming HTTP request to the server
start_time = time.time()
resp = requests.post(
    BASETEN_HOST,
    headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
    json={
        "text": "Absolutely! Let's explore together. I'll help you with this. The concept of making inferences can be very useful when you encounter new words. Inference means using clues from the text to guess the meaning of a word or phrase. For example, if I say The library is open from 8 AM to 10 PM, and you see the word library, you might guess it’s a place where people read or borrow books because of its context. Now, let’s try this with a word from our text! Here’s one: “borrow.” What do you think “borrow” means based on how it's used in the sentence?",
        "max_tokens": 10000,
        "voice": "tara",
    },
    stream=True,
)

# Create a buffer to hold multiple chunks
buffer = b""
buffer_size_threshold = 2**2

# Stream and play the audio data as it's received
for chunk in resp.iter_content(chunk_size=4096):
    if chunk:
        now = time.time()
        execution_time_ms = (now - start_time) * 1000
        print(f"Received chunk after {execution_time_ms:.2f}ms: {len(buffer)}")
        buffer += chunk
        # stream.write(buffer)
        if len(buffer) >= buffer_size_threshold:
            print(f"Writing buffer of size: {len(buffer)}")
            stream.write(buffer)
            buffer = b""  # Clear the buffer
        # stream.write(chunk)

if buffer:
    print(f"Writing final buffer of size: {len(buffer)}")
    stream.write(buffer)

# Close and terminate the stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()
