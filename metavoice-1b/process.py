import base64
import sys

b64_audio = sys.stdin.read()

# sometimes the b64 data is surrounded by weird stuff and quotes; let's grab everything within the quotes
b64_audio = b64_audio.split('"')[1]

wav_file = open("output.wav", "wb")
decode_string = base64.b64decode(b64_audio)
wav_file.write(decode_string)