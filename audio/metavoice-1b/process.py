import base64
import sys

b64_audio = sys.stdin.read()

# b64 data is surrounded by info messages and quotes if piped in from a truss command
b64_audio = b64_audio.split('"')[1]

wav_file = open("output.wav", "wb")
try:
    decode_string = base64.b64decode(b64_audio)
except:
    print("Response was not a valid base64 string. Exiting.")
    print(b64_audio)
    sys.exit(1)
wav_file.write(decode_string)
