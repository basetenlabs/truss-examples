import base64
import os
import requests
import time
from pathlib import Path

api_url = # TODO: add your truss endpoint here
api_key =  os.environ.get("BASETEN_API_KEY")

voice_clone_file_name = "obama_8s.wav"
input_folder = Path("input")
output_folder = Path("output")
voice_clone_file_path = input_folder / voice_clone_file_name
output_audio_path = output_folder / f"output_{voice_clone_file_name}"
input_text = "Hey Oliver! How are you doing?"

with open(voice_clone_file_path, "rb") as f:
    audio_data = f.read()

audio_base64 = base64.b64encode(audio_data).decode("utf-8")

start_time = time.time()
predict_response = requests.post(
    api_url,
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "audio_prompt": audio_base64,
        "text": input_text,
    }
)
end_time = time.time()

predict_response_json = predict_response.json()
if "error" in predict_response_json:
    print(f"Error: {predict_response_json['error']}")
    exit(1)

print(f"Request latency: {end_time - start_time} seconds")

output_folder.mkdir(exist_ok=True)
with open(output_audio_path, "wb") as f:
    f.write(base64.b64decode(predict_response_json["audio"]))