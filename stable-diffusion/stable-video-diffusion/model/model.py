import os
from typing import Dict

import requests
from model.helper import b64_to_pil, mp4_to_base64, sample

original_working_directory = os.getcwd()

DEFAULT_NUM_FRAMES = 14
DEFAULT_NUM_STEPS = 25
DEFAULT_FPS = 6
DEFAULT_DECODING_T = 5


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        self._model = None

    def download_model(self, model_url, destination_path):
        print(f"Downloading model {model_url} ...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            print("download response: ", response)

            # Open the destination file and write the content in chunks
            print("opening: ", destination_path)
            with open(destination_path, "wb") as file:
                print("writing chunks...")
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)

                print("done writing chunks!!!!")

            print(f"Downloaded file to: {destination_path}")
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")

    def load(self):
        os.mkdir("checkpoints")
        os.mkdir("clip_ckpt")
        self.download_model(
            "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors?download=true",
            "checkpoints/svd.safetensors",
        )
        self.download_model(
            "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            "clip_ckpt/model.pt",
        )

    def predict(self, model_input: Dict) -> Dict:
        image_b64 = model_input.get("image")
        num_frames = int(model_input.get("num_frames", DEFAULT_NUM_FRAMES))
        num_steps = int(model_input.get("num_steps", DEFAULT_NUM_STEPS))
        fps_id = int(model_input.get("fps", DEFAULT_FPS))
        frames_decoded_per_second = int(
            model_input.get("decoding_t", DEFAULT_DECODING_T)
        )

        if frames_decoded_per_second > 10:
            return {
                "output": "GPU does not have enough memory to decode more than 10 frames per second"
            }

        pil_image = b64_to_pil(image_b64)
        pil_image.save("input_image.png")

        sample(
            input_path=os.path.join(str(os.getcwd()), "input_image.png"),
            num_frames=num_frames,
            num_steps=num_steps,
            fps_id=fps_id,
            decoding_t=frames_decoded_per_second,
        )

        if os.path.isdir("outputs/simple_video_sample/svd/"):
            files_array = os.listdir("outputs/simple_video_sample/svd/")
            if len(files_array) > 0:
                output_file = files_array[0]
                output_base64 = mp4_to_base64(
                    os.path.join("outputs/simple_video_sample/svd/", output_file)
                )
                os.remove(os.path.join("outputs/simple_video_sample/svd/", output_file))
                os.remove("input_image.png")
                return {"output": output_base64}
        else:
            return {"output": "unsuccessful"}
