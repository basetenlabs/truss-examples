import requests


def download_model(model_url, destination_path):
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


def download_all_models():
    download_model(
        "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth",
        "../packages/StyleTTS2/Models/LJSpeech/epoch_2nd_00100.pth",
    )

    download_model(
        "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth",
        "../packages/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth",
    )

    download_model(
        "https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth",
        "../packages/StyleTTS2/Utils/ASR/epoch_00080.pth",
    )

    download_model(
        "https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7",
        "../packages/StyleTTS2/Utils/JDC/bst.t7",
    )

    download_model(
        "https://github.com/yl4579/StyleTTS2/raw/main/Utils/PLBERT/step_1000000.t7",
        "../packages/StyleTTS2/Utils/PLBERT/step_1000000.t7",
    )
