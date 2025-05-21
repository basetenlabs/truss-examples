import requests
import base64
import os

def image_file_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_base64_image(b64_string: str, output_path: str):
    """Decode a base64 string and write it to disk as an image."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_string))
    print(f"✅  Image saved to {output_path}")


def main():
    # ---------- Configuration ----------
    model_id = ""
    baseten_api_key = os.environ["BASETEN_API_KEY"]  # ← hard-coded
    image_path = "images/camera.png"  # local test image
    prompt_text = "a man holding a camera facing the objective."


    # ---------- Encode input image ----------
    image_b64 = image_file_to_base64(image_path)

    # ---------- Baseten payload ----------
    payload = {
        "image": image_b64,
        "prompt": prompt_text,
        "steps": 10,
        "strength_sub": 1.2,
        "strength_spat": 1.2,
        "size": 1024,
        "lora_name": "zen2con_1024_10000",
    }

    url = f"https://model-{model_id}.api.baseten.co/environments/production/predict"
    headers = {"Authorization": f"Api-Key {baseten_api_key}"}

    # ---------- Call Baseten ----------
    print("🚀  Sending request to Baseten…")
    response = requests.post(url, headers=headers, json=payload)

    # ---------- Handle response ----------
    if response.status_code == 200:
        result = response.json()
        images = result.get("data", {}).get("images")
        if not images:
            print("⚠️  No 'images' key found in response:")
            print(result)
            return
        generated_b64 = images[-1]  # take the last / best image
        save_base64_image(generated_b64, "baseten_output.png")
    else:
        print(f"❌  Error: HTTP {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()