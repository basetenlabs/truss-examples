import requests
import base64
import os


def image_file_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_base64_image(base64_string: str, output_path: str):
    """Decode a base64 string and save it as an image file."""
    image_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(image_data)
    print(f"Image saved to {output_path}")


def main():
    model_id = "abcd1234"
    baseten_api_key = os.environ["BASETEN_API_KEY"]
    image_path = "images/speaker-input.png"

    image_b64 = image_file_to_base64(image_path)

    payload = {
        "image": image_b64,
        "prompt": "sitting on a table, in front of a window with a beautiful rainy mountain forest landscape outside. Well built wooden table, wooden window, luxurious mountain cabin vibes.",
        "steps": 6,
        "strength": 1.0,
        "harmonize": False,
        "offset": 5,
        "height": 1024,
        "width": 1024,
        "delta": 0,
        "lora_name": "gen_back_7000_1024",
    }

    url = f"https://model-{model_id}.api.baseten.co/environments/production/predict"
    headers = {"Authorization": f"Api-Key {baseten_api_key}"}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("Generation successful.")

        generated_b64 = result.get("generated_image")
        if generated_b64:
            output_path = "images/generated-output.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(generated_b64))
            print(f"Image saved as {output_path}")
        else:
            print("No 'generated_image' key found in the response.")
    else:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
