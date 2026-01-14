import requests
import base64
from PIL import Image
from io import BytesIO

# url = "https://model-4q9rpd9q.api.baseten.co/development/sync/v1/images/generations"
url = "https://model-7wl8d0e3.api.baseten.co/environments/production/predict"
headers = {
    "Authorization": "Api-Key YdjqboAn.PhwbrWIckmN40jV2614E3oe8syKYWqvT",
    "Content-Type": "application/json",
}
payload = {
    "prompt": "A Cat holding Logo With Bold Large Text: Baseten",
    "n": 1,
    "size": "1328x1328",
    "response_format": "b64_json",
    "num_inference_steps": 8,  # Try higher value
}

response = requests.post(url, json=payload, headers=headers)
print(f"Status: {response.status_code}")
result = response.json()

# Debug: show structure
print(f"Response type: {type(result)}")
print(f"Response keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

# Decode and save the image
try:
    data = result["data"]
    print(f"Data type: {type(data)}")

    # Handle different response formats
    if isinstance(data, str):
        # data is the base64 string directly
        image_b64 = data
    elif isinstance(data, list):
        item = data[0]
        if isinstance(item, dict):
            image_b64 = item.get("b64_json") or item.get("url")
        else:
            image_b64 = item
    else:
        image_b64 = str(data)

    print(f"Base64 length: {len(image_b64)}")
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes))
    image.save("output.png")
    print("Image saved to output.png!")
    image.show()  # Opens in default image viewer
except Exception as e:
    import traceback

    print(f"Error: {e}")
    traceback.print_exc()
