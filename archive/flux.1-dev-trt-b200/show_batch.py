"""
Modified show.py script to handle batch responses with multiple images
Usage: curl ... | python show_batch.py
"""

import base64
import json
import os
import sys

resp = sys.stdin.read()
response_data = json.loads(resp)

# Check if the response contains multiple images or a single image
if "data" in response_data:
    data = response_data["data"]

    # If data is a list, we have multiple images
    if isinstance(data, list):
        print(f"Received {len(data)} images from batch request")
        for i, image_b64 in enumerate(data):
            img = base64.b64decode(image_b64)
            file_name = f"batch_image_{i + 1}_{image_b64[-10:].replace('/', '')}.jpeg"
            img_file = open(file_name, "wb")
            img_file.write(img)
            img_file.close()
            print(f"Saved image {i + 1} as {file_name}")
            os.system(f"open {file_name}")
    else:
        # Single image case
        img = base64.b64decode(data)
        file_name = f"single_image_{data[-10:].replace('/', '')}.jpeg"
        img_file = open(file_name, "wb")
        img_file.write(img)
        img_file.close()
        print(f"Saved single image as {file_name}")
        os.system(f"open {file_name}")
else:
    print("Error: No 'data' field found in response")
    print("Response:", response_data)
