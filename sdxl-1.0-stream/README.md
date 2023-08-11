# Stable Diffusion XL Truss with Stream

Refer to the [README for SDXL](../stable-diffusion-xl-1.0/README.md) for full model deatils

This truss adds streaming functionality for latent outputs.

## Deploying
```
truss push --publish
```

# Example client code

You can use the following code from a Jupyter notebook to stream the images

```
%matplotlib inline
import matplotlib.pyplot as plt
from IPython import display
import base64
from io import BytesIO

from PIL import Image
import numpy as np
import requests

MODEL_VERSION_ID = "TAKEN FROM BASETEN"


response = requests.request(
    "post",
    f"https://app.baseten.co/model_versions/{MODEL_VERSION_ID}/predict",
    headers=headers,
    stream=True,
    json={"prompt": "Child running on the moon"}
)

consume = None
if response.headers.get("transfer-encoding") == "chunked":
    def decode_content():
        for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
            yield chunk.decode(response.encoding or "utf-8")

    consume = decode_content()

curr_img_data = ""
for base64_data in consume:
    curr_img_data += base64_data
    if "\n" in curr_img_data:
        curr_img_data, left_over = curr_img_data.split("\n")
        print(len(curr_img_data))
        decoded_info = base64.b64decode(curr_img_data)
        pil_img = Image.open(BytesIO(decoded_info))
        plt.imshow(np.asarray(pil_img))
        plt.show()
        curr_img_data = left_over

if curr_img_data:
    decoded_info = base64.b64decode(curr_img_data)
    pil_img = Image.open(BytesIO(decoded_info))
    plt.imshow(np.asarray(pil_img))
    plt.show()

```