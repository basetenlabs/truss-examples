import baseten
import re
import cStringIO
from PIL import Image

import matplotlib.pyplot as plt
import textwrap, os

def display_images(
    images: [Image], 
    columns=5, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):

    if not images:
        print("No images to display.")
        return 

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)

        if hasattr(image, 'filename'):
            title=image.filename
            if title.endswith("/"): title = title[0:-1]
            title=os.path.basename(title)
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 

model = baseten.deployed_model_version_id('q41xddq')
results = model.predict({'prompt': 'A tree in a field under the night sky', 'use_refiner': True})

base64_results = results["data"]

images = []

for blob in base64_results:
    image_data = re.sub('^data:image/.+;base64,', '', blob).decode('base64')
    image = Image.open(cStringIO.StringIO(image_data))

    images.append(image)

display_images(images, 2, 50, 50)
