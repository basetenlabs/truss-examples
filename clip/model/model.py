# In this example, we create a Truss that uses [CLIP](https://openai.com/research/clip) to classify images,
# using some pre-defined labels. The input to this Truss will be an image, the output will be a classification.
#
# One of the major things to note about this example is that since the inputs are images, we need to have
# some mechanism for downloading the image. To accomplish this, we have the user pass a downloadable URL to
# the Truss, and in the Truss code, download the image. To do this efficiently, we will make use of the
# `preprocess` method in Truss.
#
# # Set up imports and constants
#
# For our CLIP Truss, we will be using the Hugging Face transformers library, as well as
# `pillow` for image processing.
import requests
from typing import Dict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# This is the CLIP model from Hugging Face that we will use for this example.
CHECKPOINT = "openai/clip-vit-base-patch32"

# # Define the Truss
#
# In the `load` method, we load in the pretrained CLIP model from the
# Hugging Face checkpoint specified above.
class Model:
    def __init__(self, **kwargs) -> None:
        self._processor = None
        self._model = None

    def load(self):
        """
        Loads the CLIP model and processor checkpoints.
        """
        self._model = CLIPModel.from_pretrained(CHECKPOINT)
        self._processor = CLIPProcessor.from_pretrained(CHECKPOINT)

    # In the `preprocess` method, we download the image from the url and preprocess it.
    # This method is a part of the Truss class, and is designed to be used for any logic
    # involving IO, like in this case, downloading an image.
    #
    # It is called before the predict method in a separate thread, and is not subject to the same
    # concurrency limits as the predict method, so can be called many times in parallel.
    # This makes it such that the predict method is not unnecessarily blocked on IO-bound
    # tasks, and helps improve the throughput of the Truss. See our [guide to concurrency](../guides/concurrency)
    # for more info.
    def preprocess(self, request: Dict) -> Dict:

        image = Image.open(requests.get(request.pop("url"), stream=True).raw)
        request["inputs"] = self._processor(
            text=["a photo of a cat", "a photo of a dog"], # Define preset labels to use
            images=image,
            return_tensors="pt",
            padding=True
        )
        return request

    # The `predict` method performs the actual inference, and outputs a probability associated
    # with each of the labels defined earlier.
    def predict(self, request: Dict) -> Dict:
        """
        This performs the actual classification. The predict method is subject to
        the predict concurrency constraints.
        """
        outputs = self._model(**request["inputs"])
        logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=1).tolist()