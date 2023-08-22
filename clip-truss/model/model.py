import requests
from typing import Dict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

CHECKPOINT = "openai/clip-vit-base-patch32"


class Model:
    """
    This is simple example of using CLIP to classify images.
    It outputs the probability of the image being a cat or a dog.
    """
    def __init__(self, **kwargs) -> None:
        self._processor = None
        self._model = None

    def load(self):
        """
        Loads the CLIP model and processor checkpoints.
        """
        self._model = CLIPModel.from_pretrained(CHECKPOINT)
        self._processor = CLIPProcessor.from_pretrained(CHECKPOINT)

    def preprocess(self, request: Dict) -> Dict:
        """"
        This method downloads the image from the url and preprocesses it.
        The preprocess method is used for any logic that involves IO, in this
        case downloading the image. It is called before the predict method
        in a separate thread and is not subject to the same concurrency
        limits as the predict method, so can be called many times in parallel.
        """
        image = Image.open(requests.get(request.pop("url"), stream=True).raw)
        request["inputs"] = self._processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=image,
            return_tensors="pt",
            padding=True
        )
        return request

    def predict(self, request: Dict) -> Dict:
        """
        This performs the actual classification. The predict method is subject to
        the predict concurrency constraints.
        """
        outputs = self._model(**request["inputs"])
        logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=1).tolist()
