import unittest
from typing import Dict

from model.model import Model


class TestFlorence2Large(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.model.load()

    def test_model_loading(self):
        """Test if the model and its components are loaded correctly."""
        self.assertIsNotNone(self.model.model, "Model not loaded")
        self.assertIsNotNone(self.model.processor, "Processor not loaded")

    def test_inference(self):
        """Test model inference with predefined inputs and expected outputs."""
        test_input = {
            "prompt": "<OD>",
            "image_url": "https://example.com/test_image.jpg",
        }
        expected_output_keys = ["result"]

        output = self.model.predict(test_input)

        self.assertIsInstance(output, Dict, "Output is not a dictionary")
        self.assertCountEqual(
            output.keys(),
            expected_output_keys,
            "Output keys do not match expected keys",
        )

    def test_output_handling(self):
        """Test if the model's output is correctly formatted."""
        test_input = {
            "prompt": "<OD>",
            "image_url": "https://example.com/test_image.jpg",
        }

        output = self.model.predict(test_input)
        result = output["result"]

        self.assertIsInstance(result, Dict, "Result is not a dictionary")
        self.assertTrue("objects" in result, "Objects key not found in result")
        self.assertTrue(
            "panoptic_segmentation" in result,
            "Panoptic segmentation key not found in result",
        )


if __name__ == "__main__":
    unittest.main()
