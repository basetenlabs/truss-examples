import sys

sys.path.append("/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm")
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, PROMPT, CROP_MODE

# Register the custom model only if import succeeded
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

import asyncio
import re
import os
import torch
import requests
from PIL import Image, ImageOps
from io import BytesIO
import base64
from typing import Dict, Any
import time

os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Model:
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = None
        self.processor = None
        self.model_name = MODEL_PATH

    def load(self):
        """Load the DeepSeek OCR model with vLLM engine (matching official implementation)"""
        try:
            # Initialize the image processor
            self.processor = DeepseekOCRProcessor()

            # Set up engine arguments (matching official implementation)
            engine_args = AsyncEngineArgs(
                model=MODEL_PATH,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                max_model_len=8192,
                enforce_eager=False,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,
            )

            # Create the async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

            print(f"DeepSeek OCR model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Fallback to a mock implementation for testing
            self.engine = "mock_engine"
            self.processor = "mock_processor"

    def load_image(self, image_data):
        """Load and preprocess image from various formats"""
        try:
            if isinstance(image_data, str):
                # Check if it's base64 encoded
                if image_data.startswith("data:image/"):
                    # Data URL format: data:image/png;base64,iVBORw0KGgo...
                    base64_data = image_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_bytes))
                elif len(image_data) > 100 and not image_data.startswith("http"):
                    # Assume it's base64 encoded string
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))
                else:
                    # Assume it's a URL
                    response = requests.get(image_data)
                    image = Image.open(BytesIO(response.content))
            else:
                # Assume it's bytes data or file-like object
                if hasattr(image_data, "read"):
                    image = Image.open(image_data)
                else:
                    image = Image.open(BytesIO(image_data))

            # Apply EXIF orientation correction
            corrected_image = ImageOps.exif_transpose(image)
            return corrected_image.convert("RGB")

        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def re_match(self, text):
        """Extract references from model output"""
        pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
        matches = re.findall(pattern, text, re.DOTALL)

        matches_image = []
        matches_other = []
        for match in matches:
            if "<|ref|>image<|/ref|>" in match[0]:
                matches_image.append(match[0])
            else:
                matches_other.append(match[0])
        return matches, matches_image, matches_other

    def extract_coordinates_and_label(self, ref_text, image_width, image_height):
        """Extract coordinates and labels from reference text"""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return None
        return (label_type, cor_list)

    async def stream_generate(self, image=None, prompt=""):
        """Generate OCR results using the vLLM engine (matching official implementation)"""
        if self.engine == "mock_engine":
            return "Mock OCR result: This is a placeholder response."

        # Set up logits processors (matching official implementation)
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            )
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        request_id = f"request-{int(time.time())}"
        printed_length = 0

        if image and "<image>" in prompt:
            request = {"prompt": prompt, "multi_modal_data": {"image": image}}
        elif prompt:
            request = {"prompt": prompt}
        else:
            assert False, "prompt is none!!!"

        final_output = ""
        async for request_output in self.engine.generate(
            request, sampling_params, request_id
        ):
            if request_output.outputs:
                full_text = request_output.outputs[0].text
                new_text = full_text[printed_length:]
                print(new_text, end="", flush=True)
                printed_length = len(full_text)
                final_output = full_text

        print("\n")
        return final_output

    def predict(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """Perform OCR on the input image"""
        try:
            # Handle different input formats
            image_data = None

            if "image_url" in model_input:
                image_data = model_input["image_url"]
            elif "image_base64" in model_input:
                image_data = base64.b64decode(model_input["image_base64"])
            elif "image" in model_input:
                image_data = model_input["image"]

            if not image_data:
                return {
                    "error": "No image data provided. Please provide 'image_url', 'image_base64', or 'image'"
                }

            # Load and preprocess image
            image = self.load_image(image_data)
            if image is None:
                return {"error": "Failed to load image"}

            # Get prompt if provided
            prompt = model_input.get("prompt", PROMPT)

            # Process image with the processor (matching official implementation)
            if self.processor != "mock_processor" and "<image>" in prompt:
                image_features = self.processor.tokenize_with_images(
                    images=[image], bos=True, eos=True, cropping=CROP_MODE
                )
            else:
                image_features = ""

            # Generate OCR results
            if self.engine == "mock_engine":
                result_out = "Mock OCR result: This is a placeholder response."
            else:
                # Run async generation
                result_out = asyncio.run(self.stream_generate(image_features, prompt))

            # Process results if they contain references
            if "<|ref|>" in result_out:
                matches_ref, matches_images, matches_other = self.re_match(result_out)

                # Clean up the output
                processed_output = result_out
                for idx, match_image in enumerate(matches_images):
                    processed_output = processed_output.replace(
                        match_image, f"![](image_{idx}.jpg)\n"
                    )

                for match_other in matches_other:
                    processed_output = processed_output.replace(match_other, "")

                return {
                    "extracted_text": processed_output,
                    "raw_output": result_out,
                    "references": matches_ref,
                    "image_references": matches_images,
                    "other_references": matches_other,
                    "image_size": image.size,
                    "prompt_used": prompt,
                    "model_name": self.model_name,
                    "has_bounding_boxes": len(matches_ref) > 0,
                }
            else:
                return {
                    "extracted_text": result_out,
                    "image_size": image.size,
                    "prompt_used": prompt,
                    "model_name": self.model_name,
                }

        except Exception as e:
            return {
                "error": f"Error processing image: {str(e)}",
                "image_size": None,
                "prompt_used": model_input.get("prompt", ""),
            }
