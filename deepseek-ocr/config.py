"""
Configuration file for DeepSeek OCR Truss deployment
"""

# Model configuration
MODEL_PATH = "deepseek-ai/DeepSeek-OCR"

# Input/Output paths (for local testing)
INPUT_PATH = "/tmp/input_image.jpg"
OUTPUT_PATH = "/tmp/output"

# Default prompt for OCR
PROMPT = "Extract all text from this image. <image>"

# Image processing configuration
CROP_MODE = False  # Set to True for cropping, False for padding

# Model parameters
MAX_MODEL_LEN = 8192
BLOCK_SIZE = 256
GPU_MEMORY_UTILIZATION = 0.75
TENSOR_PARALLEL_SIZE = 1

# Logits processor parameters
NGRAM_SIZE = 30
WINDOW_SIZE = 90
WHITELIST_TOKEN_IDS = {128821, 128822}  # <td>, </td>
