#!/usr/bin/env python3
"""
Test script for DeepSeek OCR Truss model
Tests OCR functionality with sample images
"""

import base64
import json
import os
from PIL import Image
import io


def test_with_url():
    """Test OCR with image URL"""
    print("=== Testing OCR with Image URL ===")

    # Test data
    test_data = {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
        "prompt": "Extract all text from this image. <image>",
    }

    print(f"Testing with URL: {test_data['image_url']}")
    print(f"Prompt: {test_data['prompt']}")

    # This would be the actual API call when deployed
    # For now, we'll simulate the response
    print("\nSimulated API call:")
    print("POST /predict")
    print(f"Body: {json.dumps(test_data, indent=2)}")

    return test_data


def test_with_base64():
    """Test OCR with base64 encoded image"""
    print("\n=== Testing OCR with Base64 Image ===")

    # Create a simple test image
    img = Image.new("RGB", (200, 100), color="white")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    test_data = {
        "image_base64": img_base64,
        "prompt": "Extract all text from this image. <image>",
    }

    print(f"Testing with base64 image (size: {len(img_base64)} chars)")
    print(f"Prompt: {test_data['prompt']}")

    print("\nSimulated API call:")
    print("POST /predict")
    print(
        f"Body: {json.dumps({**test_data, 'image_base64': f'[BASE64_DATA_{len(img_base64)}_CHARS]'}, indent=2)}"
    )

    return test_data


def test_with_local_image():
    """Test OCR with local image file"""
    print("\n=== Testing OCR with Local Image ===")

    # Create a test image
    img = Image.new("RGB", (300, 200), color="lightblue")

    # Save test image
    test_image_path = "test_image.png"
    img.save(test_image_path)

    # Read image as bytes
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()

    test_data = {
        "image": image_bytes,
        "prompt": "Extract all text from this image. <image>",
    }

    print(f"Testing with local image: {test_image_path}")
    print(f"Image size: {len(image_bytes)} bytes")
    print(f"Prompt: {test_data['prompt']}")

    print("\nSimulated API call:")
    print("POST /predict")
    print(
        f"Body: {json.dumps({**test_data, 'image': f'[BINARY_DATA_{len(image_bytes)}_BYTES]'}, indent=2)}"
    )

    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

    return test_data


def test_different_prompts():
    """Test OCR with different prompts"""
    print("\n=== Testing OCR with Different Prompts ===")

    prompts = [
        "Extract all text from this image. <image>",
        "Convert the document to markdown. <image>",
        "OCR this image. <image>",
        "Parse the figure. <image>",
        "Describe this image in detail. <image>",
        "Free OCR. <image>",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: {prompt}")

        test_data = {"image_url": "https://example.com/test.jpg", "prompt": prompt}

        print("   Simulated API call:")
        print("   POST /predict")
        print(f"   Body: {json.dumps(test_data, indent=4)}")


def test_error_cases():
    """Test error handling"""
    print("\n=== Testing Error Cases ===")

    error_cases = [
        {
            "name": "No image data",
            "data": {"prompt": "Extract text from image"},
            "expected_error": "No image data provided",
        },
        {
            "name": "Invalid URL",
            "data": {
                "image_url": "https://invalid-url.com/nonexistent.jpg",
                "prompt": "Extract text",
            },
            "expected_error": "Failed to load image",
        },
        {
            "name": "Invalid base64",
            "data": {"image_base64": "invalid_base64_data", "prompt": "Extract text"},
            "expected_error": "Failed to load image",
        },
    ]

    for case in error_cases:
        print(f"\n{case['name']}:")
        print(f"Expected error: {case['expected_error']}")
        print(f"Test data: {json.dumps(case['data'], indent=2)}")


def test_truss_predict():
    """Test using truss predict command"""
    print("\n=== Testing with Truss Predict Command ===")

    # Example truss predict command
    test_data = {
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
        "prompt": "Extract all text from this image. <image>",
    }

    print("Command to run:")
    print(f"truss predict -d '{json.dumps(test_data)}'")

    print("\nExpected output format:")
    expected_output = {
        "extracted_text": "Sample extracted text from the image",
        "image_size": [256, 256],
        "prompt_used": "Extract all text from this image. <image>",
        "model_name": "deepseek-ai/DeepSeek-OCR",
    }
    print(json.dumps(expected_output, indent=2))


def main():
    """Run all tests"""
    print("DeepSeek OCR Test Suite")
    print("=" * 50)

    # Test different input formats
    test_with_url()
    test_with_base64()
    test_with_local_image()

    # Test different prompts
    test_different_prompts()

    # Test error cases
    test_error_cases()

    # Test truss predict
    test_truss_predict()

    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo run actual tests after deployment:")
    print("1. Deploy: truss push")
    print(
        '2. Test: truss predict -d \'{"image_url": "https://example.com/image.jpg", "prompt": "Extract all text from this image. <image>"}\''
    )


if __name__ == "__main__":
    main()
