#!/usr/bin/env python3
"""
Test script for the Flux model.

Usage:
    python test_model.py
    python test_model.py --save-image
    python test_model.py --prompt "your custom prompt here"
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# Add the model directory to the path so we can import the Model class
sys.path.insert(0, str(Path(__file__).parent / "model"))

from model import Model


def test_model_load():
    """Test model loading."""
    print("Testing model loading...")
    model = Model()
    
    try:
        model.load()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None


def test_model_prediction(model, prompt="a beautiful photograph of Mt. Fuji during cherry blossom, photorealistic, high quality"):
    """Test model prediction."""
    print(f"Testing model prediction with prompt: '{prompt}'")
    
    # Prepare input
    model_input = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 30,  # Use fewer steps for faster testing
        "guidance_scale": 3.5,
        "seed": 42,
        "batch_size": 1,
        "batch_count": 1
    }
    
    try:
        start_time = time.time()
        result = model.predict(model_input)
        end_time = time.time()
        
        if result["status"] == "success":
            print(f"✅ Prediction successful!")
            print(f"   Time: {result['time']:.2f} seconds")
            print(f"   Prompt: {result['prompt']}")
            print(f"   Dimensions: {result['width']}x{result['height']}")
            print(f"   Steps: {result['num_inference_steps']}")
            print(f"   Guidance Scale: {result['guidance_scale']}")
            print(f"   Seed: {result['seed']}")
            
            return result
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction failed with exception: {e}")
        return None


def save_image(result, filename="test_output.jpg"):
    """Save the generated image to a file."""
    if not result or result["status"] != "success":
        print("❌ No valid result to save")
        return
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(result["data"])
        
        # Save to file
        with open(filename, "wb") as f:
            f.write(image_data)
        
        print(f"✅ Image saved as '{filename}'")
        
        # Try to open the image (macOS)
        try:
            os.system(f"open {filename}")
            print(f"✅ Image opened in default viewer")
        except:
            print(f"📁 Image saved at: {os.path.abspath(filename)}")
            
    except Exception as e:
        print(f"❌ Failed to save image: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test the Flux model")
    parser.add_argument("--save-image", action="store_true", help="Save the generated image to a file")
    parser.add_argument("--prompt", type=str, default="a beautiful photograph of Mt. Fuji during cherry blossom, photorealistic, high quality", 
                       help="Custom prompt for testing")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    
    args = parser.parse_args()
    
    print("🚀 Starting Flux model test...")
    print("=" * 50)
    
    # Test model loading
    model = test_model_load()
    if not model:
        print("❌ Cannot proceed without a loaded model")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Test model prediction
    model_input = {
        "prompt": args.prompt,
        "negative_prompt": "blurry, low quality, distorted",
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": 3.5,
        "seed": 42,
        "batch_size": 1,
        "batch_count": 1
    }
    
    result = test_model_prediction(model, args.prompt)
    
    if result and args.save_image:
        print("\n" + "=" * 50)
        save_image(result)
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")


if __name__ == "__main__":
    main() 