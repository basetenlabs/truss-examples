#!/usr/bin/env python3
"""
Test DeepSeek OCR with test_document.png and visualize results
"""

import requests
import base64
import io
import os
from PIL import Image
from visualizer import DeepSeekOCRVisualizer

# Baseten API configuration
API_KEY = os.getenv("BASETEN_API_KEY")  # previously committed key has been invalidated
print(API_KEY)
ENDPOINT_URL = "https://model-7wlgx8eq.api.baseten.co/environments/production/predict"


def load_test_document():
    """Load the test document image and convert to base64"""
    try:
        # Load the test document image
        image = Image.open("Bad-Handwriting.png")
        print(f"Loaded test document: {image.size[0]}x{image.size[1]} pixels")

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        print(f"Converted to base64: {len(img_base64)} characters")
        return image, img_base64

    except Exception as e:
        print(f"Error loading test document: {e}")
        return None, None


def test_ocr_with_document():
    """Test OCR with the test document"""

    # Load test document
    original_image, img_base64 = load_test_document()
    if not original_image or not img_base64:
        return None

    # Test prompts
    prompts = [
        "<image>\n<|grounding|>Convert the document to markdown.",
        "<image>\n<|grounding|>OCR this image.",
        "<image>\nFree OCR.",
        "<image>\nParse the figure.",
        "<image>\nDescribe this image in detail.",
    ]

    client = requests.Session()
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Testing prompt: {prompt}")
        print("-" * 60)

        test_data = {"prompt": prompt, "image_base64": img_base64}

        try:
            resp = client.post(
                ENDPOINT_URL,
                headers={"Authorization": f"Api-Key {API_KEY}"},
                json=test_data,
                timeout=120,  # 2 minute timeout for OCR processing
            )

            print(f"Status Code: {resp.status_code}")

            if resp.status_code == 200:
                result = resp.json()
                print("✅ Success!")

                # Print extracted text
                extracted_text = result.get("extracted_text", "No text extracted")
                print(f"Extracted text:\n{extracted_text}")

                # Print raw output for debugging
                raw_output = result.get("raw_output", "")
                if raw_output:
                    print(f"Raw output:\n{raw_output[:500]}...")

                # Check if we have bounding boxes
                has_boxes = result.get("has_bounding_boxes", False)
                print(f"Has bounding boxes: {has_boxes}")

                # Store result for visualization
                results.append(
                    {
                        "prompt": prompt,
                        "result": result,
                        "original_image": original_image,
                    }
                )

            else:
                print(f"❌ Error: {resp.status_code}")
                print(f"Response: {resp.text}")

        except Exception as e:
            print(f"❌ Error: {e}")

    return results


def visualize_results(results):
    """Visualize OCR results with bounding boxes"""

    if not results:
        print("No results to visualize")
        return

    print("\n" + "=" * 80)
    print("VISUALIZING OCR RESULTS")
    print("=" * 80)

    visualizer = DeepSeekOCRVisualizer()

    for i, result_data in enumerate(results, 1):
        prompt = result_data["prompt"]
        ocr_result = result_data["result"]
        original_image = result_data["original_image"]

        print(f"\n{i}. Visualizing: {prompt}")
        print("-" * 60)

        # Check if we have raw output with references
        raw_output = ocr_result.get("raw_output", "")
        if "<|ref|>" in raw_output:
            print("Found reference tokens - creating visualization...")

            try:
                # Create visualization
                viz_result = visualizer.create_visualization(original_image, ocr_result)

                if viz_result:
                    print("✅ Visualization created successfully!")
                    print(f"Has bounding boxes: {viz_result['has_bounding_boxes']}")
                    print(
                        f"Has geometric elements: {viz_result['has_geometric_elements']}"
                    )

                    # Save visualization
                    viz_filename = f"visualization_{i}.png"
                    viz_image = Image.open(
                        io.BytesIO(base64.b64decode(viz_result["visualization"]))
                    )
                    viz_image.save(viz_filename)
                    print(f"Visualization saved to: {viz_filename}")

                    # Save geometric visualization if available
                    if viz_result["geometric_visualization"]:
                        geo_filename = f"geometric_{i}.png"
                        geo_image = Image.open(
                            io.BytesIO(
                                base64.b64decode(viz_result["geometric_visualization"])
                            )
                        )
                        geo_image.save(geo_filename)
                        print(f"Geometric visualization saved to: {geo_filename}")

                else:
                    print("❌ Failed to create visualization")

            except Exception as e:
                print(f"❌ Error creating visualization: {e}")
        else:
            print("No reference tokens found - skipping visualization")


def main():
    """Main test function"""
    print("DeepSeek OCR Document Test")
    print("=" * 80)
    print("Testing with Bad-Handwriting.png")
    print("=" * 80)

    # Test OCR with document
    results = test_ocr_with_document()

    if results:
        # Visualize results
        visualize_results(results)

        print("\n" + "=" * 80)
        print("TEST COMPLETED!")
        print("=" * 80)
        print("Check the generated visualization files:")
        print("- visualization_*.png - OCR results with bounding boxes")
        print("- geometric_*.png - Geometric elements (if any)")
    else:
        print("❌ No results to process")


if __name__ == "__main__":
    main()
