#!/usr/bin/env python3
"""
View test images for DeepSeek OCR
"""

from PIL import Image
import os


def view_image(filename):
    """Display image information and show it"""
    if not os.path.exists(filename):
        print(f"❌ Image {filename} not found")
        return

    try:
        # Open and display image info
        img = Image.open(filename)
        print(f"\n📸 {filename}")
        print(f"   Size: {img.size[0]} x {img.size[1]} pixels")
        print(f"   Mode: {img.mode}")
        print(f"   Format: {img.format}")

        # Show image (if running in terminal that supports it)
        try:
            img.show()
            print("   ✅ Image displayed")
        except Exception as e:
            print(f"   ⚠️  Could not display image: {e}")
            print(f"   💡 Try opening {filename} manually")

    except Exception as e:
        print(f"❌ Error opening {filename}: {e}")


def main():
    """View all test images"""
    print("DeepSeek OCR Test Images")
    print("=" * 40)

    # List of test images
    test_images = ["test_text.png", "test_document.png", "test_table.png"]

    # Check which images exist
    existing_images = [img for img in test_images if os.path.exists(img)]

    if not existing_images:
        print("❌ No test images found")
        print("💡 Run: python create_test_image.py")
        return

    print(f"Found {len(existing_images)} test images:")

    # View each image
    for img in existing_images:
        view_image(img)

    print("\n" + "=" * 40)
    print("Images ready for OCR testing!")
    print("\nTo test OCR with these images:")
    print("1. Deploy: truss push")
    print(
        '2. Test: truss predict -d \'{"image_url": "path/to/image.png", "prompt": "Extract all text from this image. <image>"}\''
    )


if __name__ == "__main__":
    main()
