#!/usr/bin/env python3
"""
Create test images for DeepSeek OCR testing
"""

from PIL import Image, ImageDraw, ImageFont


def create_text_image(text="Hello World", filename="test_text.png"):
    """Create an image with text for OCR testing"""
    # Create image
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fallback if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
            )
        except:
            font = ImageFont.load_default()

    # Draw text
    draw.text((50, 80), text, fill="black", font=font)

    # Save image
    img.save(filename)
    print(f"Created test image: {filename}")
    return filename


def create_document_image():
    """Create a document-like image for testing"""
    img = Image.new("RGB", (600, 800), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
            )
        except:
            font = ImageFont.load_default()

    # Document content
    lines = [
        "Sample Document",
        "",
        "This is a test document for OCR testing.",
        "It contains multiple lines of text.",
        "",
        "Features to test:",
        "- Text extraction",
        "- Line breaks",
        "- Multiple paragraphs",
        "",
        "Date: 2024-01-01",
        "Author: Test User",
    ]

    y_position = 50
    for line in lines:
        draw.text((50, y_position), line, fill="black", font=font)
        y_position += 30

    filename = "test_document.png"
    img.save(filename)
    print(f"Created document image: {filename}")
    return filename


def create_table_image():
    """Create a table-like image for testing"""
    img = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except:
            font = ImageFont.load_default()

    # Table content
    table_data = [
        ["Name", "Age", "City"],
        ["John", "25", "New York"],
        ["Jane", "30", "London"],
        ["Bob", "35", "Paris"],
    ]

    # Draw table
    x_start = 50
    y_start = 50
    cell_width = 120
    cell_height = 30

    for row_idx, row in enumerate(table_data):
        for col_idx, cell in enumerate(row):
            x = x_start + col_idx * cell_width
            y = y_start + row_idx * cell_height

            # Draw cell border
            draw.rectangle([x, y, x + cell_width, y + cell_height], outline="black")

            # Draw text
            draw.text((x + 10, y + 8), cell, fill="black", font=font)

    filename = "test_table.png"
    img.save(filename)
    print(f"Created table image: {filename}")
    return filename


def main():
    """Create all test images"""
    print("Creating test images for DeepSeek OCR...")

    # Create different types of test images
    create_text_image("Hello OCR World!")
    create_document_image()
    create_table_image()

    print("\nTest images created:")
    print("- test_text.png: Simple text image")
    print("- test_document.png: Document-like image")
    print("- test_table.png: Table-like image")

    print("\nTo test with these images:")
    print(
        'truss predict -d \'{"image_url": "path/to/image.png", "prompt": "Extract all text from this image. <image>"}\''
    )


if __name__ == "__main__":
    main()
