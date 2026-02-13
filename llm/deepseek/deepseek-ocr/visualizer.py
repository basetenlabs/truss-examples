#!/usr/bin/env python3
"""
DeepSeek OCR Visualization Module
Handles bounding box drawing and geometric visualization
"""

import base64
import io
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional imports for geometric visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print(
        "Warning: matplotlib not available. Geometric visualization will be disabled."
    )


class DeepSeekOCRVisualizer:
    """Visualization utilities for DeepSeek OCR results"""

    def __init__(self):
        self.font = ImageFont.load_default()

    def extract_coordinates_and_label(self, ref_text, image_width, image_height):
        """Extract coordinates and labels from reference text"""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return None
        return (label_type, cor_list)

    def draw_bounding_boxes(self, image, refs):
        """Draw bounding boxes around detected text regions"""
        image_width, image_height = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)

        img_idx = 0
        label_positions = []  # Track label positions to avoid overlap

        for i, ref in enumerate(refs):
            try:
                result = self.extract_coordinates_and_label(
                    ref, image_width, image_height
                )
                if result:
                    label_type, points_list = result

                    color = (
                        np.random.randint(0, 200),
                        np.random.randint(0, 200),
                        np.random.randint(0, 255),
                    )
                    color_a = color + (20,)

                    for points in points_list:
                        x1, y1, x2, y2 = points

                        # Check if coordinates are already in pixel format (larger than 999)
                        # or normalized format (0-999)
                        if max(x1, y1, x2, y2) > 999:
                            # Already in pixel coordinates, use directly
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        else:
                            # Normalized coordinates, convert to pixels
                            x1 = int(x1 / 999 * image_width)
                            y1 = int(y1 / 999 * image_height)
                            x2 = int(x2 / 999 * image_width)
                            y2 = int(y2 / 999 * image_height)

                        if label_type == "image":
                            try:
                                cropped = image.crop((x1, y1, x2, y2))
                                # Save cropped image (would need output path in real implementation)
                                # cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")
                                img_idx += 1
                            except Exception as e:
                                print(f"Error cropping image: {e}")
                                pass

                        try:
                            if label_type == "title":
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                                draw2.rectangle(
                                    [x1, y1, x2, y2],
                                    fill=color_a,
                                    outline=(0, 0, 0, 0),
                                    width=1,
                                )
                            else:
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                draw2.rectangle(
                                    [x1, y1, x2, y2],
                                    fill=color_a,
                                    outline=(0, 0, 0, 0),
                                    width=1,
                                )

                            # Position label with smart spacing to avoid overlap
                            text_x = x1
                            base_y = max(0, y1 - 15)

                            # Check for overlap with existing labels and adjust position
                            text_bbox = draw.textbbox(
                                (0, 0), label_type, font=self.font
                            )
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]

                            # Find a non-overlapping position
                            text_y = base_y
                            while True:
                                overlaps = False
                                for (
                                    existing_x,
                                    existing_y,
                                    existing_w,
                                    existing_h,
                                ) in label_positions:
                                    if (
                                        text_x < existing_x + existing_w
                                        and text_x + text_width > existing_x
                                        and text_y < existing_y + existing_h
                                        and text_y + text_height > existing_y
                                    ):
                                        overlaps = True
                                        break

                                if not overlaps:
                                    break
                                text_y -= 20  # Move up by 20 pixels

                            # Store this label position
                            label_positions.append(
                                (text_x, text_y, text_width, text_height)
                            )

                            draw.rectangle(
                                [
                                    text_x,
                                    text_y,
                                    text_x + text_width,
                                    text_y + text_height,
                                ],
                                fill=(255, 255, 255, 30),
                            )

                            draw.text(
                                (text_x, text_y), label_type, font=self.font, fill=color
                            )
                        except Exception as e:
                            print(f"Error drawing text: {e}")
                            pass
            except Exception as e:
                print(f"Error processing reference: {e}")
                continue

        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw

    def process_image_with_refs(self, image, ref_texts):
        """Process image with reference texts and draw bounding boxes"""
        result_image = self.draw_bounding_boxes(image, ref_texts)
        return result_image

    def create_geometric_visualization(self, outputs):
        """Create geometric visualization for line and circle elements"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping geometric visualization")
            return None

        try:
            if "line_type" not in outputs:
                return None

            lines = eval(outputs)["Line"]["line"]
            line_type = eval(outputs)["Line"]["line_type"]
            endpoints = eval(outputs)["Line"]["line_endpoint"]

            fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            for idx, line in enumerate(lines):
                try:
                    p0 = eval(line.split(" -- ")[0])
                    p1 = eval(line.split(" -- ")[-1])

                    if line_type[idx] == "--":
                        ax.plot(
                            [p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k"
                        )
                    else:
                        ax.plot(
                            [p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k"
                        )

                    ax.scatter(p0[0], p0[1], s=5, color="k")
                    ax.scatter(p1[0], p1[1], s=5, color="k")
                except:
                    pass

            for endpoint in endpoints:
                try:
                    label = endpoint.split(": ")[0]
                    (x, y) = eval(endpoint.split(": ")[1])
                    ax.annotate(
                        label,
                        (x, y),
                        xytext=(1, 1),
                        textcoords="offset points",
                        fontsize=5,
                        fontweight="light",
                    )
                except:
                    pass

            try:
                if "Circle" in eval(outputs).keys():
                    circle_centers = eval(outputs)["Circle"]["circle_center"]
                    radius = eval(outputs)["Circle"]["radius"]

                    for center, r in zip(circle_centers, radius):
                        center = eval(center.split(": ")[1])
                        circle = Circle(
                            center,
                            radius=r,
                            fill=False,
                            edgecolor="black",
                            linewidth=0.8,
                        )
                        ax.add_patch(circle)
            except:
                pass

            # Convert plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            plt.close()
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()

            return plot_base64

        except Exception as e:
            print(f"Error creating geometric visualization: {e}")
            return None

    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()

    def create_visualization(self, image, ocr_result):
        """Create complete visualization from OCR result"""
        try:
            raw_output = ocr_result.get("raw_output", "")

            # Try to extract references in <|ref|> format first
            pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
            matches = re.findall(pattern, raw_output, re.DOTALL)

            # If no <|ref|> format found, try bracket format: text[[x1, y1, x2, y2]]
            if not matches:
                bracket_pattern = r"(\w+)\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]"
                bracket_matches = re.findall(bracket_pattern, raw_output)
                if bracket_matches:
                    # Convert bracket format to ref format for processing
                    # The ref format expects: (full_match, label_type, coords_string)
                    # where coords_string when eval'd becomes a list of coordinate lists
                    matches = []
                    for match in bracket_matches:
                        label_type = match[0]
                        x1, y1, x2, y2 = (
                            int(match[1]),
                            int(match[2]),
                            int(match[3]),
                            int(match[4]),
                        )
                        # Wrap coordinates in a list to match expected format: [[x1, y1, x2, y2]]
                        coords = f"[[{x1}, {y1}, {x2}, {y2}]]"
                        # Create a tuple matching the ref format: (full_match, label_type, coords)
                        matches.append(
                            (
                                f"{label_type}[[{x1}, {y1}, {x2}, {y2}]]",
                                label_type,
                                coords,
                            )
                        )

            if not matches:
                return None

            # Create bounding box visualization
            visualized_image = self.process_image_with_refs(image, matches)

            # Convert to base64
            visualization_base64 = self.image_to_base64(visualized_image)

            # Create geometric visualization if applicable
            geometric_visualization = self.create_geometric_visualization(
                ocr_result.get("raw_output", "")
            )

            return {
                "visualization": visualization_base64,
                "geometric_visualization": geometric_visualization,
                "has_bounding_boxes": len(matches) > 0,
                "has_geometric_elements": geometric_visualization is not None,
            }

        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None


def visualize_ocr_result(image_path, ocr_result, output_path=None):
    """Standalone function to visualize OCR results"""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Create visualizer
        visualizer = DeepSeekOCRVisualizer()

        # Create visualization
        viz_result = visualizer.create_visualization(image, ocr_result)

        if viz_result and output_path:
            # Save visualization
            viz_image = Image.open(
                io.BytesIO(base64.b64decode(viz_result["visualization"]))
            )
            viz_image.save(output_path)
            print(f"Visualization saved to: {output_path}")

        return viz_result

    except Exception as e:
        print(f"Error in visualize_ocr_result: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("DeepSeek OCR Visualizer")
    print("Usage:")
    print("from visualizer import DeepSeekOCRVisualizer")
    print("visualizer = DeepSeekOCRVisualizer()")
    print("result = visualizer.create_visualization(image, ocr_result)")
