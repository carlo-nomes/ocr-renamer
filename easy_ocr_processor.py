import os
import sys
import json
import logging
import argparse
import easyocr
from PIL import Image, ImageDraw, ImageFont

# Constants for configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
OCR_SUCCESS = "OCR completed."
DEFAULT_BOX_COLOR = "blue"  # Color of bounding boxes
LINE_THICKNESS = 5  # Thickness of bounding box lines
LANGUAGES = ["en", "nl"]  # Languages for OCR (English and Dutch)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

# Suppress excessive PIL debug logs
logging.getLogger("PIL").setLevel(logging.INFO)


def load_images(image_dir: str) -> list:
    """Load images from the specified directory."""
    image_paths = sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(".png") or fname.endswith(".jpg")
        ],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
    )
    return image_paths


def ocr_to_boxes(ocr_result) -> list:
    """Serialize the OCR result to ensure all data is JSON serializable."""
    serialized_result = []
    for box in ocr_result:
        box_data = {
            "box": [[int(pt[0]), int(pt[1])] for pt in box[0]],
            "text": box[1],
            "confidence": float(box[2]),
        }
        serialized_result.append(box_data)
    return serialized_result


def perform_ocr(image_path: str, reader: easyocr.Reader) -> list:
    """Perform OCR on the given image using EasyOCR."""
    result = reader.readtext(image_path)
    boxes = ocr_to_boxes(result)
    return boxes


def save_metadata(
    original_image_path: str, boxed_image_path: str, boxes: list, metadata_dir: str
):
    """Save the OCR metadata (boxes) to a JSON file."""
    metadata = {
        "original_image": original_image_path,
        "boxed_image": boxed_image_path,
        "boxes": boxes,
    }

    # Save metadata to a JSON file
    metadata_filename = (
        f"{os.path.splitext(os.path.basename(original_image_path))[0]}.json"
    )
    metadata_path = os.path.join(metadata_dir, metadata_filename)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Metadata saved to: {metadata_path}")


def convert_box_to_pillow(image: Image, box: list) -> tuple:
    """Convert EasyOCR's box format to Pillow's box format."""
    x0, y0 = box[0]
    x1, y1 = box[2]
    return x0, y0, x1, y1


def draw_bounding_boxes(
    image: Image, boxes: list, color: str = DEFAULT_BOX_COLOR, with_index: bool = False
) -> Image:
    """Draw bounding boxes on the image and save it."""
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = convert_box_to_pillow(image, box["box"])
        text = box["text"]
        logger.info(f"Drawing box: ({x0}, {y0}), ({x1}, {y1}) - {text}")

        # Calculate the box height to determine a relative font size
        box_height = abs(y1 - y0)

        # Set font size relative to the box height
        font_size = int(box_height * 0.6)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Draw bounding box
        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=LINE_THICKNESS)

        # Draw the recognized text just below the bounding box
        draw.text((x0, y1), text, font=font, fill=color)

        if with_index:
            draw.text((x0, y1 + 10), f"Index: {index}", font=font, fill=color)

    return image


def process_single_image(
    image_path: str, output_image_dir: str, metadata_dir: str, reader: easyocr.Reader
):
    """Process a single image: Perform OCR, draw bounding boxes, and save the results."""
    try:
        # Open the image
        img = Image.open(image_path)
        logger.info(f"Processing image format: {img.format}")

        # Perform OCR
        boxes = perform_ocr(image_path, reader)
        logger.info(f"Found {len(boxes)} text boxes in image {image_path}")

        # Draw bounding boxes on the image
        boxed_image = draw_bounding_boxes(img, boxes, color="red", with_index=True)

        # Save the image with bounding boxes in the output image directory
        output_image_path = os.path.join(
            output_image_dir, f"{os.path.basename(image_path)}"
        )
        boxed_image.save(output_image_path)
        logger.info(f"Boxed image saved to: {output_image_path}")

        # Save metadata (OCR results in JSON in metadata directory)
        save_metadata(image_path, output_image_path, boxes, metadata_dir)

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")

    finally:
        img.close()  # Ensure image is closed after processing


def process_images(
    image_paths: list, output_image_dir: str, metadata_dir: str, reader: easyocr.Reader
):
    """Process all images in the image_paths list."""
    for i, image_path in enumerate(image_paths):
        logger.info(f"Performing OCR on image {i+1}/{len(image_paths)}: {image_path}")
        process_single_image(image_path, output_image_dir, metadata_dir, reader)


def main(image_dir: str, output_image_dir: str, metadata_dir: str):
    """Main function to handle OCR processing and save results."""
    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(LANGUAGES)

        # Load images
        image_paths = load_images(image_dir)
        logger.info(f"Loaded images from: {image_dir}")

        # Ensure output directories exist
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        # Process all images
        process_images(image_paths, output_image_dir, metadata_dir, reader)

        logger.info(OCR_SUCCESS)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(
        description="Perform OCR on images and save results in separate directories."
    )
    parser.add_argument(
        "image_dir", type=str, help="Directory containing the image files"
    )
    parser.add_argument(
        "output_image_dir",
        type=str,
        help="Directory to save the output images with bounding boxes",
    )
    parser.add_argument(
        "metadata_dir", type=str, help="Directory to save the OCR metadata (JSON) files"
    )

    args = parser.parse_args()

    # Process the specified directories
    main(args.image_dir, args.output_image_dir, args.metadata_dir)
