import argparse
import json
import logging
import os
import platform
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import easyocr
from PIL import Image, ImageDraw, ImageFont

# Constants for configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"

LANGUAGES = ["en", "nl"]  # Languages for OCR (English and Dutch)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

# Suppress excessive PIL debug logs
logging.getLogger("PIL").setLevel(logging.INFO)


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


def save_metadata(original_image_path: str, boxed_image_path: str, boxes: list, metadata_dir: str):
    """Save the OCR metadata (boxes) to a JSON file."""
    metadata = {
        "original_image": original_image_path,
        "boxed_image": boxed_image_path,
        "boxes": boxes,
    }

    # Save metadata to a JSON file
    metadata_filename = f"{os.path.splitext(os.path.basename(original_image_path))[0]}.json"
    metadata_path = os.path.join(metadata_dir, metadata_filename)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Metadata saved to: {metadata_path}")


def convert_box_to_pillow(box: list) -> tuple:
    """Convert EasyOCR's box format to Pillow's box format."""
    x0, y0 = box[0]
    x1, y1 = box[2]
    return x0, y0, x1, y1


def get_font(size: int = 100) -> ImageFont:
    """Return a font that works on the current operating system."""
    try:
        if platform.system() == "Windows":
            return ImageFont.truetype("arial.ttf", size)
        elif platform.system() == "Darwin":  # macOS
            return ImageFont.truetype("Helvetica.ttc", size)
        else:  # Linux and others
            return ImageFont.truetype("DejaVuSans.ttf", size)
    except IOError:
        logger.warning("Font not found. Using default font")
        return ImageFont.load_default()


def get_max_font_for_box(box: list, text: str, draw: ImageDraw):
    """Get the maximum font size that fits the text inside the box."""
    x0, y0, x1, y1 = box

    # Start with a large font size
    font_size = y1 - y0

    # Decrease the font size until the text fits inside the box
    while font_size > 0:
        font = get_font(font_size)
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        if tx1 - tx0 < x1 - x0 - 2 * CONTENT_PADDING and ty1 - ty0 < y1 - y0 - 2 * CONTENT_PADDING:
            break  # Text fits inside the box
        font_size -= 1

    return font_size


BOX_COLOR_FG = "red"  # Color of bounding boxes and text
LINE_THICKNESS = 5  # Thickness of bounding box lines
CONTENT_PADDING = 10  # Padding for text inside the bounding box


def draw_bounding_boxes(image: Image, boxes: list, with_index: bool = False) -> Image:
    """Draw bounding boxes on the image and save it."""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for index, box in enumerate(boxes):
        # Draw the bounding box
        x0, y0, x1, y1 = convert_box_to_pillow(box["box"])

        # Outline the bounding box (optional, adjust the width or color)
        draw.rectangle([(x0, y0), (x1, y1)], outline=BOX_COLOR_FG, width=LINE_THICKNESS)
        logger.debug(f"Drawing box {index}: {x0, y0, x1, y1}")

        # Draw the recognized text in the center of the bounding box
        text = box["text"]

        # Get the font size that fits the text inside the box
        font = get_font(get_max_font_for_box((x0, y0, x1, y1), text, draw))

        # Calculate the width and height of the text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate the center position for the text
        text_x = x0 + (x1 - x0 - text_width) / 2
        text_y = y0 + (y1 - y0 - text_height) / 2

        # Draw the text in the center of the bounding box
        draw.text((text_x, text_y), text, fill=BOX_COLOR_FG, font=font)
        logger.debug(f"Drawing text {text} at {text_x, text_y}")

        # Draw the index of the box in the top-left corner of the bounding box
        if with_index:
            index_font_size = 30  # Adjust font size for the index
            index_text = str(index)
            index_font = get_font(index_font_size)

            # Get the bounding box for the index text
            index_text_bbox = draw.textbbox((0, 0), index_text, font=index_font)

            # Calculate index text width and height for the top-left corner
            index_text_width = index_text_bbox[2] - index_text_bbox[0]
            index_text_height = index_text_bbox[3] - index_text_bbox[1]
            index_x = x0 + CONTENT_PADDING  # Padding from the left edge of the bounding box
            index_y = y0 + CONTENT_PADDING  # Padding from the top edge of the bounding box

            # Draw the index in the top-left corner
            draw.text((index_x, index_y), index_text, fill=BOX_COLOR_FG, font=index_font)

    return draw_image


def process_single_image(image_path: str, output_image_dir: str, metadata_dir: str, reader: easyocr.Reader):
    """Process a single image: Perform OCR, draw bounding boxes, and save the results."""
    try:
        # Open the image
        img = Image.open(image_path)
        logger.info(f"Processing image format: {img.format}")

        # Perform OCR
        boxes = perform_ocr(image_path, reader)
        logger.info(f"Found {len(boxes)} text boxes in image {image_path}")

        # Draw bounding boxes on the image
        boxed_image = draw_bounding_boxes(img, boxes, with_index=True)

        # Save the image with bounding boxes in the output image directory
        output_image_path = os.path.join(output_image_dir, f"{os.path.basename(image_path)}")
        boxed_image.save(output_image_path)
        logger.info(f"Boxed image saved to: {output_image_path}")

        # Save metadata (OCR results in JSON in metadata directory)
        save_metadata(image_path, output_image_path, boxes, metadata_dir)
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
    finally:
        img.close()  # Ensure image is closed after processing


def main(input_dir: str, output_dir: str | None, clean: bool = False):
    """Main function to handle OCR processing and save results."""

    # Set the output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(output_dir):
        logger.error(f"The path {output_dir} is not a directory.")
        raise NotADirectoryError(f"The path {output_dir} is not a directory.")

    # Set up directory for the images
    output_image_dir = os.path.join(output_dir, "boxed_images")
    os.makedirs(output_image_dir, exist_ok=True)
    if clean:
        for filename in os.listdir(output_image_dir):
            file_path = os.path.join(output_image_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    logger.info(f"Initalized output directory: {output_image_dir}")

    # Set up directory for metadata
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    if clean:
        for filename in os.listdir(metadata_dir):
            file_path = os.path.join(metadata_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    logger.info(f"Initalized metadata directory: {metadata_dir}")

    try:
        # Initialize EasyOCR reader

        # Load images
        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        logger.info(f"Loaded images from: {input_dir}")

        # Process all images
        reader = easyocr.Reader(LANGUAGES)
        max_workers = min(4, len(image_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for image_path in image_paths:
                executor.submit(process_single_image, image_path, output_image_dir, metadata_dir, reader)
        logger.info("OCR processing complete.")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(description="Perform OCR on images and save results in separate directories.")
    parser.add_argument("input_dir", type=str, help="Directory containing images to process.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save the processed images.")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before processing.")
    args = parser.parse_args()

    # Process the specified directories
    main(args.input_dir, args.output_dir, args.clean)
