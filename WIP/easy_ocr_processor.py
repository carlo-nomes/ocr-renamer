import argparse
import json
import logging
import os
import platform
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import easyocr
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Constants for configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"

LANGUAGES = ["en", "nl"]  # Languages for OCR (English and Dutch)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

# Suppress excessive PIL debug logs
logging.getLogger("PIL").setLevel(logging.INFO)


# Functionality for drawing bounding boxes and text on images

FG_COLOR = "red"  # Color of bounding boxes and text
LINE_THICKNESS = 5  # Thickness of bounding box lines
CONTENT_PADDING = 10  # Padding for text inside the bounding box
DEFAULT_FONT_SIZE = 30  # Default font size for text


def get_font(size: int = DEFAULT_FONT_SIZE) -> ImageFont:
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


def convert_box_to_pillow(box: list) -> tuple:
    """Convert a box with 4 points to a tuple (x0, y0, x1, y1) for Pillow."""
    assert len(box) == 4, "Box must have 4 points."
    assert all(len(pt) == 2 for pt in box), "Each point in the box must have 2 coordinates."
    assert all(isinstance(pt[0], (int, float)) and isinstance(pt[1], (int, float)) for pt in box), "Coordinates must be integers or floats."

    x0, y0 = box[0]
    x1, y1 = box[2]
    return x0, y0, x1, y1


def draw_boxes(image: Image, boxes: list) -> Image:
    """Draw bounding boxes on the image and save it."""
    assert isinstance(image, Image.Image), "Input must be a PIL Image object."
    assert all(isinstance(box, dict) for box in boxes), "Boxes must be a list of dictionaries."
    assert all("box" in box and "text" in box and "confidence" in box for box in boxes), "Boxes must have 'box', 'text', and 'confidence' keys."

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for index, box in enumerate(boxes):
        # Draw the bounding box
        x0, y0, x1, y1 = convert_box_to_pillow(box["box"])

        # Outline the bounding box (optional, adjust the width or color)
        draw.rectangle([(x0, y0), (x1, y1)], outline=FG_COLOR, width=LINE_THICKNESS)
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
        draw.text((text_x, text_y), text, fill=FG_COLOR, font=font)
        logger.debug(f"Drew text '{text}' at {text_x, text_y}")

        # Draw the index of the box in the top-left corner of the bounding box
        index_text = str(index)
        index_font = get_font()

        # Get the bounding box for the index text
        index_text_bbox = draw.textbbox((0, 0), index_text, font=index_font)

        # Calculate index text width and height for the top-left corner
        index_text_width = index_text_bbox[2] - index_text_bbox[0]
        index_text_height = index_text_bbox[3] - index_text_bbox[1]
        index_x = x0 + CONTENT_PADDING  # Padding from the left edge of the bounding box
        index_y = y0 + CONTENT_PADDING  # Padding from the top edge of the bounding box

        # Draw the index in the top-left corner
        draw.text((index_x, index_y), index_text, fill=FG_COLOR, font=index_font)
        logger.debug(f"Drew index '{index_text}' at {index_x, index_y}")

        # Draw the confidence score in the bottom-right corner of the bounding box
        confidence_text = f"{box['confidence']:.2f}"
        confidence_font = get_font()

        # Get the bounding box for the confidence text
        confidence_text_bbox = draw.textbbox((0, 0), confidence_text, font=confidence_font)

        # Calculate confidence text width and height for the bottom-right corner
        confidence_text_width = confidence_text_bbox[2] - confidence_text_bbox[0]
        confidence_text_height = confidence_text_bbox[3] - confidence_text_bbox[1]
        confidence_x = x1 - confidence_text_width - CONTENT_PADDING
        confidence_y = y1 - confidence_text_height - CONTENT_PADDING

        # Draw the confidence score in the bottom-right corner
        draw.text((confidence_x, confidence_y), confidence_text, fill=FG_COLOR, font=confidence_font)
        logger.debug(f"Drew confidence '{confidence_text}' at {confidence_x, confidence_y}")

    return draw_image


# OCR processing with EasyOCR

EASY_OCR_READER = easyocr.Reader(LANGUAGES)


def run_easy_ocr(input_image_path: str) -> list[dict]:
    """Perform OCR on the input image and return the results in a serializable format."""
    results = EASY_OCR_READER.readtext(input_image_path)

    # Map the results to a serializable format
    serialized: list[dict] = []
    for result in results:
        mapped = {
            "box": [[int(pt[0]), int(pt[1])] for pt in result[0]],
            "text": result[1],
            "confidence": float(result[2]),
        }
        serialized.append(mapped)

    return serialized


# Preprocessing functionality

# Constants for preprocessing
RESCALE_FACTOR = 1.5  # Factor by which the image will be rescaled
DILATION_ITERATIONS = 1  # Number of times dilation is applied to the image
EROSION_ITERATIONS = 1  # Number of times erosion is applied to the image
KERNEL_SIZE = (1, 1)  # Size of the kernel used for dilation and erosion


def deskew_image(image: np.array) -> np.array:
    coords = np.column_stack(np.where(image > 0))

    # Check if there are enough coordinates to determine skew
    if len(coords) == 0:
        return image  # No deskew needed

    # Get the angle from the minimum area rectangle around the non-zero points
    angle = cv2.minAreaRect(coords)[-1]

    # Correct the angle if necessary
    if angle < -45:
        angle = 90 + angle  # Rotate counterclockwise
    elif angle > 45:
        angle = -(90 - angle)  # Rotate clockwise

    # If the skew angle is very small, skip rotation
    if abs(angle) < 0.5:  # Skip rotation if skew is minimal
        return image

    # Rotate the image to correct for skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image


# Constants for preprocessing
RESCALE_FACTOR = 1.0  # Factor by which the image will be rescaled (suggested range: 1.0 to 2.0)
DILATION_ITERATIONS = 1  # Number of times dilation is applied to the image (suggested range: 1 to 5)
EROSION_ITERATIONS = 1  # Number of times erosion is applied to the image (suggested range: 1 to 5)
KERNEL_SIZE = (1, 1)  # Size of the kernel used for dilation and erosion (suggested range: (1, 1) to (5, 5))
ADAPTIVE_THRESH_BLOCKSIZE = 11  # Block size for adaptive thresholding (suggested range: 3 to 21, must be odd)
ADAPTIVE_THRESH_C = 2  # Constant subtracted from the mean in adaptive thresholding (suggested range: 0 to 10)
MEDIAN_BLUR_KSIZE = 3  # Kernel size for median blur (suggested range: 1 to 7, must be odd)
SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Kernel for edge sharpening (customizable)
BRIGHTNESS_ALPHA = 1.0  # Alpha value for brightness/contrast adjustment (suggested range: 1.0 to 3.0)
BRIGHTNESS_BETA = 0  # Beta value for brightness/contrast adjustment (suggested range: -100 to 100)


def preprocess_image(image: Image) -> Image:
    """Preprocess an image for OCR."""
    image = np.array(image)

    # Rescale the image, if needed.
    image = cv2.resize(image, None, fx=RESCALE_FACTOR, fy=RESCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for scanned documents
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCKSIZE, ADAPTIVE_THRESH_C)

    # Denoise using median blur
    image = cv2.medianBlur(image, MEDIAN_BLUR_KSIZE)

    # Apply edge sharpening
    image = cv2.filter2D(image, -1, SHARPEN_KERNEL)

    # Apply brightness/contrast adjustments
    image = cv2.convertScaleAbs(image, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)

    # Deskew the image if necessary
    image = deskew_image(image)

    # Convert back to Pillow image
    image = Image.fromarray(image)

    # Ensure the image is in RGBA mode
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    return image


# Main processing function
def process_image(input_image_path: str, output_dir: str):
    """Process the image with OCR and save the results."""
    try:
        # Open the image and get additional information
        input_img = Image.open(input_image_path)
        input_img_name = os.path.splitext(os.path.basename(input_image_path))[0]

        # Load the metadata for the input image if available
        original_metadata = {}
        input_metadata_path = os.path.join(os.path.dirname(input_image_path), f"{input_img_name}.metadata.json")
        if os.path.exists(input_metadata_path):
            with open(input_metadata_path, "r") as f:
                original_metadata = json.load(f)
            logger.info(f"Loaded metadata from: {input_metadata_path}")
        else:
            logger.warning(f"No metadata found for image: {input_image_path}")

        # TODO: Experiment with values, currently reduces OCR accuracy
        # Preprocess the image before OCR and save it
        # preprocessed_image = preprocess_image(input_image)
        # preprocessed_path = os.path.join(output_image_dir, f"{img_name}.preprocessed.png")
        # preprocessed_image.save(preprocessed_path)
        # logger.info(f"Preprocessed image saved to: {preprocessed_path}")

        # Perform OCR on the preprocessed image
        boxes = run_easy_ocr(input_image_path)
        logger.info(f"Found {len(boxes)} OCR results in image: {input_image_path}")

        # Draw bounding boxes on the image and save it
        boxed_image = draw_boxes(input_img, boxes)
        boxed_image_path = os.path.join(output_dir, f"{input_img_name}.boxed.png")
        boxed_image.save(boxed_image_path)
        logger.info(f"Boxed image saved to: {boxed_image_path}")

        # Save metadata to a JSON file with the OCR results
        metadata = {
            "original_image": input_image_path,
            "original_metadata": original_metadata,
            # "preprocessed_image": preprocessed_path,
            "boxed_image": boxed_image_path,
            "boxes": boxes,
        }
        metadata_path = os.path.join(output_dir, f"{input_img_name}.metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to: {metadata_path}")

    except Exception as e:
        logger.error(f"Error processing image '{input_image_path}': {e}")
    finally:
        input_img.close()  # Ensure original image is closed after processing
        # preprocessed_image.close()  # Ensure preprocessed image is closed after processing
        boxed_image.close()  # Ensure boxed image is closed after processing


MAX_CONCURRENT_WORKERS = 4


def main(input_path: str, output_path: str = tempfile.mkdtemp(), clean: bool = False):
    """Main function to handle OCR processing and save results."""

    os.makedirs(output_path, exist_ok=True)

    # Clean the output directory if requested
    if clean:
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
        logger.info(f"Cleaned output directory: {output_path}")

    # Load images from the input directory or file
    input_images = []
    if os.path.isfile(input_path):
        input_images = [input_path]
    elif os.path.isdir(input_path):
        input_images = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    logger.info(f"Found {len(input_images)} images for path: {input_path}")

    # Process images with OCR using EasyOCR in parallel
    max_workers = min(MAX_CONCURRENT_WORKERS, len(input_images))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for input_image_path in input_images:
            executor.submit(process_image, input_image_path, output_path)
    logger.info("OCR processing complete.")


if __name__ == "__main__":
    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(description="Perform OCR on images and save results in separate directories.")
    parser.add_argument("input_dir", type=str, help="Directory containing images to process.")
    parser.add_argument("output_dir", type=str, nargs="?", default=None, help="Directory to save processed images and metadata.")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before processing.")
    args = parser.parse_args()

    # Process the specified directories
    main(args.input_dir, args.output_dir, args.clean)
