import os
import sys
import logging
import pytesseract
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

# Constants for configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
OCR_SUCCESS = "OCR completed."

PADDING_MULTIPLIER = 0.2  # Padding multiplier for bounding boxes
DEFAULT_BOX_COLOR = "blue"  # Color of bounding boxes
LINE_THICKNESS = 5  # Thickness of bounding box lines

# Maximum gap between characters to consider them part of the same word
MAX_X_GAP = 0.80
# Minimum vertical overlap between characters to consider them part of the same word
MIN_Y_OVERLAP = 0.5

# Edge distances for bounding boxes
MIN_EDGE_DISTANCE = 5

# Maximum size relative to the image size
MAX_BOX_WIDTH = 0.2
MAX_BOX_HEIGHT = 0.2
# Minimum size relative to the image size
MIN_BOX_WIDTH = 0.002
MIN_BOX_HEIGHT = 0.005

# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()


def load_images(image_dir: str) -> list:
    """Load preprocessed images from the specified directory."""
    image_paths = sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(".png")
        ],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
    )
    return image_paths


def map_box(box: str) -> tuple:
    """Convert a box string to coordinates."""
    # Convert coordinates to integers
    x0, y0, x1, y1, *_ = map(int, box.split()[1:])
    # Get the character
    character = box.split()[0]
    return x0, y0, x1, y1, character


def pad_box(box: tuple) -> tuple:
    """Pad bounding box relative to the character size."""
    x0, y0, x1, y1, *_ = box

    width = x1 - x0
    height = y1 - y0

    # Calculate padding
    padding = int(min(width, height) * PADDING_MULTIPLIER)

    # Apply padding
    x0 -= padding
    y0 -= padding
    x1 += padding
    y1 += padding

    return x0, y0, x1, y1, *_


def get_y_overlap(box1: tuple, box2: tuple) -> int:
    """Calculate the vertical overlap between two bounding boxes as a percentage of the smaller box."""
    x0_1, y0_1, x1_1, y1_1, *_ = box1
    x0_2, y0_2, x1_2, y1_2, *_ = box2

    # Calculate the overlap in the y-axis
    y_overlap = min(y1_1, y1_2) - max(y0_1, y0_2)

    # Calculate the height of the smaller box
    min_height = min(y1_1 - y0_1, y1_2 - y0_2)

    return y_overlap / min_height


def get_x_gap(box1: tuple, box2: tuple) -> int:
    """Calculate the horizontal gap between two bounding boxes as a percentage of the total width."""
    total_width = box2[2] - box1[0]
    gap = box2[0] - box1[2]

    return gap / total_width


def is_overlapping(box1: tuple, box2: tuple) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x0_1, y0_1, x1_1, y1_1, *_ = box1
    x0_2, y0_2, x1_2, y1_2, *_ = box2

    # Calculate the area of the two bounding boxes
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

    # Calculate the area of the intersection
    x0 = max(x0_1, x0_2)
    y0 = max(y0_1, y0_2)
    x1 = min(x1_1, x1_2)
    y1 = min(y1_1, y1_2)
    intersection_area = max(0, x1 - x0) * max(0, y1 - y0)

    # Calculate the area of the union
    union_area = area1 + area2 - intersection_area

    # Calculate the intersection over union (IoU)
    iou = intersection_area / union_area
    logger.debug(f"IoU: {iou}")

    return iou > 0.1


def ocr_boxes(image: Image) -> list:
    """Perform OCR on the given image and return bounding boxes."""
    custom_config = r"--oem 3 --psm 6"
    boxes = pytesseract.image_to_boxes(image, config=custom_config)

    return [map_box(box) for box in boxes.splitlines()]


def ocr_words(image: Image) -> list:
    """Perform OCR on the given image and return words."""
    custom_config = r"--oem 3 --psm 6"
    words = pytesseract.image_to_string(image, config=custom_config)
    return words


def merge_bounding_boxes(boxes: list) -> list:
    """Merge character bounding boxes into word/number bounding boxes."""
    if not boxes:
        return []

    word_boxes = []
    current_word_box = []
    prev_box = None

    for _i, box in enumerate(boxes):
        x_gap = get_x_gap(prev_box, box) if prev_box else 0

        # Determine if the current character is on the same line as the previous character
        y_overlap = get_y_overlap(prev_box, box) if prev_box else 0

        # Break into a new word if there is a character recognized as a space or if the x-gap is too large or if the y-overlap is too small
        if prev_box and (
            box[4].isspace() or x_gap > MAX_X_GAP or y_overlap < MIN_Y_OVERLAP
        ):
            word_boxes.append(current_word_box)
            current_word_box = []

        current_word_box.append(box)
        prev_box = box

    # Append the last word
    if current_word_box:
        word_boxes.append(current_word_box)

    merged_objects = []
    for word_box in word_boxes:
        x0 = min([b[0] for b in word_box])
        y0 = min([b[1] for b in word_box])
        x1 = max([b[2] for b in word_box])
        y1 = max([b[3] for b in word_box])

        word = "".join([b[4] for b in word_box])
        merged_objects.append((x0, y0, x1, y1, word))

    return merged_objects


def filter_bounding_boxes(boxes: list, image_width: int, image_height: int) -> list:
    """Filter out large and small bounding boxes"""

    filtered_boxes = []
    for index, box in enumerate(boxes):
        x0, y0, x1, y1, *_ = box

        # Calculate box width
        box_width = x1 - x0
        box_width_ratio = box_width / image_width
        if box_width_ratio > MAX_BOX_WIDTH:
            logger.debug(f"Box {index} too wide ({box_width_ratio})")
            continue
        if box_width_ratio < MIN_BOX_WIDTH:
            logger.debug(f"Box {index} too narrow ({box_width_ratio})")
            continue

        # Calculate box height
        box_height = y1 - y0
        box_height_ratio = box_height / image_height
        if box_height_ratio > MAX_BOX_HEIGHT:
            logger.debug(f"Box {index} too tall ({box_height_ratio})")
            continue
        if box_height_ratio < MIN_BOX_HEIGHT:
            logger.debug(f"Box {index} too short ({box_height_ratio})")
            continue

        # Check if the box is too close to the edge
        if x0 < MIN_EDGE_DISTANCE:
            logger.debug(f"Box {index} too close to the left edge ({x0})")
            continue
        if y0 < MIN_EDGE_DISTANCE:
            logger.debug(f"Box {index} too close to the top edge ({y0})")
            continue
        if image_width - x1 < MIN_EDGE_DISTANCE:
            logger.debug(
                f"Box {index} too close to the right edge ({image_width - x1})"
            )
            continue
        if image_height - y1 < MIN_EDGE_DISTANCE:
            logger.debug(
                f"Box {index} too close to the bottom edge ({image_height - y1})"
            )
            continue

        filtered_boxes.append(box)

    return filtered_boxes


def convert_box_to_pillow(image: Image, box: tuple) -> tuple:
    """Convert Tesseract's box format to Pillow's box format."""
    x0, y0, x1, y1, *_ = box

    # Convert Tesseract's bottom-left origin to Pillow's top-left origin
    x0_pillow = x0
    x1_pillow = x1
    y0_pillow = image.height - y0
    y1_pillow = image.height - y1

    # Swap if necessary to ensure y0 is greater than or equal to y1
    if y0_pillow < y1_pillow:
        y0_pillow, y1_pillow = y1_pillow, y0_pillow

    return x0_pillow, y0_pillow, x1_pillow, y1_pillow


def draw_bounding_boxes(
    image: Image, boxes: list, color: str = DEFAULT_BOX_COLOR, with_index: bool = False
) -> Image:
    """Draw bounding boxes on the image and save it."""
    # Copy the image to draw on
    image = image.copy()

    draw = ImageDraw.Draw(image)
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = convert_box_to_pillow(image, box)

        # Draw bounding box
        draw.rectangle([(x0, y1), (x1, y0)], outline=color, width=LINE_THICKNESS)

        # Draw the index of the bounding box
        if with_index:
            draw.text(
                (x0, y1),
                str(index),
                font=ImageFont.load_default(35),
                fill=color,
                # Align text to the top-left corner of the box
                align="left",
                anchor="ld",
            )

    return image


def preprocess_image(img: Image) -> Image:
    """Preprocess an image for OCR."""
    # Convert to numpy array
    img = np.array(img)

    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale if the image is in color
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    # increases the white region in the image
    img = cv2.dilate(img, kernel, iterations=1)
    # erodes away the boundaries of foreground object
    img = cv2.erode(img, kernel, iterations=1)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Convert back to Pillow image
    return Image.fromarray(img)


def main(image_dir: str, output_dir: str):
    """Main function to handle OCR processing and draw bounding boxes."""
    try:
        # Load preprocessed images
        image_paths = load_images(image_dir)
        max_index = len(image_paths) - 1
        logger.info(f"Loaded images from: {image_dir}")

        # Perform OCR, draw bounding boxes, and save annotated images
        for i, image_path in enumerate(image_paths):
            logger.info(f"Performing OCR on image {i}/{max_index}: {image_path}")

            # Load image
            image = Image.open(image_path)
            image_width = image.width
            image_height = image.height

            # Perform OCR and get character bounding boxes
            boxes = ocr_boxes(image)
            logger.info(f"Found {len(boxes)} bounding boxes.")

            # Draw character bounding boxes on the image
            boxed_image = draw_bounding_boxes(image, boxes, "red", True)

            # Filter out very large and very small bounding boxes
            boxes = filter_bounding_boxes(boxes, image_width, image_height)
            logger.info(f"Filtered to {len(boxes)} bounding boxes.")

            # Merge character bounding boxes into word/number bounding boxes
            boxes = merge_bounding_boxes(boxes)
            logger.info(f"Merged into {len(boxes)} word/number bounding boxes.")

            # Pad bounding boxes
            boxes = [pad_box(box) for box in boxes]
            logger.info(f"Padded bounding boxes.")

            # Draw processed bounding boxes on the image
            boxed_image = draw_bounding_boxes(boxed_image, boxes, "blue", True)

            # Save image with bounding boxes
            boxed_image_path = os.path.join(output_dir, f"image_{i}_boxes.png")
            boxed_image.save(boxed_image_path)
            logger.info(f"Annotated image saved at: {boxed_image_path}")

            # Crop and preprocess each bounding box
            boxed_images = []
            for j, box in enumerate(boxes):
                x0, y0, x1, y1 = convert_box_to_pillow(image, box)
                cropped_image = image.crop((x0, y1, x1, y0))
                processed_image = preprocess_image(cropped_image)
                boxed_images.append(processed_image)

            # Run OCR on cropped images
            for j, boxed_image in enumerate(boxed_images):
                words = ocr_words(boxed_image)

                # Draw OCR output on the image
                draw = ImageDraw.Draw(boxed_image)
                draw.text((10, 10), words, fill="red")

                # Save OCR output
                image_path = os.path.join(output_dir, f"image_{i}_box_{j}.png")
                boxed_image.save(image_path)
                logger.info(f"OCR output saved at: {image_path}")

        logger.info(OCR_SUCCESS)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: python ocr_process.py <image_dir> <output_dir>")
        sys.exit(1)

    image_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    main(image_dir, output_dir)
