import os
import sys
import logging
import tempfile
import cv2
import numpy as np
from typing import List
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)
from PIL import Image

# Constants
TEMP_DIR = tempfile.mkdtemp()


TEMP_DIR_IMAGES = os.path.join(TEMP_DIR, "converted")
os.makedirs(TEMP_DIR_IMAGES, exist_ok=True)

TEMP_DIR_PREPROCESSED = os.path.join(TEMP_DIR, "preprocessed")
os.makedirs(TEMP_DIR_PREPROCESSED, exist_ok=True)

LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
PDF_CONVERT_SUCCESS = "PDF successfully converted to images."
PREPROCESS_SUCCESS = "Preprocessed images saved at: {}"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()


def convert_pdf_to_images(pdf_path: str) -> List[str]:
    """Convert a PDF file to images and save them to TEMP_DIR."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    try:
        logger.info("Converting PDF to images...")
        images = convert_from_path(pdf_path)
        logger.info(PDF_CONVERT_SUCCESS)
    except PDFInfoNotInstalledError:
        raise RuntimeError("Poppler is not installed or not in PATH.")
    except PDFPageCountError:
        raise RuntimeError(
            "Unable to get page count. Ensure the PDF file is not corrupted."
        )
    except PDFSyntaxError:
        raise RuntimeError("The PDF file is malformed.")

    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(TEMP_DIR_IMAGES, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
        logger.debug(f"Saved image: {image_path}")

    return image_paths


def preprocess_image(image_path: str, image_index: int) -> str:
    """Preprocess an image and save the preprocessed image to TEMP_DIR."""
    # Load image
    image = Image.open(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Save preprocessed image for inspection
    preprocessed_image_path = os.path.join(
        TEMP_DIR_PREPROCESSED, f"page_{image_index + 1}.png"
    )
    cv2.imwrite(preprocessed_image_path, thresh)
    logger.debug(f"Preprocessed image saved: {preprocessed_image_path}")

    return preprocessed_image_path


def main(pdf_path: str):
    """Main function to handle PDF conversion and image preprocessing."""
    try:
        # Step 1: Convert PDF to images
        image_paths = convert_pdf_to_images(pdf_path)
        logger.info(f"Converted images saved at: {TEMP_DIR_IMAGES}")

        # Step 2: Preprocess each image
        preprocessed_image_paths = []
        for i, image_path in enumerate(image_paths):
            preprocessed_image_path = preprocess_image(image_path, i)
            preprocessed_image_paths.append(preprocessed_image_path)
        logger.info(PREPROCESS_SUCCESS.format(TEMP_DIR_PREPROCESSED))

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python pdf_converter.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)
