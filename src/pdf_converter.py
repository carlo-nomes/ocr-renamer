import os
import sys
import logging
import tempfile
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

# Constants
OUTPUT_DIR = tempfile.mkdtemp()

LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
CONVERSION_SUCCESS = "PDF file has been converted to images. Images are saved in: {}"

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


def main(pdf_path: str):
    """Main function to convert a PDF file to images."""
    # Check if the PDF file exists
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    # Convert the PDF file to images
    try:
        images = convert_from_path(pdf_path)
    except PDFInfoNotInstalledError:
        raise RuntimeError("Poppler is not installed or not in PATH.")
    except PDFPageCountError:
        raise RuntimeError(
            "Unable to get page count. Ensure the PDF file is not corrupted."
        )
    except PDFSyntaxError:
        raise RuntimeError("The PDF file is malformed.")

    # Save the images to a temporary directory
    for i, image in enumerate(images):
        image_path = os.path.join(OUTPUT_DIR, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        logger.debug(f"Saved image: {image_path}")

    logger.info(CONVERSION_SUCCESS.format(OUTPUT_DIR))
    print(OUTPUT_DIR)  # This is the output of the script


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python pdf_converter.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)
