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
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
CONVERSION_SUCCESS = "PDF file has been converted to images. Images are saved in: {}"

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


def main(pdf_path: str, output_dir: str = None):
    """Main function to convert a PDF file to images."""

    # Set output directory (default to temporary directory if not provided)
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory at: {output_dir}")
            os.makedirs(output_dir)

    logger.info(f"Output directory set to: {output_dir}")

    # Check if the PDF file exists
    if not os.path.isfile(pdf_path):
        logger.error(f"The file {pdf_path} does not exist.")
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    # Convert the PDF file to images
    try:
        logger.info(f"Converting PDF file: {pdf_path}")
        images = convert_from_path(pdf_path)
    except PDFInfoNotInstalledError:
        logger.error("Poppler is not installed or not in PATH.")
        raise RuntimeError("Poppler is not installed or not in PATH.")
    except PDFPageCountError:
        logger.error("Unable to get page count. Ensure the PDF file is not corrupted.")
        raise RuntimeError(
            "Unable to get page count. Ensure the PDF file is not corrupted."
        )
    except PDFSyntaxError:
        logger.error("The PDF file is malformed.")
        raise RuntimeError("The PDF file is malformed.")

    # Save the images to the output directory
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        logger.info(f"Saved image: {image_path}")

    logger.info(CONVERSION_SUCCESS.format(output_dir))
    print(output_dir)  # Print the output directory


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        logger.error("Usage: python pdf_converter.py <path_to_pdf> [output_directory]")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Optional argument for output directory
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None

    main(pdf_path, output_dir)
