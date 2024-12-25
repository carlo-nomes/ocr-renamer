import argparse
from datetime import datetime
import json
import logging
import os
import tempfile
import uuid
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Constants
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


def convert_pdf(pdf_path: str, output_dir: str) -> None:
    """Convert a single PDF file to images."""

    # Get original file metadata
    pdf_size = os.path.getsize(pdf_path)
    pdf_created = datetime.fromtimestamp(os.path.getctime(pdf_path)).isoformat()
    pdf_modified = datetime.fromtimestamp(os.path.getmtime(pdf_path)).isoformat()

    logger.info(f"Converting PDF file: {pdf_path}")
    try:
        images = convert_from_path(pdf_path)
        logger.info(f"Found {len(images)} pages in the PDF file.")
    except PDFInfoNotInstalledError:
        logger.error("Poppler is not installed or not in PATH.")
        raise RuntimeError("Poppler is not installed or not in PATH.")
    except PDFPageCountError:
        logger.error("Unable to get page count. Ensure the PDF file is not corrupted.")
        raise RuntimeError("Unable to get page count. Ensure the PDF file is not corrupted.")
    except PDFSyntaxError:
        logger.error("The PDF file is malformed.")
        raise RuntimeError("The PDF file is malformed.")

    # Get the maximum width of the images
    max_width = max(image.width for image in images)
    # Get the total height of the images
    total_height = sum(image.height for image in images)

    # Create a new image with the maximum width and total height
    merged_image = Image.new("RGB", (max_width, total_height))

    # Paste each image into the new image
    y_offset = 0
    for image in images:
        merged_image.paste(image, (0, y_offset))
        y_offset += image.height

    # Save the new image
    image_file_name = str(uuid.uuid4()) + ".png"
    image_path = os.path.join(output_dir, image_file_name)
    merged_image.save(image_path)

    image_size = os.path.getsize(image_path)
    image_created = datetime.fromtimestamp(os.path.getctime(image_path)).isoformat()
    image_modified = datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat()
    logger.info(f"Image saved: {image_path}")

    # Close the images
    for image in images:
        image.close()

    # Save metadata to a file
    metadata = {
        # PDF metadata
        "pdf": pdf_path,
        "pdf_size": pdf_size,
        "pdf_created": pdf_created,
        "pdf_modified": pdf_modified,
        # Image metadata
        "image": image_path,
        "image_size": image_size,
        "image_created": image_created,
        "image_modified": image_modified,
    }
    metadata_file_name = image_file_name.replace(".png", "") + ".metadata.json"
    metadata_file_path = os.path.join(output_dir, metadata_file_name)
    with open(metadata_file_path, "w") as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=4))


def main(input_path: str, output_dir: str | None = None, clean: bool = False):
    """Main function to convert PDF files to images."""
    # Set the output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(output_dir):
        logger.error(f"The path {output_dir} is not a directory.")
        raise NotADirectoryError(f"The path {output_dir} is not a directory.")

    # Clean the output directory
    if clean:
        logger.info("Cleaning the output directory.")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    logger.info(f"Output directory: {output_dir}")

    # Convert the PDF file(s) to images
    if os.path.isfile(input_path):
        logger.info(f"Converting PDF file: {input_path}")
        convert_pdf(input_path, output_dir)
    elif os.path.isdir(input_path):
        logger.info(f"Converting PDF files in directory: {input_path}")
        pdf_files = [f for f in os.listdir(input_path) if f.endswith(".pdf")]
        logger.info(f"Found {len(pdf_files)} PDF files.")

        with ThreadPoolExecutor() as executor:
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_path, pdf_file)
                executor.submit(convert_pdf, pdf_path, output_dir)
    else:
        logger.error(f"The path {input_path} is not a valid file or directory.")
        raise FileNotFoundError(f"The path {input_path} is not a valid file or directory.")

    logger.info(f"Conversion completed. Images saved in: {output_dir}")


if __name__ == "__main__":

    # Set up arguments for command line usage
    parser = argparse.ArgumentParser(description="Convert PDF files to images.")
    parser.add_argument("input_path", help="Path to the PDF file or directory containing PDF files.")
    parser.add_argument("--output_dir", help="Directory to save the images. Default is a temporary directory.")
    parser.add_argument("--clean", action="store_true", help="Clean the temporary directory after conversion.")
    args = parser.parse_args()

    # Call the main function
    main(args.input_path, args.output_dir, args.clean)
