import logging
import os
import tempfile
from typing import List

from pdf2image import convert_from_path

from src.utils.file_io import list_files_in_directory

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DPI = 300
DEFAULT_OUTPUT_FORMAT = "JPEG"


class PDFConverter:
    """
    A class for preprocessing PDF files to run OCR on them.
    """

    def __init__(self, dpi: int = DEFAULT_DPI, output_format: str = DEFAULT_OUTPUT_FORMAT) -> None:
        """
        Constructor for the PDFConverter class.

        :param dpi: The DPI to use for the conversion.
        :param output_format: The output format to use for the conversion.
        """
        self.dpi = dpi
        self.output_format = output_format

    def convert(
        self,
        pdf_path: str,
        output_dir: str | None = None,
        clear_output=True,
        recursive: bool = False,
        single_file: bool = True,
    ) -> List[str]:
        """
        Converts a PDF file to images.

        :param pdf_path: The path to the PDF file or directory containing PDF files.
        :param output_dir: The output directory for the converted images, defaults to a temporary directory.
        :param clear_output: Whether to clear the output directory before converting the PDF files.
        :param recursive: Whether to search for PDF files recursively in the directory.
        :param single_file: Whether to save all pages in a single image or create separate images for each page.
        :return: A list of paths to the converted images.
        """
        # Get the paths to the PDF files, either a single file or all files in a directory
        if os.path.isdir(pdf_path):
            pdf_paths = list_files_in_directory(pdf_path, recursive, extensions=[".pdf"])
        elif pdf_path.endswith(".pdf"):
            logger.debug(f"Converting PDF file: {pdf_path}")
            pdf_paths = [pdf_path]
        else:
            logger.error(f"Invalid file extension for PDF file: {pdf_path}")
            raise ValueError(f"Invalid file extension for PDF file: {pdf_path}")

        # Check if any PDF files were found
        if not pdf_paths or len(pdf_paths) == 0:
            logger.error(f"No PDF files found in directory: {pdf_path}")
            raise ValueError(f"No PDF files found in directory: {pdf_path}")

        # Prepare the output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            logger.info(f"Using temporary directory for converted images: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        if clear_output:
            files = list_files_in_directory(output_dir)
            for file in files:
                os.remove(file)
            logger.info(f"Cleared output directory: {output_dir}")

        result = []
        for pdf_path in pdf_paths:
            pdf_name = os.path.basename(pdf_path)
            pdf_name = pdf_name.split(".")[0]
            output_dir_pdf = os.path.join(output_dir, pdf_name)

            # Convert the PDF file to images
            images = convert_from_path(
                pdf_path,
                output_folder=output_dir_pdf,
                single_file=single_file,
                dpi=self.dpi,
                fmt=self.output_format,
            )
            result.extend(images)
            logger.debug(f'Converted PDF file "{pdf_path}" to {len(images)} images in "{output_dir_pdf}".')

        return result
