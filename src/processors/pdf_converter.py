import logging
import os
from pathlib import Path
import tempfile
from PIL import Image
from typing import List

from pdf2image import convert_from_path

from src.utils.file_io import empty_directory, list_files_in_directory

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FORMAT = "JPEG"


class PDFConverter:
    """
    A class for preprocessing PDF files to run OCR on them.
    """

    def __init__(self, output_dir: Path = None, output_format: str = DEFAULT_OUTPUT_FORMAT, overwrite: bool = False) -> None:
        """
        Initializes the PDFConverter class.
        """
        self.output_dir = output_dir if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.output_dir.is_dir() or not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {self.output_dir}")

        self.output_format = output_format
        self.overwrite = overwrite

    def combine_images(self, images: List[Image.Image]) -> Image.Image:
        """
        Combine a list of images into a single long image.
        """
        widths, heights = zip(*(i.size for i in images))
        total_width = max(widths)
        total_height = sum(heights)

        combined_image = Image.new("RGB", (total_width, total_height))
        y_offset = 0
        for image in images:
            combined_image.paste(image, (0, y_offset))
            y_offset += image.size[1]

        return combined_image

    def convert(self, pdf_path: Path, recursive: bool = False, single_image: bool = True) -> list[Path]:
        pdf_paths = []
        if pdf_path.is_dir():
            pdf_paths = list_files_in_directory(pdf_path, recursive=recursive, extensions=[".pdf"])
        elif pdf_path.is_file() and pdf_path.suffix == ".pdf":
            pdf_paths = [pdf_path]
        else:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        results = []
        for pdf in pdf_paths:
            output_dir = self.output_dir / pdf.stem
            # If overwrite is enabled, empty the directory
            if self.overwrite:
                empty_directory(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

            # Convert the PDF file to images
            images = convert_from_path(str(pdf), fmt=self.output_format)

            # Save each image and collect paths
            if not single_image:
                image_paths = []
                for i, image in enumerate(images):
                    image_path = output_dir / f"{pdf.stem}-{i+1}.{self.output_format.lower()}"
                    image.save(image_path, self.output_format)
                    image_paths.append(image_path)

                results.extend(image_paths)
                logger.debug(f"Converted {pdf} to {len(images)} images in {output_dir}.")
                continue

            # Combine images into a single file
            combined_image = self.combine_images(images)
            combined_image_path = output_dir / f"{pdf.stem}.{self.output_format.lower()}"
            combined_image.save(combined_image_path, self.output_format)
            results.append(combined_image_path)
            logger.debug(f"Converted {pdf} to a single image in {output_dir}.")

        return results
