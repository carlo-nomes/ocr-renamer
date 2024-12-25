import os
import re
import json
import logging
import sys
from datetime import datetime

# Constants for configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
DATE_RECOGNITION = (
    # 01-12-2024 (zero-padded day and month with 4-digit year separated by hyphen)
    (r"\b\d{2}-\d{2}-\d{4}\b", "%d-%m-%Y"),
    # 01/12/2024 (zero-padded day and month with 4-digit year separated by slash)
    (r"\b\d{2}/\d{2}/\d{4}\b", "%d/%m/%Y"),
    # 1-12-2024 (day and month with 4-digit year separated by hyphen)
    (r"\b\d{1,2}-\d{1,2}-\d{4}\b", "%-d-%-m-%Y"),
    # 1/12/2024 (day and month with 4-digit year separated by slash)
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "%-d/%-m/%Y"),
    # 01-12-24 (zero-padded day and month with 2-digit year separated by hyphen)
    (r"\b\d{2}-\d{2}-\d{2}\b", "%d-%m-%y"),
    # 01/12/24 (zero-padded day and month with 2-digit year separated by slash)
    (r"\b\d{2}/\d{2}/\d{2}\b", "%d/%m/%y"),
    # 1-12-24 (day and month with 2-digit year separated by hyphen)
    (r"\b\d{1,2}-\d{1,2}-\d{2}\b", "%-d-%-m-%y"),
    # 1/12/24 (day and month with 2-digit year separated by slash)
    (r"\b\d{1,2}/\d{1,2}/\d{2}\b", "%-d/%-m/%y"),
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()


def extract_dates(text):
    """Extract dates from a given text using regular expressions."""
    # Extract dates using regular expressions
    dates = []
    for ptrn, fmt in DATE_RECOGNITION:
        matches = re.findall(ptrn, text)
        for match in matches:
            try:
                date_obj = datetime.strptime(match, fmt)
                logger.debug(f"Extracted date: {date_obj}")
                dates.append(date_obj)
            except ValueError:
                logger.warning(f"Unable to parse date {match} with format {fmt}")

    return dates


def get_name_suggestion(text_list: list) -> str:
    """Get a name suggestion based on the OCR results."""
    # Merge the text into a single string
    text_str = " ".join(text_list)

    # Extract dates from the text and pick the most recent one before today
    dates = extract_dates(text_str)
    sorted_dates = sorted(dates)
    today = datetime.now()
    picked_date = None
    for date in reversed(sorted_dates):
        if date <= today:
            picked_date = date
            break
    formatted_date = picked_date.strftime("%Y-%m-%d") if picked_date else "unknown"

    # Generate a name suggestion based on the picked date
    suggestion = f"Ticket_{formatted_date}"
    return suggestion


def main(input_dir):
    """Main function to parse metadata from OCR results."""
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"The directory {input_dir} does not exist.")

    # List all subdirectories in the input directory
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    logger.debug(f"Found {len(subdirs)} subdirectories in {input_dir}")

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        logger.debug(f"Processing subdirectory: {subdir_path}")

        # Check if the OCR results file exists
        ocr_file = os.path.join(subdir_path, "metadata.json")
        if not os.path.isfile(ocr_file):
            logger.warning(f"Metadata file not found in {subdir_path}")
            continue

        # Load the OCR results from the JSON file
        with open(ocr_file, "r") as file:
            ocr_result = json.load(file)

        # Extract text from OCR results
        boxes = ocr_result.get("boxes", [])
        text = [r.get("text", "") for r in boxes]

        # Extract dates from the text
        suggestion = get_name_suggestion(text)
        logger.info(f"Name suggestion for {subdir}: {suggestion}")

        # Add the suggestion to the metadata
        ocr_result["name_suggestion"] = suggestion

        # Save the updated metadata
        with open(ocr_file, "w") as file:
            json.dump(ocr_result, file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python metadata_parser.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    main(input_dir)
