import os
import re
import json
import logging
import shutil
from datetime import datetime

from collections import Counter
import re
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()


def find_transaction_date(boxes):
    # Define regex patterns for common date formats, including single-digit day/month and 2-digit years
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # Matches DD/MM/YYYY or MM/DD/YYYY format
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",  # Matches DD-MM-YYYY or MM-DD-YYYY format
        r"\b\d{4}/\d{1,2}/\d{1,2}\b",  # Matches YYYY/MM/DD format
        r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",  # Matches DD.MM.YYYY format
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # Matches YYYY-MM-DD format
        r"\b\d{1,2}/\d{1,2}/\d{2}\b",  # Matches DD/MM/YY or MM/DD/YY format
        r"\b\d{1,2}-\d{1,2}-\d{2}\b",  # Matches DD-MM-YY or MM-DD-YY format
        r"\b\d{1,2} \w{3} \d{2}\b",  # Matches "27 Jul 24" format
    ]

    # Iterate over the boxes to find the transaction date
    for box in boxes:
        text = box["text"]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group()

                # Try different parsing formats for the matched date
                for fmt in [
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                    "%d-%m-%Y",
                    "%m-%d-%Y",  # Full year formats
                    "%Y/%m/%d",
                    "%Y-%m-%d",
                    "%d.%m.%Y",  # Full year formats with other delimiters
                    "%d/%m/%y",
                    "%m/%d/%y",
                    "%d-%m-%y",
                    "%m-%d-%y",  # 2-digit year formats
                    "%d %b %y",  # "27 Jul 24" format
                ]:
                    try:
                        # Parse the date to the required format YYYY_MM_DD
                        parsed_date = datetime.strptime(date_str, fmt)

                        # If it's a 2-digit year, adjust the century
                        if parsed_date.year < 100:
                            parsed_date = parsed_date.replace(
                                year=parsed_date.year + 2000
                            )

                        return parsed_date.strftime("%Y_%m_%d")
                    except ValueError:
                        continue  # Try the next format if parsing fails

                # Return the date in a basic format if all parsing fails
                return date_str.replace("/", "_").replace("-", "_").replace(".", "_")

    return None  # No valid date found


from collections import Counter
import re


from collections import Counter
import re

from collections import Counter
import re
from collections import Counter
import re

# List of common words to exclude from company names (e.g., stopwords)
STOPWORDS = {
    "totaal",
    "totaalrekening",
    "med",
    "product",
    "btw",
    "bestelling",
    "order",
    "datum",
    "rekening",
    "factuur",
    "bon",
    "factuurdatum",
    "aantal",
    "prijs",
    "bedrag",
    "klant",
    "nummer",
    "subtotaal",
}


# Function to calculate the area of a bounding box
def calculate_box_area(box):
    # A bounding box is defined by four points: [top-left, top-right, bottom-right, bottom-left]
    top_left = box[0]
    bottom_right = box[2]

    width = abs(bottom_right[0] - top_left[0])
    height = abs(bottom_right[1] - top_left[1])

    return width * height


# Function to find company name in OCR data
def find_company_name(boxes):
    phrase_counter = Counter()
    box_areas = []

    # First, calculate the area for each bounding box
    for box in boxes:
        box_area = calculate_box_area(box["box"])
        box_areas.append(box_area)

    # Calculate the median bounding box area to define "large" bounding boxes
    median_box_area = sorted(box_areas)[len(box_areas) // 2]

    # Iterate over the boxes to count the frequency of words/phrases
    for box in boxes:
        text = box["text"]
        confidence = box["confidence"]
        box_area = calculate_box_area(box["box"])

        # Only consider high-confidence boxes
        if confidence > 0.7:
            # Split the text into words/phrases and ignore surrounding punctuation
            words = re.findall(r"\b\w+\b", text)

            # Filter out numbers, single letters, and stopwords
            filtered_words = [
                word
                for word in words
                if not word.isdigit()
                and len(word) > 1
                and word.lower() not in STOPWORDS
            ]

            # Join filtered words into phrases, count phrases longer than one word
            phrase = " ".join(filtered_words)
            if len(phrase.split()) > 1:  # Only count multi-word phrases
                # If the bounding box is larger than the median, give it higher weight
                weight = 2 if box_area > median_box_area else 1
                phrase_counter[phrase] += weight

    # Find the most common phrase
    if phrase_counter:
        most_common_phrase, count = phrase_counter.most_common(1)[0]
        return most_common_phrase

    return None  # No valid company name found


import re


import re


# Function to fully normalize the company name with underscore as a separator
def normalize_company_name(company_name):
    if not company_name:
        return None

    # Remove all non-alphanumeric characters (but keep spaces for now)
    company_name = re.sub(r"[^a-zA-Z0-9\s]", "", company_name)

    # Remove extra spaces, lowercase the entire name, and replace spaces with underscores
    company_name = "_".join(company_name.split()).lower()

    # Return the fully normalized company name
    return company_name


# Function to rename and copy the image file based on extracted company and date
def get_target_filename(image_path, company_name, transaction_date, target_dir):
    # Extract the original file extension
    ext = os.path.splitext(image_path)[1]

    # Generate a new filename based on company name and transaction date
    if company_name and transaction_date:
        normalized_company_name = normalize_company_name(company_name)
        new_filename = f"{transaction_date}-{normalized_company_name}{ext}"  # Adjusted format to YYYY_MM_DD-Company
    else:
        new_filename = f"unkown-date_{os.path.basename(image_path)}"

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    return os.path.join(target_dir, new_filename)


# Function to process all metadata files and rename the corresponding images
def process_metadata(metadata_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Loop through all metadata files
    for metadata_filename in os.listdir(metadata_dir):
        if not metadata_filename.endswith(".json"):
            logger.warning(
                f"Skipping non-JSON file '{metadata_filename}' in metadata directory"
            )
            continue  # Early return for non-JSON files

        metadata_path = os.path.join(metadata_dir, metadata_filename)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get the image file referenced in the metadata
        image_path = metadata.get("original_image")
        if not image_path:
            logger.warning(f"No 'original_image' key found in '{metadata_path}'")
            continue
        if not os.path.exists(image_path):
            logger.warning(f"Image file '{image_path}' not found")
            continue

        # Extract the company name and transaction date
        company_name = find_company_name(metadata.get("boxes"))
        transaction_date = find_transaction_date(metadata.get("boxes"))

        # Generate the new target filename based on company name and date
        target_filename = get_target_filename(
            image_path, company_name, transaction_date, target_dir
        )

        # Copy the image file to the target directory with the new filename
        shutil.copy(image_path, target_filename)
        logger.info(f"Image file '{image_path}' copied to '{target_filename}'")


if __name__ == "__main__":

    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(
        description="Rename and copy image files based on OCR metadata."
    )
    parser.add_argument(
        "metadata_dir", type=str, help="Directory containing the metadata JSON files"
    )
    parser.add_argument(
        "target_dir", type=str, help="Directory to copy the renamed images to"
    )

    args = parser.parse_args()

    # Process the metadata files and copy renamed images to target directory
    process_metadata(args.metadata_dir, args.target_dir)
