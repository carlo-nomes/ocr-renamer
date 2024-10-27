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

# Define the patterns to match different date formats
DATE_PATTERNS = [
    r"\b\d{1,4}([./\-_\\ ])\d{1,2}\1\d{1,4}\b",  # Matches structured dates with separators
    r"\b[a-zA-Z]+[, ]+\d{1,4}[, ]+\d{1,4}\b",  # Matches dates with month names at the beginning
    r"\b\d{1,4}[, ]+[a-zA-Z]+[, ]+\d{1,4}\b",  # Matches dates with month names in the middle
    r"\b\d{1,4}[, ]+\d{1,4}[, ]+[a-zA-Z]+\b",  # Matches dates with month names at the end
]

# List of formats to try for parsing dates in descending order of probability
DATE_FORMATS = [
    # Day first formats
    "%d-%m-%Y",  # Day first '-' format with 4-digit year
    "%d-%m-%y",  # Day first '-' format with 2-digit year
    "%d %b %y",  # Day first with abbreviated month name and 2-digit year
    "%d %b %Y",  # Day first with abbreviated month name and 4-digit year
    "%d %B %y",  # Day first with full month name and 2-digit year
    "%d %B %Y",  # Day first with full month name and 4-digit year
    # Year first formats
    "%Y-%m-%d",  # Year first '-' format with 4-digit year
    "%y-%m-%d",  # Year first '-' format with 2-digit year
    "%b %d %y",  # Year first with abbreviated month name and 2-digit year
    "%b %d %Y",  # Year first with abbreviated month name and 4-digit year
    "%B %d %y",  # Year first with full month name and 2-digit year
    "%B %d %Y",  # Year first with full month name and 4-digit year
    # Month first formats
    "%m-%d-%Y",  # Month first '-' format with 4-digit year
    "%m-%d-%y",  # Month first '-' format with 2-digit year
    "%b %d %Y",  # Month first with abbreviated month name and 4-digit year
    "%B %d %Y",  # Month first with full month name and 4-digit year
    "%b %d %y",  # Month first with abbreviated month name and 2-digit year
    "%B %d %y",  # Month first with full month name and 2-digit year
]
CENTURY_ADJUSTMENT = 2000


def parse_date(date_str: str) -> datetime | None:
    logger.debug(f"Parsing date string: '{date_str}'")
    parsed_date = None

    # Normalize the date string for easier parsing
    normalized_date_str = date_str.lower()
    # Replace separators with common '-'
    normalized_date_str = re.sub(r"[./_\\]", "-", normalized_date_str)
    # Replace ',' with space
    normalized_date_str = re.sub(r",", " ", normalized_date_str)
    # Remove double spaces
    normalized_date_str = re.sub(r"\s+", " ", normalized_date_str)
    # Remove leading/trailing whitespaces
    normalized_date_str = normalized_date_str.strip()

    # Try parsing the date with different formats
    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(normalized_date_str, fmt)
        except ValueError:
            # Skip to the next format if the current one fails
            continue

        # Adjust the parsed year for 2-digit years
        if parsed_date.year < 100:
            parsed_date = parsed_date.replace(year=parsed_date.year + CENTURY_ADJUSTMENT)

        # If date is in the future, try the next format
        if parsed_date > datetime.now():
            parsed_date = None
            continue

        # If the date is longer than 100 years ago, try the next format
        if datetime.now().year - parsed_date.year > 100:
            parsed_date = None
            continue

        # Exit the loop if date is successfully parsed
        break

    return parsed_date


def find_transaction_date(boxes: list[dict]) -> datetime | None:
    possible_transaction_dates: list[datetime] = []

    # Iterate over the boxes to find the transaction date
    for box in boxes:
        text = box["text"]
        logger.debug(f"Checking text '{text}' for a date")

        for pattern in DATE_PATTERNS:
            match = re.search(pattern, text)
            # If no match is found, try the next pattern
            if not match:
                logger.debug(f"No match found for pattern '{pattern}' in '{text}'")
                continue

            # Attempt to parse the date, try the next pattern if parsing fails
            date_str = match.group()
            parsed_date = parse_date(date_str)
            if not parsed_date:
                logger.debug(f"Failed to parse date from '{date_str}'")
                continue

            logger.debug(f"Found date '{parsed_date}' in '{text}'")
            possible_transaction_dates.append(parsed_date)

    # Return None if no valid date is found
    if not possible_transaction_dates:
        return None

    # Return the most recent date found
    return max(possible_transaction_dates)


# Function to calculate the area of a bounding box
# A bounding box is defined by four points: [top-left, top-right, bottom-right, bottom-left]
def calculate_box_area(box) -> float:
    assert isinstance(box, list), "Bounding box must be a list of points"
    assert len(box) == 4, "Bounding box must have 4 points"
    assert all(isinstance(point, list) and len(point) == 2 for point in box), "Each point must be a list of 2 coordinates"
    assert all(isinstance(coord, (int, float)) for point in box for coord in point), "Coordinates must be integers or floats"

    # Extract the top-left and bottom-right points and convert them to floats
    top_left = [float(coord) for coord in box[0]]
    bottom_right = [float(coord) for coord in box[2]]

    # Calculate the width and height of the bounding box
    width = abs(bottom_right[0] - top_left[0])
    height = abs(bottom_right[1] - top_left[1])

    # Calculate the area as the product of width and height
    return width * height


# Calculate the median bounding box area to define "large" bounding boxes
def calculate_median_box_area(boxes: list[dict]) -> float:
    box_areas = []
    for box in boxes:
        box_area = calculate_box_area(box["box"])
        box_areas.append(box_area)

    return sorted(box_areas)[len(box_areas) // 2]


# Normalize the text in the OCR boxes for matching
def normalize_boxes(boxes: list[dict]) -> list[dict]:
    normalize_boxes = []
    for box in boxes:
        # Create a copy of the box to avoid modifying the original
        normalized_box = box.copy()

        # Normalize the text term by term
        terms = re.findall(r"\b\w+\b", normalized_box["text"])
        normalized_terms = []
        for term in terms:
            # Convert term to lowercase
            normalized_term = term.lower()

            # Do not further normalize email addresses, URLs, or domain names
            if re.match(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", normalized_term):
                normalized_terms.append(normalized_term)
                continue
            if re.match(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", normalized_term):
                normalized_terms.append(normalized_term)
                continue
            if re.match(r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", normalized_term):
                normalized_terms.append(normalized_term)
                continue
            if re.match(r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[a-zA-Z0-9.-]+", normalized_term):
                normalized_terms.append(normalized_term)
                continue

            # Remove non-alphanumeric characters
            normalized_term = re.sub(r"[^a-z0-9 ]", "", normalized_term)
            # Remove leading/trailing spaces
            normalized_term = normalized_term.strip()

            # Skip terms with less than 3 characters
            if len(normalized_term) < 3:
                continue

            # Skip terms containing only digits and separators
            if re.match(r"^\d+[\s.,-]*$", normalized_term):
                continue

            # Replace multiple spaces with a single space
            normalized_term = re.sub(r"\s+", " ", normalized_term)
            normalized_terms.append(normalized_term)

        # Overwrite the text with the normalized version
        normalized_box["text"] = " ".join(normalized_terms)
        normalize_boxes.append(normalized_box)

    return normalize_boxes


# List of common words to exclude from company names (e.g., stopwords)
STOPWORDS = [
    # Common Bill terms
    "factuur",
    "invoice",
    "rekening",
    "bill",
    "receipt",
    "order",
    "bestelling",
    "order",
    "orderbevestiging",
    "ticket",
    "bon",
    "voucher",
    "coupon",
    "kassabon",
    # Common VAT terms
    "vat",
    "tax",
    "btw",
    # Common date terms
    "datum",
    "date",
    "factuurdatum",
    "invoice date",
    "periode",
    "period",
    "vervaldatum",
    "due date",
    "betaaldatum",
    "payment date",
    "leverdatum",
    "delivery date",
    "aankoopdatum",
    "purchase date",
    # Common total terms
    "totaal",
    "total",
    "totaalrekening",
    "tegoed",
    "credit",
    "subtotaal",
    "subtotal",
    # Common payment terms
    "betaald",
    "paid",
    "betaling",
    "payment",
    "openstaand",
    "outstanding",
    "saldo",
    "balance",
    "creditnota",
    "credit note",
    # Common Subscription terms
    "abonnement",
    "subscription",
    "contract",
    "overeenkomst",
    "agreement",
    "service",
    "dienst",
    # Common currency terms
    "euro",
    "eur",
    "usd",
    "dollar",
    # Extra terms
    "wifi",
    "restaurant",
    # Personal terms
    "carlo",
    "nomes",
    "secretaris",
    "meyerlei",
]


def contains_stopword(text: str) -> bool:
    for word in STOPWORDS:
        matched = re.search(rf"{word}", text, re.IGNORECASE)
        if matched:
            return True
    return False


def filter_boxes(boxes: list[dict]) -> list[dict]:
    filtered_boxes = []
    for box in boxes:
        text = box["text"]
        confidence = box["confidence"]

        # Skip boxes with low confidence
        if confidence < 0.7:
            continue

        # Skip empty boxes
        if not text:
            continue

        # Skip boxes containing stopwords
        if contains_stopword(text):
            continue

        filtered_boxes.append(box)

    return filtered_boxes


MIN_TERM_LENGTH = 3


def compute_weighted_terms(boxes: list[dict], median_area: float) -> list[dict]:
    term_counter = Counter()

    for box in boxes:
        text = box["text"]
        confidence = box["confidence"]
        box_area = calculate_box_area(box["box"])

        # Start with the confidence as the weight
        weight = confidence

        # Increase the weight based on its relation to the median box area
        weight *= box_area / median_area

        # Split the text into terms
        terms = re.findall(r"\b\w+\b", text)

        # Filter out terms with less than 3 characters
        terms = [term for term in terms if len(term) >= MIN_TERM_LENGTH]

        for term in terms:
            # Check if the term is an email address or url, normalize it and boost its weight
            email_match = re.match(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+)\.[a-zA-Z]{2,}", term)
            if email_match:
                weight *= 2
                term = email_match.group(1)
                term = re.sub(r"[^a-z0-9]", " ", term)
                term = re.sub(r"\s+", " ", term)
            url_match = re.match(r"https?://([a-zA-Z0-9.-]+)\.[a-zA-Z]{2,}", term)
            if url_match:
                weight *= 2
                term = url_match.group(1)
                term = re.sub(r"[^a-z0-9]", " ", term)
                term = re.sub(r"\s+", " ", term)

            # Increase the weight of the term
            term_counter[term] += weight

    return term_counter


SENTENCE_THRESHOLD = 5


# Function to find company name in OCR data
def find_company_name(boxes: list[dict]) -> str | None:
    # Calculate the median box area
    median_box_area = calculate_median_box_area(boxes)

    # Normalize the boxes for matching
    normalized_boxes = normalize_boxes(boxes)

    # Filter out irrelevant boxes
    filtered_boxes = filter_boxes(normalized_boxes)
    if not filtered_boxes:
        return None

    # Assign weights to the filtered boxes
    weighed_terms = compute_weighted_terms(filtered_boxes, median_box_area)

    # Get the highest weighted term
    company_name = max(weighed_terms, key=weighed_terms.get)

    return company_name


# Function to fully normalize the company name with underscore as a separator
def normalize_company_name(company_name: str) -> str:
    # Replace special characters with spaces
    normalized_name = re.sub(r"[^a-zA-Z0-9 ]", " ", company_name)
    # Replace multiple spaces with a single space
    normalized_name = re.sub(r"\s+", " ", normalized_name)
    # Remove leading/trailing spaces
    normalized_name = normalized_name.strip()
    # Replace spaces with underscores
    normalized_name = normalized_name.replace(" ", "_")
    # Convert to lowercase
    normalized_name = normalized_name.lower()

    logger.debug(f"Normalized company name: '{company_name}' -> '{normalized_name}'")
    return normalized_name


# Function to rename and copy the image file based on extracted company and date
def get_target_filename(image_path: str, company_name: str | None, transaction_date: datetime | None, target_dir: str) -> str:
    # Extract the original file extension
    ext = os.path.splitext(image_path)[1]

    # If company name is found, normalize it
    normalized_company_name = normalize_company_name(company_name) if company_name else "unknown_company"

    # If transaction date is found, format it as YYYY_MM_DD
    formatted_date = transaction_date.strftime("%Y_%m_%d") if transaction_date else "unknown_date"

    # Add a timestamp to the filename to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate the new filename with the format: {date}-{company}-{timestamp}{ext}
    new_filename = f"{formatted_date}-{normalized_company_name}-{timestamp}{ext}"

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Return the full path to the new filename
    return os.path.join(target_dir, new_filename)


# List of common OCR errors to correct
COMMON_ERRORS_ONE_WAY = {
    "|": "I",
    "|": "1",
    "|": "l",
}
COMMON_ERRORS_TWO_WAY = {
    "I": "1",
    "l": "1",
    "o": "0",
    "O": "0",
    "s": "5",
    "S": "5",
    "Z": "2",
    "B": "8",
}


def preprocess_boxes(boxes: list[dict]) -> list[dict]:
    # Filter out boxes with empty text
    filtered_boxes = [box for box in boxes if box.get("text")]

    # Correct boxes with common OCR errors (e.g., "I" misread as "|")
    one_way_corrected_boxes = []
    for box in filtered_boxes:
        text = box["text"]
        for error, correction in COMMON_ERRORS_ONE_WAY.items():
            corrected_text = text.replace(error, correction)
            if corrected_text != text:
                corrected_box = box.copy()
                corrected_box["text"] = corrected_text
                one_way_corrected_boxes.append(corrected_box)

    # Duplicate boxes with common OCR errors (e.g., "1" misread as "I" and vice versa)
    two_way_corrected_boxes = []
    for box in filtered_boxes:
        text = box["text"]
        for error, correction in COMMON_ERRORS_TWO_WAY.items():
            corrected_text = text.replace(error, correction)
            if corrected_text != text:
                corrected_box = box.copy()
                corrected_box["text"] = corrected_text
                two_way_corrected_boxes.append(corrected_box)

    # Combine all corrected boxes and return the result
    return filtered_boxes + one_way_corrected_boxes + two_way_corrected_boxes


# Function to process all metadata files and rename the corresponding images
def process_metadata(metadata_dir: str, target_dir: str, clean: bool = False) -> None:
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Clean the target directory if requested
    if clean:
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            os.remove(file_path)
            logger.info(f"Removed file '{file_path}'")

    # Loop through all metadata files
    for metadata_filename in os.listdir(metadata_dir):

        # Skip non-JSON files
        if not metadata_filename.endswith(".json"):
            logger.warning(f"Skipping non-JSON file '{metadata_filename}' in metadata directory")
            continue

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

        # Type check the 'boxes' key in the metadata
        boxes = metadata.get("boxes")
        assert boxes is not None, f"No 'boxes' key found in '{metadata_path}'"
        assert isinstance(boxes, list), f"Invalid 'boxes' data type in '{metadata_path}'"
        assert all(isinstance(box, dict) for box in boxes), f"Invalid 'boxes' data type in '{metadata_path}'"

        # Preprocess the OCR boxes to correct common errors
        cleaned_boxes = preprocess_boxes(boxes)

        # Find the company name and transaction date from the cleaned boxes
        company_name = find_company_name(cleaned_boxes)
        transaction_date = find_transaction_date(cleaned_boxes)

        # Generate the new target filename based on company name and date
        target_filename = get_target_filename(image_path, company_name, transaction_date, target_dir)

        # Copy the image file to the target directory with the new filename
        shutil.copy(image_path, target_filename)
        logger.info(f"Image file '{image_path}' copied to '{target_filename}'")


if __name__ == "__main__":

    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(description="Rename and copy image files based on OCR metadata.")
    parser.add_argument("metadata_dir", type=str, help="Directory containing the metadata JSON files")
    parser.add_argument("target_dir", type=str, help="Directory to copy the renamed images to")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--clean", action="store_true", help="Clean the target directory before copying images")

    args = parser.parse_args()

    # Process the metadata files and copy renamed images to target directory
    process_metadata(args.metadata_dir, args.target_dir, clean=args.clean)
