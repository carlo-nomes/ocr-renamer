# Standard library imports
import argparse
import csv
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations


# Third-party imports
from Levenshtein import distance as levenshtein

# Constants for configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"

LANGUAGES = ["en", "nl", "companies"]


# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


# Load OCR correction map from CSV
SUBSTITUTION_CSV_PATH = f"{os.path.dirname(__file__)}/ocr_substitutions.csv"
OCR_CORRECTION_MAP = {}
with open(SUBSTITUTION_CSV_PATH, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=";", quotechar='"')
    for i, row in enumerate(csv_reader):
        # Skip header
        if i == 0:
            continue

        # Check if row has 2 columns
        if len(row) != 2:
            raise ValueError(f"Invalid row in CSV: {row} on row {i}")

        # Add substitution to dictionary
        pattern, substitution = row

        # Check if pattern and substitution are valid
        try:
            re.sub(pattern, substitution, "test")
        except re.error as e:
            raise ValueError(f"Invalid substitution pattern: {pattern} -> {substitution} on row {i}") from e

        OCR_CORRECTION_MAP[pattern] = substitution
logger.info(f"Loaded {len(OCR_CORRECTION_MAP)} substitutions from CSV: {SUBSTITUTION_CSV_PATH}")

# Load dictionaries for each language
DICTIONARY_PATH = f"{os.path.dirname(__file__)}/dictionaries"
DICTIONARIES = {}
for lang in LANGUAGES:
    dictionary_file = os.path.join(DICTIONARY_PATH, f"{lang}.txt")
    with open(dictionary_file, "r") as f:
        # Filter out empty lines and comments
        dictionary = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    logger.info(f"Loaded {len(DICTIONARIES[lang])} words in dictionary for language {lang}")


def preprocess_dictionary(dictionary: set) -> dict:
    """
    Preprocess the dictionary to group words by their lengths.
    """
    length_groups = defaultdict(list)
    for word in dictionary:
        length_groups[len(word)].append(word)
    return dict(length_groups)


GROUPED_DICTIONARIES = {lang: preprocess_dictionary(DICTIONARIES[lang]) for lang in LANGUAGES}


WORD_LENGTH_TOLERANCE = 2  # Maximum length difference allowed for matches


def get_candidate_words(token: str) -> list[str]:
    """Find candidate words in the dictionary for a token."""

    # Find candidate words by length
    token_length = len(token)
    candidates = []
    # Check each language
    for lang in LANGUAGES:
        for length in range(token_length - WORD_LENGTH_TOLERANCE, token_length + WORD_LENGTH_TOLERANCE + 1):
            if length in GROUPED_DICTIONARIES[lang]:
                candidates.extend(GROUPED_DICTIONARIES[lang][length])

    return candidates


@lru_cache(maxsize=None)
def get_best_candidate(token: str) -> str:
    candidates = get_candidate_words(token)
    best_candidate = None
    best_candidate_distance = float("inf")
    best_candidate_char_count_difference = float("inf")
    for candidate in candidates:
        # Calculate the Levenshtein distance between the split and the candidate in lowercase
        distance = levenshtein(token, candidate.lower())
        char_count_difference = abs(len(token) - len(candidate))

        # If the distance is larger than the threshold, skip this candidate
        if distance > DISTANCE_THRESHOLD:
            continue

        # If the distance is higher than the current best match, skip this candidate
        if distance > best_candidate_distance:
            continue

        # If the distance is lower than the current best match, update the best match
        if distance < best_candidate_distance:
            best_candidate = candidate
            best_candidate_distance = distance
            best_candidate_char_count_difference = char_count_difference
            continue

        # If the distance is equal, check the length difference to the split
        if char_count_difference < best_candidate_char_count_difference:
            best_candidate = candidate
            best_candidate_distance = distance
            best_candidate_char_count_difference = char_count_difference
            continue

    return best_candidate, best_candidate_distance, best_candidate_char_count_difference


# Precompile regex patterns
NON_WORD_PATTERN = re.compile(r"[^\w\s]")
MULTI_WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_text(text: str) -> str:
    """
    Sanitize the text for processing.
    """
    text = NON_WORD_PATTERN.sub(" ", text.lower())
    text = MULTI_WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


# Precompile regex patterns for digit text sanitization
WHITESPACE_BEFORE_NON_DIGIT_PATTERN = re.compile(r"\s+(\D)")
WHITESPACE_AFTER_NON_DIGIT_PATTERN = re.compile(r"(\D)\s+")


def sanitize_digit_text(text: str) -> str:
    """Sanitize text containing only digits."""
    # Remove whitespace before and after non-digit characters
    text = WHITESPACE_BEFORE_NON_DIGIT_PATTERN.sub(r"\1", text)
    text = WHITESPACE_AFTER_NON_DIGIT_PATTERN.sub(r"\1", text)
    text = MULTI_WHITESPACE_PATTERN.sub(" ", text.strip())
    return text


# Precompiled regex pattern to split into words, digits, and special characters
SPLIT_REGEX = re.compile(r"\s+|\w+|\d+|[^\w\s\d]")

# Maximum number of splits to generate
MAX_SPLIT_COUNT = 10000


def get_all_splits(tokens: list[str], max_splits: int = MAX_SPLIT_COUNT) -> list[list[str]]:
    """
    Generate all possible split combinations for a token, retaining special characters and spaces as separate elements.
    """
    all_splits = set()
    for token in tokens:
        # Split the token into words, spaces, and special characters
        splits = SPLIT_REGEX.findall(token)
        if len(splits) == 1:
            return [splits]  # Only one way to split a single element

        n = len(splits)
        # Include original token
        all_splits.add((token,))

        # Generate all combinations of splits
        for r in range(1, n):  # Generate combinations of indices
            for indices in combinations(range(1, n), r):
                result = []
                start = 0
                for index in indices:
                    result.append("".join(splits[start:index]))  # Join segments up to the split point
                    start = index
                result.append("".join(splits[start:]))  # Add the last segment

                # Add the split combination to the set
                all_splits.add(tuple(result))

                # Stop generating splits if the maximum count is reached
                if len(all_splits) > max_splits:
                    logger.warning(f"Exceeded maximum split count of {max_splits}")
                    return list(all_splits)

    return all_splits


# Patterns to skip processing for certain types of text
SKIP_UUID_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
SKIP_DIGITS_ONLY_PATTERN = re.compile(r"^[\W\s\d]+$")
ALPHA_PATTERN = re.compile(r"[a-zA-Z]")

# Precompiled regex patterns for email, URL, and domain text sanitization and extraction
EMAIL_PREP_PATTERN = re.compile(r"[^a-zA-Z0-9._%+-@]")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_PREP_PATTERN = re.compile(r"[^a-zA-Z0-9.-:/]")
URL_PATTERN = re.compile(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
DOMAIN_PREP_PATTERN = re.compile(r"[^a-zA-Z0-9.-:/]")
DOMAIN_PATTERN = re.compile(r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Minimum length for a split to be considered for lookup
LENGTH_LOOKUP_THRESHOLD = 3
# Maximum distance for a match to be considered valid
DISTANCE_THRESHOLD = 4


def find_best_match(original: str, alternatives: list[str]) -> str:
    """Find the best match for a text token in the dictionary."""

    # Skip processing if the original text is empty
    if not original:
        return original

    # Skip processing if the original text does not contain any alphabetic characters
    if SKIP_DIGITS_ONLY_PATTERN.match(original):
        logger.info(f"Skipping digits only: {original}")
        return sanitize_digit_text(original)

    # Skip processing if the original text does not contain enough alphabetic characters
    alpha_count = len(ALPHA_PATTERN.findall(original))
    if alpha_count < LENGTH_LOOKUP_THRESHOLD:
        logger.info(f"Skipping text with less than {LENGTH_LOOKUP_THRESHOLD} alphabetic characters: {original}")
        return sanitize_digit_text(original)

    # Skip processing if the original text is a UUID
    if SKIP_UUID_PATTERN.match(original):
        logger.info(f"Skipping UUID: {original}")
        return original

    best_match_text = None
    lowest_total_distance = float("inf")
    char_count_difference = float("inf")  # Check this if the total distance is equal

    # Generate all possible split combinations for the original text
    all_splits = get_all_splits(alternatives)
    logger.debug(f"Generated {len(all_splits)} split combinations for '{original}'")

    for split_combination in all_splits:
        split_distance_sum = 0
        split_candidates = []
        for split in split_combination:
            # Skip splits that consist of only digits or whitespace or punctuation
            if SKIP_DIGITS_ONLY_PATTERN.match(split):
                split_candidates.append(split)
                split_distance_sum += 0  # Do not penalize digits, whitespace, or punctuation
                logger.debug(f"Skipping split '{split}' with only digits, whitespace, or punctuation")
                continue

            # Skip splits with mostly the same character
            if len(set(split)) < len(split) / 2:
                split_candidates.append(split)
                split_distance_sum += len(split)  # Penalize same character splits
                logger.debug(f"Skipping split '{split}' with mostly the same character")
                continue

            # If the text contains an email use that as the split
            email_prep = EMAIL_PREP_PATTERN.sub("", split.lower())
            email_match = EMAIL_PATTERN.search(email_prep)
            if email_match:
                email = email_match.group()
                split_candidates.append(email)
                split_distance_sum -= len(email)  # Reward emails
                logger.debug(f"Skipping split '{split}' that is an email address")
                continue

            # If the text contains a URL use that as the split
            url_prep = URL_PREP_PATTERN.sub("", split.lower())
            url_match = URL_PATTERN.search(url_prep)
            if url_match:
                url = url_match.group()
                split_candidates.append(url)
                split_distance_sum -= len(url)  # Reward URLs
                logger.debug(f"Skipping split '{split}' that is a URL")
                continue

            # If the text contains a domain use that as the split
            domain_prep = DOMAIN_PREP_PATTERN.sub("", split.lower())
            domain_match = DOMAIN_PATTERN.search(domain_prep)
            if domain_match:
                domain = domain_match.group()
                split_candidates.append(domain)
                split_distance_sum -= len(domain)  # Reward domains
                logger.debug(f"Skipping split '{split}' that is a domain")
                continue

            # Skip splits that are too short
            if len(split) < LENGTH_LOOKUP_THRESHOLD:
                split_candidates.append(split)
                split_distance_sum += len(split)  # Penalize short splits
                logger.debug(f"Skipping split '{split}' that is too short")
                continue

            # Sanitize the split for processing
            best_candidate, best_candidate_distance, _ = get_best_candidate(sanitize_text(split))
            # If no candidate was found, add the split to the list and update the total distance
            if not best_candidate:
                split_candidates.append(split)
                split_distance_sum += len(split)  # Penalize splits that are not in the dictionary
                logger.debug(f"No candidate found for split '{split}'")
                continue

            # Add the best candidate to the list and update the total distance
            split_candidates.append(best_candidate)
            split_distance_sum += best_candidate_distance
            logger.debug(f"Found best candidate '{best_candidate}' for split '{split}'")

        # Calculate the total character count difference for the split combination
        split_match_text = "".join(split_candidates)
        split_char_count_difference = abs(len(original) - len(split_match_text))

        # If the total distance is higher than the current best match, skip this split combination
        if split_distance_sum > lowest_total_distance:
            logger.debug(f"Skipping split combination '{split_candidates}' with distance {split_distance_sum}")
            continue

        # If the total distance is lower than the current best match, update the best match
        if split_distance_sum < lowest_total_distance:
            best_match_text = split_match_text
            lowest_total_distance = split_distance_sum
            char_count_difference = split_char_count_difference
            logger.debug(f"Found new best match '{best_match_text}' with distance {split_distance_sum}")
            continue

        # If the total distance is equal, check the length difference
        if split_char_count_difference < char_count_difference:
            best_match_text = split_match_text
            lowest_total_distance = split_distance_sum
            char_count_difference = split_char_count_difference
            logger.debug(f"Found new best match '{best_match_text}' with distance {split_distance_sum}")
            continue

    return best_match_text


def find_ocr_mistakes(token: str) -> list[str]:
    """
    Clean up OCR recognition mistakes in text and provide alternatives.
    """
    # Use a set to avoid duplicates immediately
    alternatives = {token}

    for pattern, substitution in OCR_CORRECTION_MAP.items():
        if re.search(pattern, token):
            alternatives.add(re.sub(pattern, substitution, token))

    return list(alternatives)


MAX_CONCURRENT_WORKERS = 10


def text_normalization(text: str) -> str:
    """Normalize the text using a dictionary and fuzzy search."""

    # Strip leading and trailing whitespace
    text = text.strip()

    # Skip text normalization if the text is empty
    if not text:
        logger.debug(f"Skipping text normalization for empty text")
        return text

    # Get OCR mistake alternatives
    alternatives = find_ocr_mistakes(text)
    logger.debug(f"Found {len(alternatives)} alternatives for '{text}'")

    # Perform fuzzy search on the dictionary
    normalized_text = find_best_match(text, alternatives)
    logger.info(f"Normalized '{text}' to '{normalized_text}'")

    return normalized_text


def process_metadata(metadata_file: str, output_path: str):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    logger.info(f"Processing metadata file: {metadata_file}")

    assert "boxes" in metadata, f"Metadata file does not contain 'boxes' key: {metadata_file}"
    assert type(metadata["boxes"]) == list, f"Metadata file 'boxes' key is not a list: {metadata_file}"
    assert all(type(box) == dict for box in metadata["boxes"]), f"Metadata file 'boxes' key contains non-dictionary elements: {metadata_file}"
    assert all("text" in box for box in metadata["boxes"]), f"Metadata file 'boxes' key contains elements without 'text' key: {metadata_file}"
    assert all(
        type(box["text"]) == str for box in metadata["boxes"]
    ), f"Metadata file 'boxes' key contains elements with non-string 'text' key: {metadata_file}"

    # Normalize the text in each box
    for box in metadata["boxes"]:
        box["normalized_text"] = text_normalization(box["text"])
    logger.info(f"Normalized text in {len(metadata['boxes'])} boxes")

    # Save the processed text to a new metadata file
    output_metadata_file = os.path.join(output_path, os.path.basename(metadata_file))
    with open(output_metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved processed metadata to: {output_metadata_file}")


def main(input_path: str, output_path: str = tempfile.mkdtemp(), clean: bool = False):
    # Create the output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Clean the output directory if specified
    if clean:
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
        logger.info(f"Cleaned output directory: {output_path}")

    # Load input metadata
    input_metadata = []
    if os.path.isfile(input_path):
        input_metadata.append(input_path)
    elif os.path.isdir(input_path):
        for f in os.listdir(input_path):
            # Check if file ends with "metadata.json"
            if f.endswith("metadata.json"):
                input_metadata.append(os.path.join(input_path, f))
    logger.info(f"Loaded {len(input_metadata)} metadata files from {input_path}")

    # Process metadata files
    # for metadata_file in input_metadata:
    #     process_metadata(metadata_file, output_path)

    # Process metadata files concurrently
    max_workers = min(MAX_CONCURRENT_WORKERS, len(input_metadata))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for metadata_file in input_metadata:
            executor.submit(process_metadata, metadata_file, output_path)

    logger.info("Finished processing metadata files")


if __name__ == "__main__":
    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(description="Perform OCR on images and save results in separate directories.")
    parser.add_argument("input_dir", type=str, help="Directory containing images to process.")
    parser.add_argument("output_dir", type=str, nargs="?", default=None, help="Directory to save processed images and metadata.")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before processing.")
    args = parser.parse_args()

    # Process the specified directories
    main(args.input_dir, args.output_dir, args.clean)
