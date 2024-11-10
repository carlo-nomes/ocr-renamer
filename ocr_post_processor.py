import argparse
import json
import csv
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from Levenshtein import distance as levenshtein

# Constants for configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"

LANGUAGES = ["en", "nl"]


# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
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
        DICTIONARIES[lang] = set(f.read().splitlines())
    logger.info(f"Loaded {len(DICTIONARIES[lang])} words in dictionary for language {lang}")


def preprocess_dictionary(dictionary: set) -> dict:
    """
    Preprocess the dictionary to group words by their lengths.
    :param dictionary: A set of words.
    :return: A dictionary with word lengths as keys and words as values.
    """
    length_groups = {}
    for word in dictionary:
        length = len(word)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(word)
    return length_groups


GROUPED_DICTIONARIES = {lang: preprocess_dictionary(DICTIONARIES[lang]) for lang in LANGUAGES}


WORD_LENGTH_TOLERANCE = 2  # Maximum length difference allowed for matches


def get_candidate_words(token: str) -> list[str]:
    """Find candidate words in the dictionary for a token."""

    # Find candidate words by length
    token_length = len(token)
    candidates = []
    for length in range(token_length - WORD_LENGTH_TOLERANCE, token_length + WORD_LENGTH_TOLERANCE + 1):
        # Check each language
        for lang in LANGUAGES:
            if length in GROUPED_DICTIONARIES[lang]:
                candidates.extend(GROUPED_DICTIONARIES[lang][length])

    return candidates


def sanitize_text(text: str) -> str:
    """Sanitize the text for processing."""
    # Make the text lowercase
    text = text.lower()
    # Replace non-word characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text


DISTANCE_THRESHOLD = 4  # Maximum distance for a match to be considered valid


def get_all_splits(token: str) -> list[list[str]]:
    """
    Generate all possible split combinations for a token.
    :param token: The string to split.
    :return: A list of all possible split combinations.
    """
    # Split the token by non-word characters
    splits = re.split(r"\W+", token)
    if len(splits) == 1:
        return [[token]]  # Only one way to split a single word

    all_splits = []
    # Include the original unsplit token
    all_splits.append([token])

    # Generate all combinations of indices to split at
    for num_splits in range(1, len(splits)):
        for split_indices in combinations(range(1, len(splits)), num_splits):
            split_result = []
            prev_index = 0
            for index in split_indices:
                split_result.append(" ".join(splits[prev_index:index]))
                prev_index = index
            split_result.append(" ".join(splits[prev_index:]))
            all_splits.append(split_result)

    return all_splits


def find_best_match(alternatives: list[str]) -> str:
    """Find the best match for a text token in the dictionary."""

    best_match_text = None
    lowest_total_distance = float("inf")
    char_count_difference = float("inf")  # Check this if the total distance is equal

    all_splits = []
    for alternative in alternatives:
        alternative_splits = get_all_splits(alternative)
        all_splits.extend(alternative_splits)

    for split_combination in all_splits:
        split_distance_sum = 0
        split_candidates = []
        for split in split_combination:
            # If the split consists of only digits or whitespace, or is too short, add it to the list and update the total distance
            if re.match(r"^[\d\s]+$", split) or len(split) < 3:
                split_candidates.append(split)
                split_distance_sum += len(split)
                continue

            # Find candidate words in the dictionary
            candidates = get_candidate_words(split)
            # logger.debug(f"Found {len(candidates)} candidates for '{sanitized_split}'")

            # Find the best match for the split in the dictionary, based on Levenshtein distance and length difference
            best_candidate = None
            best_candidate_distance = float("inf")
            best_candidate_char_count_difference = float("inf")
            for candidate in candidates:
                distance = levenshtein(split, candidate)
                char_count_difference = abs(len(split) - len(candidate))

                # If the distance is higher than the threshold, skip this candidate
                if distance > best_candidate_distance:
                    continue

                # If the distance is larger than the threshold, skip this candidate
                if distance > DISTANCE_THRESHOLD:
                    continue

                # If the distance is lower than the current best match, update the best match
                if distance < best_candidate_distance:
                    best_candidate = candidate
                    best_candidate_distance = distance
                    best_candidate_char_count_difference = char_count_difference
                    continue

                # If the distance is equal, check the length difference
                if char_count_difference < best_candidate_char_count_difference:
                    best_candidate = candidate
                    best_candidate_distance = distance
                    best_candidate_char_count_difference = char_count_difference
                    continue

            # If no candidate was found, add the split to the list and update the total distance
            if not best_candidate:
                split_candidates.append(split)
                split_distance_sum += len(split)
                continue

            # Add the best candidate to the list and update the total distance
            split_candidates.append(best_candidate)
            split_distance_sum += best_candidate_distance

        split_char_count_difference = abs(len(alternatives) - len(" ".join(split_candidates)))

        # If the total distance is higher than the current best match, skip this split combination
        if split_distance_sum > lowest_total_distance:
            continue

        # If the total distance is lower than the current best match, update the best match
        if split_distance_sum < lowest_total_distance:
            best_match_text = " ".join(split_candidates)
            lowest_total_distance = split_distance_sum
            char_count_difference = split_char_count_difference
            continue

        # If the total distance is equal, check the length difference
        if split_char_count_difference < char_count_difference:
            best_match_text = " ".join(split_candidates)
            lowest_total_distance = split_distance_sum
            char_count_difference = split_char_count_difference
            continue

    return best_match_text


def find_ocr_mistakes(token: str) -> list[str]:
    """Clean up OCR recognition mistakes in text."""

    # Apply substitutions to the text
    alternatives = [token]

    # Apply OCR correction map
    for pattern, substitution in OCR_CORRECTION_MAP.items():
        # Search for the pattern in the token
        match = re.search(pattern, token)
        if not match:
            continue

        # Replace the pattern with the substitution
        mapped = re.sub(pattern, substitution, token)
        alternatives.append(mapped)

        # Recursively apply OCR correction to the new token
        # children = find_ocr_mistakes(mapped)
        # alternatives.extend(children)

    return alternatives


MAX_CONCURRENT_WORKERS = 4


def text_normalization(text: str) -> str:
    # Skip text normalization if the text is empty
    if not text:
        return text

    # Skip text normalization if the text consists of only digits or whitespace, or is too short
    if re.match(r"^[\d\s]+$", text) or len(text) < 3:
        return text

    # Correct common OCR mistakes (give a list of alternatives)
    ocr_alternatives = find_ocr_mistakes(text)

    # Sanitize the text for processing
    ocr_alternatives = [sanitize_text(alt) for alt in ocr_alternatives]

    # Remove duplicates
    ocr_alternatives = list(set(ocr_alternatives))

    # Remove empty strings
    ocr_alternatives = [alt for alt in ocr_alternatives if alt]

    # Perform fuzzy search on the dictionary
    normalized_text = find_best_match(ocr_alternatives)
    logger.debug(f"Normalized text: '{normalized_text}'")

    return normalized_text


def process_metadata(metadata_file: str, output_path: str):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    logger.info(f"Processing metadata file: {metadata_file}")

    for box in metadata["boxes"]:
        if "text" in box and type(box["text"]) == str:
            box["normalized_text"] = text_normalization(box["text"])
        else:
            raise ValueError(f"Text box in metadata is not a string: {box}")

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
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
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
