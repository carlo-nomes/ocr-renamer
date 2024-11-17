import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import tempfile

import spacy

# Configure logging
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

# Load spaCy models
SPACY_LANGUAGE_MAP = {"en": "en_core_web_sm", "nl": "nl_core_news_sm"}

NLP_ENGINES = {lang: spacy.load(model) for lang, model in SPACY_LANGUAGE_MAP.items()}
logger.info(f"Loaded spaCy models for languages: {', '.join(SPACY_LANGUAGE_MAP.keys())}")


# Configure logging
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()


def extract_entities(text: str, language: str = "en") -> list:
    assert language in SPACY_LANGUAGE_MAP, f"Unsupported language: {language}"
    nlp = NLP_ENGINES[language]
    doc = nlp(text)
    # Filter out entities that are not persons or organizations
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]


def compose_normalized_text(boxes: list) -> str:
    assert all(type(box) == dict for box in boxes), "Boxes must be a list of dictionaries"
    assert all("normalized_text" in box for box in boxes), "Boxes must contain 'normalized_text' key"
    assert all(type(box["normalized_text"]) == str for box in boxes), "Boxes 'normalized_text' values must be strings"
    return " ".join(box["normalized_text"] for box in boxes)


def compose_text(boxes: list) -> str:
    assert all(type(box) == dict for box in boxes), "Boxes must be a list of dictionaries"
    assert all("text" in box for box in boxes), "Boxes must contain 'text' key"
    assert all(type(box["text"]) == str for box in boxes), "Boxes 'text' values must be strings"
    return " ".join(box["text"] for box in boxes)


# Precompile regex patterns
NON_WORD_PATTERN = re.compile(r"[^\w\s]")
MULTI_WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_text(text: str) -> str:
    text = NON_WORD_PATTERN.sub(" ", text.lower())
    text = MULTI_WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def process_metadata(metadata_file: str, output_path: str):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    logger.info(f"Processing metadata file: {metadata_file}")

    assert "boxes" in metadata, f"Metadata file does not contain 'boxes' key: {metadata_file}"
    assert type(metadata["boxes"]) == list, f"Metadata file 'boxes' key is not a list: {metadata_file}"
    boxes = metadata["boxes"]

    # Add recognized entities to metadata
    all_entities: list[dict] = []

    # Extract entities from complete normalized text
    normalized_text = compose_normalized_text(boxes)
    en_entities = extract_entities(normalized_text, "en")
    nl_entities = extract_entities(normalized_text, "nl")
    all_entities.extend(en_entities + nl_entities)

    # Extract entities from complete text
    text = compose_text(boxes)
    en_entities = extract_entities(text, "en")
    nl_entities = extract_entities(text, "nl")
    all_entities.extend(en_entities + nl_entities)

    # Extract entities from individual boxes
    for box in boxes:
        if "normalized_text" in box:
            en_entities = extract_entities(box["normalized_text"], "en")
            nl_entities = extract_entities(box["normalized_text"], "nl")
            # Add recognized entities to metadata
            merged_entities = en_entities + nl_entities
            box["entities"] = merged_entities
            all_entities.extend(merged_entities)

    # Rank entities by frequency
    entity_counts = {}
    for ent in all_entities:
        sanitized_text = sanitize_text(ent["text"])
        if sanitized_text not in entity_counts:
            entity_counts[sanitized_text] = 0
        entity_counts[sanitized_text] += 1
    ranked_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    metadata["entities"] = [{"text": text, "count": count} for text, count in ranked_entities]

    # Save metadata to output directory
    output_file = os.path.join(output_path, os.path.basename(metadata_file))
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved processed metadata to: {output_file}")


MAX_CONCURRENT_WORKERS = 4


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
    for metadata_file in input_metadata:
        process_metadata(metadata_file, output_path)

    # Process metadata files concurrently
    # max_workers = min(MAX_CONCURRENT_WORKERS, len(input_metadata))
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     for metadata_file in input_metadata:
    #         executor.submit(process_metadata, metadata_file, output_path)

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
