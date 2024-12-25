import os
import logging
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from nltk.metrics.distance import edit_distance

from src.utils.file_io import read_lines

# Configure logging
logger = logging.getLogger(__name__)

# Default constants
DEFAULT_DISTANCE_THRESHOLD = 4
DICTIONARY_EXTENSION = ".txt"


class DictionaryManager:
    """
    Manages dictionary loading, preprocessing, and lookups.
    """

    def __init__(self, edit_distance: int = DEFAULT_DISTANCE_THRESHOLD) -> None:
        """
        Initialize the dictionary manager.
        :param edit_distance: Maximum edit distance for candidate matches.
        """
        self.edit_distance_threshold = edit_distance
        self.grouped_dictionaries = {}

    def load_dictionary(self, dictionary_path: Path) -> dict[int, list[str]]:
        """
        Load a dictionary file and preprocess it into length-based groups.
        :param file_path: Path to the dictionary file.
        :return: Grouped dictionary by word length.
        """
        words = read_lines(dictionary_path)
        grouped = defaultdict(list)
        for word in words:
            grouped[len(word)].append(word)

        logger.info(f"Loaded {len(words)} words from {dictionary_path}")
        return dict(grouped)

    def add_dictionary(self, key: str, dictionary_path: Path) -> None:
        """
        Load a dictionary for a specific language and add it to the manager.
        :param lang: Language identifier (e.g., "en", "nl").
        :param dictionary_path: Path to the dictionary file.
        """
        self.grouped_dictionaries[key] = self.load_dictionary(dictionary_path)
        logger.info(f"Added dictionary for '{key}'")

    @lru_cache()
    def get_ranked_candidates(self, token: str, key: str) -> list[str]:
        """
        Fetch candidate terms from the dictionary based on the token length.
        :param token: The token to search for.
        :param lang: Language identifier.
        :return: List of candidate terms from the dictionary.
        """
        if key not in self.grouped_dictionaries:
            raise ValueError(f"Language '{key}' not loaded.")

        token_length = len(token)
        token_complexity = len(set(token))

        def rank_candidate(candidate: str) -> tuple[int, int, int]:
            """
            Rank the candidate based on edit distance, length difference, and complexity difference.
            :param candidate: Candidate term from the dictionary.
            :return: Tuple of edit distance, length difference, and complexity difference.
            """
            distance = edit_distance(token, candidate)
            length_diff = abs(token_length - len(candidate))
            complexity_diff = abs(token_complexity - len(set(candidate)))
            return distance, length_diff, complexity_diff

        # Fetch candidates within the edit distance threshold
        range_start = max(0, token_length - self.edit_distance_threshold)
        range_end = token_length + self.edit_distance_threshold + 1

        candidates: list[tuple[str, int, int, int]] = []
        for length in range(range_start, range_end):
            for candidate in self.grouped_dictionaries[key].get(length, []):
                edit_dist, len_diff, comp_diff = rank_candidate(candidate)
                # Skip candidates with edit distance above the threshold
                if edit_dist > self.edit_distance_threshold:
                    continue
                candidates.append((candidate, edit_dist, len_diff, comp_diff))

        # Sort candidates by edit distance, length difference, and complexity difference
        candidates.sort(key=lambda x: (x[1], x[2], x[3]))

        # Return only the candidate terms
        return [candidate for candidate, *_ in candidates]

    @lru_cache()
    def find_best_match(self, token: str, key: str) -> str:
        """
        Find the best matching term in the dictionary for a given token.
        :param token: The token to search for.
        :param lang: Language identifier.
        :return: Best matching term from the dictionary.
        """

        candidates = self.get_ranked_candidates(token, key)
        return candidates[0] if candidates else None
