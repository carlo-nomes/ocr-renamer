"""
Constants for the OCR Package
-----------------------------

This module contains constants that are used throughout the OCR application.
"""

# Supported languages
LANGUAGES = ["en", "nl"]

# Default paths
DICTIONARY_PATH = "data/dictionaries"
SUBSTITUTION_FILE = "data/ocr_substitutions.csv"

# File extensions
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff"]
ALLOWED_PDF_EXTENSIONS = [".pdf"]
ALLOWED_TEXT_EXTENSIONS = [".txt"]

# Default configurations
DEFAULT_LANGUAGE = "en"
MAX_WORKERS = 4  # Maximum number of workers for multithreading
DISTANCE_THRESHOLD = 4  # Maximum Levenshtein distance for dictionary matches
WORD_LENGTH_TOLERANCE = 2  # Tolerance for word length differences in candidate lookups

# Regex patterns
UUID_PATTERN = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
URL_PATTERN = r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
DOMAIN_PATTERN = r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# Logging configuration
LOG_FORMAT = "%(levelname)s: %(asctime)s - %(message)s"
LOG_LEVEL = "DEBUG"

# Metadata validation keys
REQUIRED_METADATA_KEYS = ["boxes"]
REQUIRED_BOX_KEYS = ["text"]
TEXT_KEY_TYPE = str
