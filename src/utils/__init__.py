"""
Initialization module for the utils package.
Provides access to shared utility functions and modules.
"""

from .dictionary import DictionaryManager
from .file_io import *
from .regex_utils import *
from .string_utils import *

__all__ = [
    "DictionaryManager",
    "read_file",
    "write_file",
    "read_lines",
    "write_lines",
    "read_json",
    "NON_WORD_PATTERN",
    "MULTI_WHITESPACE_PATTERN",
    "DIGIT_ONLY_PATTERN",
    "UUID_PATTERN",
    "EMAIL_PATTERN",
    "URL_PATTERN",
    "DOMAIN_PATTERN",
    "SPLIT_PATTERN",
    "sanitize_text",
    "sanitize_digit_text",
    "split_token_with_special_chars",
]
