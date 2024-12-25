"""
regex_utils.py

This module contains precompiled regular expressions and helper functions for regex operations.
"""

import re

# Precompiled regex patterns for common text patterns
NON_WORD_PATTERN = re.compile(r"[^\w\s]")
DIGIT_ONLY_PATTERN = re.compile(r"^[\W\s\d]+$")

# Precompiled regex patterns for whitespace patterns
MULTI_WHITESPACE_PATTERN = re.compile(r"\s+")
WHITESPACE_BEFORE_NON_DIGIT_PATTERN = re.compile(r"\s+([^\d])")
WHITESPACE_AFTER_NON_DIGIT_PATTERN = re.compile(r"([^\d])\s+")

# Precompiled regex patterns for common data patterns
UUID_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_PATTERN = re.compile(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
DOMAIN_PATTERN = re.compile(r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Split pattern to split text into spaces, alpha characters, digits into groups and special characters individually
SPLIT_PATTERN = re.compile(r"\s+|[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]")


def split_text(text: str) -> list[str]:
    """
    Splits a given text into words, spaces, and special characters while preserving them.
    :param text: The input text to split.
    :return: A list of split components.
    """
    return SPLIT_PATTERN.findall(text)
