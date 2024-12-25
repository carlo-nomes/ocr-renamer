"""
string_utils.py

This module contains utility functions for string manipulation.
"""

from .regex_utils import NON_WORD_PATTERN, MULTI_WHITESPACE_PATTERN, WHITESPACE_AFTER_NON_DIGIT_PATTERN, WHITESPACE_BEFORE_NON_DIGIT_PATTERN


def sanitize_text(text: str) -> str:
    """
    Sanitize text by converting to lowercase and removing non-word characters.
    :param text: Input text to sanitize.
    :return: Sanitized text.
    """
    text = NON_WORD_PATTERN.sub(" ", text.lower())
    text = MULTI_WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def sanitize_digit_text(text: str) -> str:
    """
    Sanitize text by removing whitespace around non-digit characters.
    :param text: Input text to sanitize.
    :return: Sanitized text.
    """
    text = WHITESPACE_BEFORE_NON_DIGIT_PATTERN.sub(r"\1", text)
    text = WHITESPACE_AFTER_NON_DIGIT_PATTERN.sub(r"\1", text)
    text = MULTI_WHITESPACE_PATTERN.sub(" ", text.strip())
    return text
