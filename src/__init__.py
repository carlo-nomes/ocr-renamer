"""
OCR Package
-----------

This package contains core logic for processing OCR data using various tools
and utilities for handling text, dictionaries, and more.
"""

import logging
from .constants import LANGUAGES


# Configure the logger for the OCR package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a default null handler to the logger to avoid "No handler found" warnings
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__all__ = [
    "LANGUAGES",  # Export the LANGUAGES constant for package-wide use
]
