import unittest

from src.utils.string_utils import sanitize_text
from src.utils.string_utils import sanitize_digit_text


class TestStringUtils(unittest.TestCase):

    def test_sanitize_text_lowercase(self) -> None:
        self.assertEqual(sanitize_text("Hello World"), "hello world")

    def test_sanitize_text_remove_non_word_characters(self) -> None:
        self.assertEqual(sanitize_text("Hello, World!"), "hello world")

    def test_sanitize_text_multiple_whitespace(self) -> None:
        self.assertEqual(sanitize_text("Hello    World"), "hello world")

    def test_sanitize_text_combined(self) -> None:
        self.assertEqual(sanitize_text("  Hello,    World!  "), "hello world")

    def test_sanitize_text_empty_string(self) -> None:
        self.assertEqual(sanitize_text(""), "")

    def test_sanitize_text_only_non_word_characters(self) -> None:
        self.assertEqual(sanitize_text("!!!"), "")

    def test_sanitize_text_lowercase(self) -> None:
        self.assertEqual(sanitize_text("Hello World"), "hello world")

    def test_sanitize_text_remove_non_word_characters(self) -> None:
        self.assertEqual(sanitize_text("Hello, World!"), "hello world")

    def test_sanitize_text_multiple_whitespace(self) -> None:
        self.assertEqual(sanitize_text("Hello    World"), "hello world")

    def test_sanitize_text_combined(self) -> None:
        self.assertEqual(sanitize_text("  Hello,    World!  "), "hello world")

    def test_sanitize_text_empty_string(self) -> None:
        self.assertEqual(sanitize_text(""), "")

    def test_sanitize_text_only_non_word_characters(self) -> None:
        self.assertEqual(sanitize_text("!!!"), "")

    def test_sanitize_digit_text_remove_whitespace(self) -> None:
        self.assertEqual(sanitize_digit_text(" 123 456 "), "123 456")

    def test_sanitize_digit_text_multiple_whitespace(self) -> None:
        self.assertEqual(sanitize_digit_text("  123   456  "), "123 456")

    def test_sanitize_digit_text_empty_string(self) -> None:
        self.assertEqual(sanitize_digit_text(""), "")

    def test_sanitize_digit_text_only_whitespace(self) -> None:
        self.assertEqual(sanitize_digit_text("     "), "")

    def test_sanitize_digit_text_mixed_characters(self) -> None:
        self.assertEqual(sanitize_digit_text(" 123 abc 456 "), "123 abc 456")

    def test_sanitize_text_lowercase(self) -> None:
        self.assertEqual(sanitize_text("Hello World"), "hello world")

    def test_sanitize_text_remove_non_word_characters(self) -> None:
        self.assertEqual(sanitize_text("Hello, World!"), "hello world")

    def test_sanitize_text_multiple_whitespace(self) -> None:
        self.assertEqual(sanitize_text("Hello    World"), "hello world")

    def test_sanitize_text_combined(self) -> None:
        self.assertEqual(sanitize_text("  Hello,    World!  "), "hello world")

    def test_sanitize_text_empty_string(self) -> None:
        self.assertEqual(sanitize_text(""), "")

    def test_sanitize_text_only_non_word_characters(self) -> None:
        self.assertEqual(sanitize_text("!!!"), "")

    def test_sanitize_digit_text_remove_whitespace(self) -> None:
        self.assertEqual(sanitize_digit_text(" 123 456 "), "123 456")

    def test_sanitize_digit_text_multiple_whitespace(self) -> None:
        self.assertEqual(sanitize_digit_text("  123   456  "), "123 456")

    def test_sanitize_digit_text_empty_string(self) -> None:
        self.assertEqual(sanitize_digit_text(""), "")

    def test_sanitize_digit_text_only_whitespace(self) -> None:
        self.assertEqual(sanitize_digit_text("     "), "")

    def test_sanitize_digit_text_mixed_characters(self) -> None:
        self.assertEqual(sanitize_digit_text("10   000,  00 EUR"), "10 000,00EUR")


if __name__ == "__main__":
    unittest.main()
