import unittest
from src.utils.regex_utils import split_text


class TestRegexUtils(unittest.TestCase):

    def test_split_text_with_words_and_spaces(self) -> None:
        text = "Hello world"
        expected = ["Hello", " ", "world"]
        result = split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_special_characters(self) -> None:
        text = "Hello, world!"
        expected = ["Hello", ",", " ", "world", "!"]
        result = split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_numbers(self) -> None:
        text = "123 456"
        expected = ["123", " ", "456"]
        result = split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_mixed_content(self) -> None:
        text = "Hello, world! 123"
        expected = ["Hello", ",", " ", "world", "!", " ", "123"]
        result = split_text(text)
        self.assertEqual(result, expected)

    def test_split_text_with_only_special_characters(self):
        text = "!@#$%^&*()"
        expected = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"]
        result = split_text(text)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
