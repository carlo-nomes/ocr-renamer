from src.utils.string_utils import sanitize_text
from src.utils.string_utils import sanitize_digit_text


def test_sanitize_text() -> None:
    assert sanitize_text("Hello,  World!") == "hello world"


def test_sanitize_digit_tex() -> None:
    assert sanitize_digit_text(" 12  634, 00 EUR") == "12 634,00EUR"
