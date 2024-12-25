from src.utils.regex_utils import split_text


def test_split_text() -> None:
    splits = split_text("Hello, world! 123")
    assert splits == ["Hello", ",", " ", "world", "!", " ", "123"]
