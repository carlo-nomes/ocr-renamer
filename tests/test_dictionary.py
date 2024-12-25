from pathlib import Path
import pytest
from src.utils.dictionary import DictionaryManager


@pytest.fixture
def dictionary_manager() -> DictionaryManager:
    return DictionaryManager()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture to create and return a temporary directory."""
    test_dir_path = tmp_path / "subdir"
    test_dir_path.mkdir()
    return test_dir_path


@pytest.fixture
def tmp_dictionary_file(temp_dir: Path) -> Path:
    """Fixture to create and return a temporary dictionary file."""
    test_dict_path = temp_dir / "test_dict.txt"
    with open(test_dict_path, "w") as file:
        file.write("hello\n wOrld\n# comment\npython\n")
    return test_dict_path


def test_add_dictionary(tmp_dictionary_file: Path, dictionary_manager: DictionaryManager) -> None:
    dictionary_manager.add_dictionary(key="test", dictionary_path=tmp_dictionary_file)
    assert "test" in dictionary_manager.grouped_dictionaries
    assert dictionary_manager.grouped_dictionaries["test"] == {5: ["hello", "world"], 6: ["python"]}


def test_get_ranked_candidates(tmp_dictionary_file: Path, dictionary_manager: DictionaryManager) -> None:
    dictionary_manager.add_dictionary(key="test", dictionary_path=tmp_dictionary_file)
    candidates = dictionary_manager.get_ranked_candidates("test", "world")
    assert candidates == ["world", "hello"]
    candidates = dictionary_manager.get_ranked_candidates("test", "pythn")
    assert candidates == ["python"]


def test_find_best_match(tmp_dictionary_file: Path, dictionary_manager: DictionaryManager) -> None:
    dictionary_manager.add_dictionary(key="test", dictionary_path=tmp_dictionary_file)
    best_match = dictionary_manager.find_best_match("test", "pythn")
    assert best_match == "python"
    best_match = dictionary_manager.find_best_match("test", "wOrld")
    assert best_match == "world"
    best_match = dictionary_manager.find_best_match("test", "helo")
    assert best_match == "hello"
