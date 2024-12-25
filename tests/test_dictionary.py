import pytest
from unittest.mock import patch, mock_open
from src.utils.dictionary import DictionaryManager


@pytest.fixture
def manager():
    return DictionaryManager()


@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="apple\nbanana\ncherry\n")
def test_load_dictionary(mock_open, mock_exists, manager: DictionaryManager):
    result = manager.load_dictionary("dummy_path")
    expected = {5: ["apple"], 6: ["banana", "cherry"]}
    assert result == expected


@patch("os.path.exists", return_value=False)
def test_load_dictionary_file_not_found(mock_exists, manager: DictionaryManager) -> None:
    with pytest.raises(FileNotFoundError):
        manager.load_dictionary("dummy_path")


@patch.object(DictionaryManager, "load_dictionary", return_value={5: ["apple"], 6: ["banana", "cherry"]})
def test_add_dictionary(mock_load_dictionary, manager: DictionaryManager) -> None:
    manager.add_dictionary("en", "dummy_path")
    assert "en" in manager.grouped_dictionaries
    assert manager.grouped_dictionaries["en"] == {5: ["apple"], 6: ["banana", "cherry"]}


def test_get_ranked_candidates(manager: DictionaryManager) -> None:
    manager.grouped_dictionaries = {"en": {5: ["apple", "apoll"], 6: ["bapple", "banana", "cherry"], 9: ["pineapple"]}}
    candidates = manager.get_ranked_candidates("app", "en")
    assert candidates[0] == "apple"  # Best match
    assert "apoll" in candidates  # Within the max edit distance
    assert "bapple" in candidates  # Within the max edit distance
    assert "pineapple" not in candidates  # Exceeds the max edit distance


def test_get_candidate_words_language_not_loaded(manager: DictionaryManager) -> None:
    with pytest.raises(ValueError):
        manager.get_ranked_candidates("app", "en")


@patch("nltk.metrics.distance.edit_distance", side_effect=lambda x, y: abs(len(x) - len(y)))
def test_find_best_match(mock_edit_distance, manager: DictionaryManager) -> None:
    manager.grouped_dictionaries = {"en": {5: ["apple"], 6: ["banana", "cherry"], 7: ["another"]}}
    best_match = manager.find_best_match("appl", "en")
    assert best_match == "apple"


@patch("nltk.metrics.distance.edit_distance", side_effect=lambda x, y: abs(len(x) - len(y)))
def test_find_best_match_no_valid_match(mock_edit_distance, manager: DictionaryManager) -> None:
    manager.grouped_dictionaries = {"en": {5: ["apple"], 6: ["banana", "cherry"], 7: ["another"]}}
    best_match = manager.find_best_match("xyz", "en")
    assert best_match is None
