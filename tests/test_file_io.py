from typing import List
from pathlib import Path
import pytest

from src.utils.file_io import list_files_in_directory
from src.utils.file_io import read_file
from src.utils.file_io import write_file
from src.utils.file_io import read_lines
from src.utils.file_io import write_lines
from src.utils.file_io import read_json
from src.utils.file_io import write_json


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture to create and return a temporary directory."""
    test_dir_path = tmp_path / "subdir"
    test_dir_path.mkdir()
    return test_dir_path


@pytest.fixture
def test_files(temp_dir: Path) -> List[str]:
    """Fixture to create test files and return their paths."""
    filenames = ["file1.txt", "file2.log", "file3.txt", "subdir/file4.txt", "subdir/file5.log"]
    for filename in filenames:
        file_path = temp_dir / filename
        file_path.parent.mkdir(exist_ok=True)
        file_path.write_text("test")
    return filenames


@pytest.fixture
def test_file(temp_dir: Path) -> Path:
    """Fixture to create a test file and return its path."""
    file_path = temp_dir / "file.txt"
    file_path.write_text("line1\nline2\nline3")
    return file_path


@pytest.fixture
def test_json_file(temp_dir: Path) -> Path:
    """Fixture to create a test file and return its path."""
    file_path = temp_dir / "file.json"
    file_path.write_text('{"key": "value"}')
    return file_path


def test_list_files_in_directory_non_recursive(temp_dir: Path, test_files: List[str]) -> None:
    files = list_files_in_directory(str(temp_dir))
    expected_files = [str(temp_dir / filename) for filename in test_files[:3]]
    assert set(files) == set(expected_files)


def test_list_files_in_directory_recursive(temp_dir: Path, test_files: List[str]) -> None:
    files = list_files_in_directory(str(temp_dir), recursive=True)
    expected_files = [str(temp_dir / filename) for filename in test_files]
    assert set(files) == set(expected_files)


def test_list_files_in_directory_directory_not_found() -> None:
    with pytest.raises(NotADirectoryError):
        list_files_in_directory("non_existent_directory")


@pytest.mark.parametrize(
    "extension, expected_files",
    [
        ([".txt"], ["file1.txt", "file3.txt", "subdir/file4.txt"]),
        ([".log"], ["file2.log", "subdir/file5.log"]),
        ([".md"], []),
    ],
)
def test_list_files_in_directory_with_extensions_recursive(
    temp_dir: Path, test_files: List[str], extension: List[str], expected_files: List[str]
) -> None:
    files = list_files_in_directory(str(temp_dir), recursive=True, extensions=extension)
    expected_files = [str(temp_dir / filename) for filename in expected_files]
    assert set(files) == set(expected_files)


def test_read_file(test_file: Path) -> None:
    content = read_file(str(test_file))
    assert content == "line1\nline2\nline3"


def test_read_file_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        read_file("non_existent_file.txt")


def test_write_file(temp_dir: Path) -> None:
    file_path = temp_dir / "file.txt"
    write_file(str(file_path), "test")
    assert file_path.read_text() == "test"


def test_read_lines(test_file: Path) -> None:
    lines = read_lines(str(test_file))
    assert lines == ["line1", "line2", "line3"]


def test_write_lines(temp_dir: Path) -> None:
    file_path = temp_dir / "file.txt"
    write_lines(str(file_path), ["line1", "line2", "line3"])
    assert file_path.read_text() == "line1\nline2\nline3"


def test_read_json_file(test_json_file: Path) -> None:
    content = read_json(str(test_json_file))
    assert content == {"key": "value"}


def test_write_json_file(temp_dir: Path) -> None:
    file_path = temp_dir / "file.json"
    write_json(str(file_path), {"key": "value"}, indent=False)
    assert file_path.read_text() == '{\n"key": "value"\n}'
