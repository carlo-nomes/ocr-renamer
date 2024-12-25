from pathlib import Path
import pytest

from src.utils.file_io import list_files_in_directory, empty_directory, read_file, write_file, read_lines, write_lines, read_json, write_json


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture to create and return a temporary directory."""
    test_dir_path = tmp_path / "subdir"
    test_dir_path.mkdir()
    return test_dir_path


@pytest.fixture
def test_files(temp_dir: Path) -> list[Path]:
    """Fixture to create test files and return their paths."""
    filenames = ["file1.txt", "file2.log", "file3.txt", "subdir/file4.txt", "subdir/file5.log"]
    files = [temp_dir.joinpath(filename) for filename in filenames]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(file.stem)
    return files


@pytest.fixture
def test_file(temp_dir: Path) -> Path:
    """Fixture to create a test file and return its path."""
    file_path = temp_dir / "file.txt"
    file_path.write_text("line1\nline2\nline3")
    return file_path


@pytest.fixture
def test_json_file(temp_dir: Path) -> Path:
    """Fixture to create a JSON file and return its path."""
    file_path = temp_dir / "file.json"
    file_path.write_text('{"key": "value"}')
    return file_path


def test_list_files_in_directory_non_recursive(temp_dir: Path, test_files: list[Path]) -> None:
    files = list_files_in_directory(temp_dir)
    expected_files = test_files[:3]  # Assuming these do not include files in subdir
    assert set(files) == set(expected_files), "Mismatch in listed files in non-recursive mode."


def test_list_files_in_directory_recursive(temp_dir: Path, test_files: list[Path]) -> None:
    files = list_files_in_directory(temp_dir, recursive=True)
    assert set(files) == set(test_files), "Mismatch in listed files in recursive mode."


def test_list_files_in_directory_directory_not_found() -> None:
    with pytest.raises(NotADirectoryError):
        list_files_in_directory(Path("/non_existent_directory"))


def test_empty_directory(temp_dir: Path, test_files: list[Path]) -> None:
    empty_directory(temp_dir)
    assert not list(temp_dir.iterdir()), "Directory is not empty after emptying."


@pytest.mark.parametrize(
    "extension, expected_filenames",
    [
        ([".txt"], ["file1.txt", "file3.txt", "subdir/file4.txt"]),
        ([".log"], ["file2.log", "subdir/file5.log"]),
        ([".md"], []),
    ],
)
def test_list_files_in_directory_with_extensions_recursive(
    temp_dir: Path, test_files: list[Path], extension: list[str], expected_filenames: list[str]
) -> None:
    files = list_files_in_directory(temp_dir, recursive=True, extensions=extension)
    expected_files = [temp_dir / filename for filename in expected_filenames]
    assert set(files) == set(expected_files), "Files filtered by extension did not match expectations."


def test_read_file(test_file: Path) -> None:
    content = read_file(test_file)
    assert content == "line1\nline2\nline3", "File content mismatch."


def test_read_file_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        read_file(Path("/non_existent_file.txt"))


def test_write_file(temp_dir: Path) -> None:
    file_path = temp_dir / "file.txt"
    write_file(file_path, "test")
    assert file_path.read_text() == "test", "Failed to write text correctly to file."


def test_read_lines(test_file: Path) -> None:
    lines = read_lines(test_file)
    assert lines == ["line1", "line2", "line3"], "Lines read from file do not match expected lines."


def test_write_lines(temp_dir: Path) -> None:
    file_path = temp_dir / "file.txt"
    write_lines(file_path, ["line1", "line2", "line3"])
    assert file_path.read_text() == "line1\nline2\nline3", "Failed to write lines correctly to file."


def test_read_json_file(test_json_file: Path) -> None:
    content = read_json(test_json_file)
    assert content == {"key": "value"}, "JSON content read from file does not match."


def test_write_json_file(temp_dir: Path) -> None:
    file_path = temp_dir / "file.json"
    write_json(file_path, {"key": "value"}, indent=False)
    assert file_path.read_text() == '{"key": "value"}', "JSON content written to file does not match."
