import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def list_files_in_directory(directory: Path, recursive: bool = False, extensions: list[str] | None = None) -> list[Path]:
    if not directory.exists():
        raise NotADirectoryError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"The path is not a directory: {directory}")

    files = []
    for root, _, filenames in os.walk(str(directory), topdown=True):
        if not recursive and root != str(directory):
            continue
        for filename in filenames:
            if extensions and not any(filename.endswith(ext) for ext in extensions):
                continue
            files.append(Path(root) / filename)
    return files


from pathlib import Path


def empty_directory(directory: Path) -> None:
    """
    Empty a directory of all files and subdirectories.
    """
    for item in directory.iterdir():
        if item.is_dir():
            empty_directory(item)
            item.rmdir()  # Remove the directory after it has been emptied
        else:
            item.unlink()  # Remove the file

    logger.debug(f"Emptied directory: {directory}")


def read_file(file_path: Path, mode: str = "r", encoding: str = "utf-8") -> str:
    """
    Read the contents of a file.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open(mode, encoding=encoding) as f:
        contents = f.read()

    logger.debug(f"Read {len(contents)} characters from {file_path}")
    return contents


def write_file(file_path: Path, contents: str, mode: str = "w", encoding: str = "utf-8") -> None:
    """
    Write contents to a file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open(mode, encoding=encoding) as f:
        f.write(contents)

    logger.debug(f"Wrote {len(contents)} characters to {file_path}")


def read_lines(file_path: Path) -> List[str]:
    """
    Read the lines of a file.
    """
    lines = read_file(file_path).splitlines()
    logger.debug(f"Read {len(lines)} lines from {file_path}")
    return lines


def write_lines(file_path: Path, lines: List[str]) -> None:
    """
    Write lines to a file.
    """
    write_file(file_path, "\n".join(lines))
    logger.debug(f"Wrote {len(lines)} lines to {file_path}")


def read_json(file_path: Path) -> Dict[str, Any]:
    """
    Read the contents of a JSON file.
    """
    try:
        data = json.loads(read_file(file_path))
        logger.debug(f"Loaded JSON data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to load JSON data from {file_path}. Reason: {e}")
        raise


def write_json(file_path: Path, data: Dict[str, Any], indent: int | None = None) -> None:
    """
    Write data to a JSON file.
    :param file_path: Path to the JSON file.
    :param data: Data to write.
    :param indent: JSON indentation level or None for no indentation.
    """
    if indent is False:  # If indent is explicitly set to False, change it to None
        indent = None
    contents = json.dumps(data, indent=indent)
    write_file(file_path, contents)
    logger.debug(f"Wrote JSON data to {file_path}")
