import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def list_files_in_directory(directory: str, recursive: bool = False, extensions: list[str] | None = None) -> list[str]:
    """
    List files in a directory.
    :param directory: Directory path.
    :param recursive: Recursively list files.
    :param extensions: List of file extensions to filter by.
    :return: List of file paths.
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        raise NotADirectoryError(f"Directory not found: {directory}")

    files = []
    for root, _, filenames in os.walk(directory, topdown=True):
        if not recursive and root != directory:
            continue
        for filename in filenames:
            # Filter by file extensions
            if extensions and not filename.endswith(tuple(extensions)):
                continue
            files.append(os.path.join(root, filename))

    logger.debug(f"Found {len(files)} files in {directory} (recursive={recursive}) with extensions {extensions}")
    return files


def read_file(file_path: str, mode: str = "r") -> str:
    """
    Read the contents of a file.
    :param file_path: Path to the file.
    :param mode: File open mode.
    :return: File contents.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, mode, encoding="utf-8") as f:
        contents = f.read()

    logger.debug(f"Read {len(contents)} characters from {file_path}")
    return contents


def write_file(file_path: str, contents: str, mode: str = "w") -> None:
    """
    Write contents to a file.
    :param file_path: Path to the file.
    :param contents: Contents to write.
    :param mode: File open mode.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode, encoding="utf-8") as f:
        f.write(contents)

    logger.debug(f"Wrote {len(contents)} characters to {file_path}")


def read_lines(file_path: str) -> list[str]:
    """
    Read the lines of a file.
    :param file_path: Path to the file.
    :return: List of lines.
    """
    contents = read_file(file_path)
    lines = contents.splitlines()
    logger.debug(f"Read {len(lines)} lines from {file_path}")
    return lines


def write_lines(file_path: str, lines: list[str]) -> None:
    """
    Write lines to a file.
    :param file_path: Path to the file.
    :param lines: List of lines to write.
    """
    contents = "\n".join(lines)
    write_file(file_path, contents)
    logger.debug(f"Wrote {len(lines)} lines to {file_path}")


def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read the contents of a JSON file.
    :param file_path: Path to the JSON file.
    :return: JSON contents.
    """
    contents = read_file(file_path)

    try:
        data = json.loads(contents)
        logger.debug(f"Loaded JSON data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to load JSON data from {file_path}. Reason: {e}")
        raise e


def write_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Write data to a JSON file.
    :param file_path: Path to the JSON file.
    :param data: Data to write.
    :param indent: JSON indentation level.
    """
    contents = json.dumps(data, indent=indent)
    write_file(file_path, contents)
    logger.debug(f"Wrote JSON data to {file_path}")
