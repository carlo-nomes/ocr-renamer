import json
import os
import tempfile
import unittest
from unittest.mock import patch
from src.utils.file_io import list_files_in_directory
from src.utils.file_io import read_lines
from src.utils.file_io import write_lines
from src.utils.file_io import read_file
from src.utils.file_io import write_file
from src.utils.file_io import read_json
from src.utils.file_io import write_json


class TestListFilesInDirectory(unittest.TestCase):

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.test_dir.name

        # Create some test files
        self.test_files = ["file1.txt", "file2.log", "file3.txt", "subdir/file4.txt", "subdir/file5.log"]

        for file in self.test_files:
            file_path = os.path.join(self.test_dir_path, file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("test")

    def tearDown(self) -> None:
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_list_files_in_directory_non_recursive(self) -> None:
        files = list_files_in_directory(self.test_dir_path)
        expected_files = [
            os.path.join(self.test_dir_path, "file1.txt"),
            os.path.join(self.test_dir_path, "file2.log"),
            os.path.join(self.test_dir_path, "file3.txt"),
        ]
        self.assertCountEqual(files, expected_files)

    def test_list_files_in_directory_recursive(self) -> None:
        files = list_files_in_directory(self.test_dir_path, recursive=True)
        expected_files = [
            os.path.join(self.test_dir_path, "file1.txt"),
            os.path.join(self.test_dir_path, "file2.log"),
            os.path.join(self.test_dir_path, "file3.txt"),
            os.path.join(self.test_dir_path, "subdir/file4.txt"),
            os.path.join(self.test_dir_path, "subdir/file5.log"),
        ]
        self.assertCountEqual(files, expected_files)

    def test_list_files_in_directory_with_extensions(self) -> None:
        files = list_files_in_directory(self.test_dir_path, extensions=[".txt"])
        expected_files = [os.path.join(self.test_dir_path, "file1.txt"), os.path.join(self.test_dir_path, "file3.txt")]
        self.assertCountEqual(files, expected_files)

    def test_list_files_in_directory_with_extensions_recursive(self) -> None:
        files = list_files_in_directory(self.test_dir_path, recursive=True, extensions=[".txt"])
        expected_files = [
            os.path.join(self.test_dir_path, "file1.txt"),
            os.path.join(self.test_dir_path, "file3.txt"),
            os.path.join(self.test_dir_path, "subdir/file4.txt"),
        ]
        self.assertCountEqual(files, expected_files)

    @patch("src.utils.file_io.logger")
    def test_list_files_in_directory_not_a_directory(self, mock_logger) -> None:
        with self.assertRaises(NotADirectoryError):
            list_files_in_directory("not_a_directory")
        mock_logger.error.assert_called_with("Directory not found: not_a_directory")


class TestReadFile(unittest.TestCase):

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.test_dir.name

        # Create a test file
        self.test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        with open(self.test_file_path, "w") as f:
            f.write("test content")

    def tearDown(self) -> None:
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_read_file_success(self) -> None:
        content = read_file(self.test_file_path)
        self.assertEqual(content, "test content")

    def test_read_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_file(os.path.join(self.test_dir_path, "non_existent_file.txt"))

    @patch("src.utils.file_io.logger")
    def test_read_file_not_found_logging(self, mock_logger) -> None:
        with self.assertRaises(FileNotFoundError):
            read_file(os.path.join(self.test_dir_path, "non_existent_file.txt"))
        mock_logger.error.assert_called_with(f"File not found: {os.path.join(self.test_dir_path, 'non_existent_file.txt')}")

    def test_read_file_with_mode(self) -> None:
        content = read_file(self.test_file_path, mode="r")
        self.assertEqual(content, "test content")


class TestWriteFile(unittest.TestCase):

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.test_dir.name

    def tearDown(self) -> None:
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_write_file_success(self) -> None:
        test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        write_file(test_file_path, "test content")
        with open(test_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "test content")

    def test_write_file_append_mode(self) -> None:
        test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        write_file(test_file_path, "test content")
        write_file(test_file_path, " appended", mode="a")
        with open(test_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "test content appended")

    @patch("src.utils.file_io.logger")
    def test_write_file_logging(self, mock_logger) -> None:
        test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        write_file(test_file_path, "test content")
        mock_logger.debug.assert_called_with(f"Wrote {len('test content')} characters to {test_file_path}")

    def test_write_file_create_directories(self) -> None:
        test_file_path = os.path.join(self.test_dir_path, "subdir/test_file.txt")
        write_file(test_file_path, "test content")
        with open(test_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "test content")


class TestReadLines(unittest.TestCase):

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.test_dir.name

        # Create a test file
        self.test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        with open(self.test_file_path, "w") as f:
            f.write("line1\nline2\nline3")

    def tearDown(self) -> None:
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_read_lines_success(self) -> None:
        lines = read_lines(self.test_file_path)
        self.assertEqual(lines, ["line1", "line2", "line3"])

    def test_read_lines_empty_file(self) -> None:
        empty_file_path = os.path.join(self.test_dir_path, "empty_file.txt")
        with open(empty_file_path, "w") as f:
            f.write("")
        lines = read_lines(empty_file_path)
        self.assertEqual(lines, [])

    def test_read_lines_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_lines(os.path.join(self.test_dir_path, "non_existent_file.txt"))

    @patch("src.utils.file_io.logger")
    def test_read_lines_not_found_logging(self, mock_logger) -> None:
        with self.assertRaises(FileNotFoundError):
            read_lines(os.path.join(self.test_dir_path, "non_existent_file.txt"))
        mock_logger.error.assert_called_with(f"File not found: {os.path.join(self.test_dir_path, 'non_existent_file.txt')}")

    @patch("src.utils.file_io.logger")
    def test_read_lines_logging(self, mock_logger) -> None:
        lines = read_lines(self.test_file_path)
        mock_logger.debug.assert_called_with(f"Read {len(lines)} lines from {self.test_file_path}")


class TestWriteLines(unittest.TestCase):

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.test_dir.name

    def tearDown(self) -> None:
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_write_lines_success(self) -> None:
        test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        lines = ["line1", "line2", "line3"]
        write_lines(test_file_path, lines)
        with open(test_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "line1\nline2\nline3")

    def test_write_lines_empty(self) -> None:
        test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        lines = []
        write_lines(test_file_path, lines)
        with open(test_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "")

    @patch("src.utils.file_io.logger")
    def test_write_lines_logging(self, mock_logger) -> None:
        test_file_path = os.path.join(self.test_dir_path, "test_file.txt")
        lines = ["line1", "line2", "line3"]
        write_lines(test_file_path, lines)
        mock_logger.debug.assert_called_with(f"Wrote {len(lines)} lines to {test_file_path}")

    def test_write_lines_create_directories(self) -> None:
        test_file_path = os.path.join(self.test_dir_path, "subdir/test_file.txt")
        lines = ["line1", "line2", "line3"]
        write_lines(test_file_path, lines)
        with open(test_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "line1\nline2\nline3")


class TestReadJson(unittest.TestCase):

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.test_dir.name

        # Create a test JSON file
        self.test_json_path = os.path.join(self.test_dir_path, "test_file.json")
        with open(self.test_json_path, "w") as f:
            json.dump({"key": "value"}, f)

    def tearDown(self) -> None:
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_read_json_success(self) -> None:
        data = read_json(self.test_json_path)
        self.assertEqual(data, {"key": "value"})

    def test_read_json_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_json(os.path.join(self.test_dir_path, "non_existent_file.json"))

    @patch("src.utils.file_io.logger")
    def test_read_json_not_found_logging(self, mock_logger) -> None:
        with self.assertRaises(FileNotFoundError):
            read_json(os.path.join(self.test_dir_path, "non_existent_file.json"))
        mock_logger.error.assert_called_with(f"File not found: {os.path.join(self.test_dir_path, 'non_existent_file.json')}")

    def test_read_json_invalid_json(self) -> None:
        invalid_json_path = os.path.join(self.test_dir_path, "invalid.json")
        with open(invalid_json_path, "w") as f:
            f.write("invalid json")
        with self.assertRaises(json.JSONDecodeError):
            read_json(invalid_json_path)

    @patch("src.utils.file_io.logger")
    def test_read_json_invalid_json_logging(self, mock_logger) -> None:
        invalid_json_path = os.path.join(self.test_dir_path, "invalid.json")
        with open(invalid_json_path, "w") as f:
            f.write("invalid json")
        with self.assertRaises(json.JSONDecodeError):
            read_json(invalid_json_path)
        mock_logger.error.assert_called_with(f"Failed to load JSON data from {invalid_json_path}. Reason: Expecting value: line 1 column 1 (char 0)")


class TestWriteJson(unittest.TestCase):

    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = self.test_dir.name

    def tearDown(self) -> None:
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_write_json_success(self) -> None:
        test_json_path = os.path.join(self.test_dir_path, "test_file.json")
        data = {"key": "value"}
        write_json(test_json_path, data)
        with open(test_json_path, "r") as f:
            content = json.load(f)
        self.assertEqual(content, data)

    def test_write_json_with_indent(self) -> None:
        test_json_path = os.path.join(self.test_dir_path, "test_file.json")
        data = {"key": "value"}
        write_json(test_json_path, data, indent=4)
        with open(test_json_path, "r") as f:
            content = json.load(f)
        self.assertEqual(content, data)

    @patch("src.utils.file_io.logger")
    def test_write_json_logging(self, mock_logger) -> None:
        test_json_path = os.path.join(self.test_dir_path, "test_file.json")
        data = {"key": "value"}
        write_json(test_json_path, data)
        mock_logger.debug.assert_called_with(f"Wrote JSON data to {test_json_path}")

    def test_write_json_create_directories(self) -> None:
        test_json_path = os.path.join(self.test_dir_path, "subdir/test_file.json")
        data = {"key": "value"}
        write_json(test_json_path, data)
        with open(test_json_path, "r") as f:
            content = json.load(f)
        self.assertEqual(content, data)


if __name__ == "__main__":
    unittest.main()
