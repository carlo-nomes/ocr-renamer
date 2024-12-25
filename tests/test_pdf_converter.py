from pathlib import Path
import pytest
from src.processors.pdf_converter import PDFConverter


@pytest.fixture
def pdf_path() -> Path:
    """Fixture to return the path to a PDF file."""
    return Path("tests/fixtures/sample.pdf")


@pytest.fixture
def tmp_pdf_dir(tmp_path: Path) -> Path:
    """Fixture to create and return a temporary directory with PDF files."""
    pdf_dir = tmp_path / "pdf_dir"
    pdf_dir.mkdir()
    return pdf_dir


@pytest.fixture
def pdf_converter(tmp_pdf_dir: Path) -> PDFConverter:
    """Fixture to return a PDFConverter instance."""
    return PDFConverter(output_dir=tmp_pdf_dir)


def util_open_output_dir(pdf_converter: PDFConverter) -> None:
    """Open the output directory for manual inspection."""
    import webbrowser

    webbrowser.open(pdf_converter.output_dir.as_uri())


def test_pdf_to_single_image(pdf_converter: PDFConverter, pdf_path: Path) -> None:
    """Test converting a PDF file to a single image."""
    images = pdf_converter.convert(pdf_path, single_image=True)
    assert len(images) == 1
    # util_open_output_dir(pdf_converter)


def test_pdf_to_multiple_images(pdf_converter: PDFConverter, pdf_path: Path) -> None:
    """Test converting a PDF file to multiple images."""
    images = pdf_converter.convert(pdf_path, single_image=False)
    assert len(images) == 2
    # util_open_output_dir(pdf_converter)


def test_invalid_pdf_path(pdf_converter: PDFConverter) -> None:
    """Test converting an invalid PDF path."""
    with pytest.raises(FileNotFoundError):
        pdf_converter.convert(Path("invalid.pdf"))
