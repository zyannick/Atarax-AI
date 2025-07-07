import pytest
from unittest import mock
from pathlib import Path
from ataraxai.app_logic.modules.rag.parser.pdf_parser import PDFParser
from ataraxai.app_logic.modules.rag.parser.document_base_parser import DocumentChunk

@pytest.fixture
def mock_fitz_document():
    mock_doc = mock.MagicMock()
    mock_doc.page_count = 2
    mock_page1 = mock.MagicMock()
    mock_page1.get_text.return_value = "Page 1 text"
    mock_page2 = mock.MagicMock()
    mock_page2.get_text.return_value = "Page 2 text"
    mock_doc.load_page.side_effect = [mock_page1, mock_page2]
    return mock_doc

@pytest.fixture
def patch_fitz_open(monkeypatch, mock_fitz_document):
    monkeypatch.setattr("fitz.open", lambda path: mock_fitz_document)

@pytest.fixture
def patch_set_base_metadata(monkeypatch):
    monkeypatch.setattr(
        "ataraxai.app_logic.modules.rag.parser.pdf_parser.set_base_metadata",
        lambda path: {"filename": str(path)}
    )

def test_parse_returns_chunks_for_each_page(patch_fitz_open, patch_set_base_metadata):
    parser = PDFParser()
    path = Path("dummy.pdf")
    chunks = parser.parse(path)
    assert len(chunks) == 2
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert chunks[0].content == "Page 1 text"
    assert chunks[1].content == "Page 2 text"
    assert chunks[0].metadata["page"] == 1
    assert chunks[1].metadata["page"] == 2
    assert chunks[0].metadata["filename"] == "dummy.pdf"

def test_parse_skips_empty_pages(patch_set_base_metadata, monkeypatch):
    mock_doc = mock.MagicMock()
    mock_doc.page_count = 3
    mock_page1 = mock.MagicMock()
    mock_page1.get_text.return_value = "  "
    mock_page2 = mock.MagicMock()
    mock_page2.get_text.return_value = "Some text"
    mock_page3 = mock.MagicMock()
    mock_page3.get_text.return_value = ""
    mock_doc.load_page.side_effect = [mock_page1, mock_page2, mock_page3]
    monkeypatch.setattr("fitz.open", lambda path: mock_doc)
    parser = PDFParser()
    path = Path("dummy.pdf")
    chunks = parser.parse(path)
    assert len(chunks) == 1
    assert chunks[0].content == "Some text"
    assert chunks[0].metadata["page"] == 2