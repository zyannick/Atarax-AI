import tempfile
from pathlib import Path
from docx import Document as DocxDocument
import pytest
from ataraxai.app_logic.modules.rag.parser.docx_parser import DOCXParser
from ataraxai.app_logic.modules.rag.parser.document_base_parser import DocumentChunk


def create_temp_docx(paragraphs):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc = DocxDocument()
    for para in paragraphs:
        doc.add_paragraph(para)
    doc.save(tmp.name)
    tmp.close()
    return Path(tmp.name)


def test_parse_returns_chunks_for_non_empty_paragraphs():
    paragraphs = ["First paragraph.", "", "Second paragraph.", "   ", "Third."]
    docx_path = create_temp_docx(paragraphs)
    parser = DOCXParser()
    chunks = parser.parse(docx_path)

    expected_texts = ["First paragraph.", "Second paragraph.", "Third."]
    assert len(chunks) == len(expected_texts)
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.content == expected_texts[i]
        assert chunk.source == str(docx_path)
        assert chunk.metadata["type"] == "paragraph"
        assert chunk.metadata["index"] == i


def test_parse_empty_docx_returns_empty_list():
    docx_path = create_temp_docx([])
    parser = DOCXParser()
    chunks = parser.parse(docx_path)
    assert chunks == []


def test_parse_docx_with_only_empty_paragraphs_returns_empty_list():
    docx_path = create_temp_docx(["", "   ", "\n"])
    parser = DOCXParser()
    chunks = parser.parse(docx_path)
    assert chunks == []


def test_parse_preserves_paragraph_order_and_index():
    paragraphs = ["Alpha", "Beta", "Gamma"]
    docx_path = create_temp_docx(paragraphs)
    parser = DOCXParser()
    chunks = parser.parse(docx_path)
    for i, chunk in enumerate(chunks):
        assert chunk.content == paragraphs[i]
        assert chunk.metadata["index"] == i
