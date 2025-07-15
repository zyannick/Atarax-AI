import tempfile
from pathlib import Path
from docx import Document as DocxDocument
import pytest
from ataraxai.praxis.modules.rag.parser.docx_parser import DOCXParser
from ataraxai.praxis.modules.rag.parser.document_base_parser import DocumentChunk


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
        
        
        
def test_parse_handles_unicode_and_special_characters():
    paragraphs = ["Hello, world!", "Café naïve résumé.", "中文段落", "Emoji 🚀🔥"]
    docx_path = create_temp_docx(paragraphs)
    parser = DOCXParser()
    chunks = parser.parse(docx_path)
    assert len(chunks) == len(paragraphs)
    for i, chunk in enumerate(chunks):
        assert chunk.content == paragraphs[i]
        assert chunk.metadata["index"] == i

def test_parse_handles_long_paragraphs():
    long_text = "A" * 10000
    paragraphs = [long_text, "Short one."]
    docx_path = create_temp_docx(paragraphs)
    parser = DOCXParser()
    chunks = parser.parse(docx_path)
    assert chunks[0].content == long_text
    assert chunks[1].content == "Short one."

def test_parse_metadata_contains_correct_keys():
    paragraphs = ["Para1", "Para2"]
    docx_path = create_temp_docx(paragraphs)
    parser = DOCXParser()
    chunks = parser.parse(docx_path)
    for chunk in chunks:
        assert "type" in chunk.metadata
        assert "index" in chunk.metadata
        assert chunk.metadata["type"] == "paragraph"

def test_parse_with_nonexistent_file_raises_exception():
    parser = DOCXParser()
    with pytest.raises(Exception):
        parser.parse(Path("/nonexistent/path/to/file.docx"))

