import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ataraxai.app_logic.modules.rag.smart_chunker import SmartChunker, EXT_PARSER_MAP
from ataraxai.app_logic.modules.rag.parser.document_base_parser import DocumentChunk

@pytest.fixture
def dummy_document_chunk():
    return DocumentChunk(
        content="This is a test document. It has several sentences. Here is another one.",
        source="dummy.txt",
        metadata={"author": "tester"}
    )

def test_init_default_tokenizer(monkeypatch):
    with patch("tiktoken.encoding_for_model", side_effect=KeyError), \
         patch("tiktoken.get_encoding") as mock_get_encoding:
        mock_get_encoding.return_value = MagicMock()
        chunker = SmartChunker(model_name_for_tiktoken="nonexistent-model")
        assert hasattr(chunker, "_tokenizer")

def test_init_overlap_greater_than_size():
    with pytest.raises(ValueError):
        SmartChunker(chunk_size_tokens=100, chunk_overlap_tokens=200)

def test_chunk_single_document_content(dummy_document_chunk):
    chunker = SmartChunker(chunk_size_tokens=10, chunk_overlap_tokens=2)
    chunks = chunker._chunk_single_document_content(
        document_content=dummy_document_chunk.content,
        source_path=dummy_document_chunk.source,
        base_metadata=dummy_document_chunk.metadata
    )
    assert isinstance(chunks, list)
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(c.source == "dummy.txt" for c in chunks)
    assert all("chunk_index_in_doc" in c.metadata for c in chunks)

def test_chunk_empty_content():
    chunker = SmartChunker()
    result = chunker._chunk_single_document_content("", "source.txt", {})
    assert result == []

def test_chunk_list_of_document_chunks(dummy_document_chunk):
    chunker = SmartChunker(chunk_size_tokens=10, chunk_overlap_tokens=2)
    result = chunker.chunk([dummy_document_chunk])
    assert isinstance(result, list)
    assert all(isinstance(c, DocumentChunk) for c in result)

def test_chunk_skips_non_documentchunk():
    chunker = SmartChunker()
    result = chunker.chunk(["not a DocumentChunk"])
    assert result == []

def test_ingest_file_with_parser(tmp_path, monkeypatch):
    # Create a dummy file
    file_path = tmp_path / "test.pdf"
    file_path.write_text("dummy content")

    dummy_parser = MagicMock()
    dummy_parser.parse.return_value = [
        DocumentChunk(content="abc def ghi", source=str(file_path), metadata={})
    ]
    monkeypatch.setitem(EXT_PARSER_MAP, ".pdf", dummy_parser)

    chunker = SmartChunker(chunk_size_tokens=2, chunk_overlap_tokens=0)
    chunks = chunker.ingest_file(file_path)
    assert isinstance(chunks, list)
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    dummy_parser.parse.assert_called_once_with(file_path)

def test_ingest_file_no_parser(tmp_path):
    file_path = tmp_path / "test.unknown"
    file_path.write_text("dummy content")
    chunker = SmartChunker()
    chunks = chunker.ingest_file(file_path)
    assert chunks == []

def test_ingest_file_not_a_file(tmp_path):
    dir_path = tmp_path / "adir"
    dir_path.mkdir()
    chunker = SmartChunker()
    chunks = chunker.ingest_file(dir_path)
    assert chunks == []

def test_ingest_directory(tmp_path, monkeypatch):
    # Setup files
    pdf_path = tmp_path / "a.pdf"
    docx_path = tmp_path / "b.docx"
    pdf_path.write_text("pdf content")
    docx_path.write_text("docx content")

    dummy_pdf_parser = MagicMock()
    dummy_pdf_parser.parse.return_value = [
        DocumentChunk(content="pdf", source=str(pdf_path), metadata={})
    ]
    dummy_docx_parser = MagicMock()
    dummy_docx_parser.parse.return_value = [
        DocumentChunk(content="docx", source=str(docx_path), metadata={})
    ]
    monkeypatch.setitem(EXT_PARSER_MAP, ".pdf", dummy_pdf_parser)
    monkeypatch.setitem(EXT_PARSER_MAP, ".docx", dummy_docx_parser)

    chunker = SmartChunker(chunk_size_tokens=2, chunk_overlap_tokens=0)
    chunks = chunker.ingest_directory(tmp_path)
    assert isinstance(chunks, list)
    assert any(c.source == str(pdf_path) for c in chunks)
    assert any(c.source == str(docx_path) for c in chunks)