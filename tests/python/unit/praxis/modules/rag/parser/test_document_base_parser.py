import pytest
from pathlib import Path
from ataraxai.praxis.modules.rag.parser.document_base_parser import DocumentChunk, DocumentParser

def test_document_chunk_repr_truncates_content_and_metadata():
    chunk = DocumentChunk(
        content="This is a long content string that should be truncated in the repr output. " * 2,
        source="test_source.txt",
        metadata={"author": "Alice", "date": "2024-06-01", "extra": "value"}
    )
    repr_str = repr(chunk)
    assert "test_source.txt" in repr_str
    assert "This is a long content string" in repr_str
    assert "..." in repr_str 
    assert "author" in repr_str and "date" in repr_str
    assert "extra" not in repr_str  

def test_document_chunk_repr_with_few_metadata():
    chunk = DocumentChunk(
        content="Short content.",
        source="source.txt",
        metadata={"key": "value"}
    )
    repr_str = repr(chunk)
    print(repr_str)
    assert "Short content." in repr_str
    assert "key" in repr_str

def test_document_parser_parse_not_implemented():
    class DummyParser(DocumentParser):
        pass

    parser = DummyParser()
    with pytest.raises(NotImplementedError):
        parser.parse(Path("dummy.txt"))