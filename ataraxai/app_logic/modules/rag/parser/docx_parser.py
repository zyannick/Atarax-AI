from docx import Document
from pathlib import Path
from typing import List
from ataraxai.app_logic.modules.rag.parser.document_base_parser import (
    DocumentParser,
    DocumentChunk,
)


class DOCXParser(DocumentParser):
    def parse(self, path: Path) -> List[DocumentChunk]:
        """
        Parses a DOCX file and extracts its paragraphs as document chunks.

        Args:
            path (Path): The path to the DOCX file to be parsed.

        Returns:
            List[DocumentChunk]: A list of DocumentChunk objects, each representing a non-empty paragraph from the DOCX file.
                Each chunk contains the paragraph content, the source file path, and metadata with the type ('paragraph') and its index.
        """
        doc = Document(str(path))
        text_blocks = [
            para.text.strip() for para in doc.paragraphs if para.text.strip()
        ]
        chunks = [
            DocumentChunk(
                content=block,
                source=str(path),
                metadata={"type": "paragraph", "index": i},
            )
            for i, block in enumerate(text_blocks)
        ]
        return chunks


if __name__ == "__main__":
    docx_path = Path("tests/python/assets/Attention is All you need.docx")  
    parser = DOCXParser()
    document_chunks = parser.parse(docx_path)
    for chunk in document_chunks:
        print(f"Paragraph {chunk.metadata.get('index')}: {chunk.content[:100]}...")  
        print(f"Source: {chunk.source}")
        print(f"Metadata: {chunk.metadata}")
    print(f"Total paragraphs parsed: {len(document_chunks)}")
