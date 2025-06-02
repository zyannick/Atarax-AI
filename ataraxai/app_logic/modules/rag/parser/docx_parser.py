from docx import Document
from pathlib import Path
from typing import List
from .document_base_parser import (
    DocumentParser,
    DocumentChunk,
)


class DOCXParser(DocumentParser):
    def parse(self, path: Path) -> List[DocumentChunk]:
        doc = Document(path)
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
