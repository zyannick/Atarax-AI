import fitz  # PyMuPDF
from pathlib import Path
from typing import List
from .document_base_parser import (
    DocumentParser,
    DocumentChunk,
)
from .base_meta_data import get_file_hash, set_base_metadata


class PDFParser(DocumentParser):
    def parse(self, path: Path) -> List[DocumentChunk]:
        doc = fitz.open(path)
        chunks = []
        base_metadata = set_base_metadata(path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks.append(
                    DocumentChunk(
                        content=text.strip(), source=str(path), metadata={**base_metadata, "page": i + 1}
                    )
                )
        return chunks
