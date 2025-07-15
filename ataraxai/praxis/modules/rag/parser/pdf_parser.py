import fitz 
from pathlib import Path
from typing import List
from ataraxai.praxis.modules.rag.parser.document_base_parser import (
    DocumentParser,
    DocumentChunk,
)
from ataraxai.praxis.modules.rag.parser.base_meta_data import set_base_metadata


class PDFParser(DocumentParser):
    def parse(self, path: Path) -> List[DocumentChunk]:
        """
        Parses a PDF file and extracts its text content into a list of DocumentChunk objects, one per page.

        Args:
            path (Path): The path to the PDF file to be parsed.

        Returns:
            List[DocumentChunk]: A list of DocumentChunk objects, each containing the text content and metadata for a page in the PDF.
        """
        doc: fitz.Document = fitz.open(path)
        chunks: List[DocumentChunk] = []
        base_metadata = set_base_metadata(path)
        for i in range(doc.page_count): # type: ignore
            page = doc.load_page(i) # type: ignore
            text = page.get_text() # type: ignore
            if text.strip(): # type: ignore
                chunks.append(
                    DocumentChunk(
                        content=text.strip(), source=str(path), metadata={**base_metadata, "page": i + 1} # type: ignore
                    )
                )
        return chunks


if __name__ == "__main__":
    parser = PDFParser()
    pdf_path = Path("tests/python/assets/1706.03762v7.pdf") 
    document_chunks = parser.parse(pdf_path)
    for chunk in document_chunks:
        print(f"Page {chunk.metadata.get('page')}: {chunk.content[:100]}...")  
        print(f"Source: {chunk.source}")
        print(f"Metadata: {chunk.metadata}")
    print(f"Total pages parsed: {len(document_chunks)}")