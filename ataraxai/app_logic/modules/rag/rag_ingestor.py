from pathlib import Path
from typing import List
from .parser.document_base_parser import DocumentChunk
from .parser.pdf_parser import PDFParser
from .parser.docx_parser import DOCXParser
from .parser.pptx_parser import PPTXParser
from smart_chunker import SmartChunker

EXT_PARSER_MAP = {
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".pptx": PPTXParser(),
}


class RAGIngestor:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 500,
        overlap: int = 50,
    ):
        self.chunker = SmartChunker(
            model_name=model_name, max_tokens=max_tokens, overlap=overlap
        )

    def ingest_directory(self, directory: Path) -> List[DocumentChunk]:
        all_chunks = []

        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            parser = EXT_PARSER_MAP.get(path.suffix.lower())
            if parser:
                print(f"[+] Parsing: {path}")
                try:
                    raw_chunks = parser.parse(path)
                    smart_chunks = self.chunker.chunk(raw_chunks)
                    all_chunks.extend(smart_chunks)
                except Exception as e:
                    print(f"[!] Failed to process {path}: {e}")
        return all_chunks
