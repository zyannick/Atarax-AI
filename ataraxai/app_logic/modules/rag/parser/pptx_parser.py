from pptx import Presentation
from pathlib import Path
from typing import List
from .document_base_parser import (
    DocumentParser,
    DocumentChunk,
)


class PPTXParser(DocumentParser):
    def parse(self, path: Path) -> List[DocumentChunk]:
        prs = Presentation(path)
        chunks = []
        for i, slide in enumerate(prs.slides):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                chunks.append(
                    DocumentChunk(
                        content="\n".join(texts),
                        source=str(path),
                        metadata={"slide": i + 1},
                    )
                )
        return chunks
