from pathlib import Path
from typing import List
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    content: str
    source: str
    metadata: dict
    
    def __repr__(self):
        metadata_preview = {k: v for k, v in list(self.metadata.items())[:2]} 
        return (f"DocumentChunk(source='{self.source}', "
                f"content='{self.content[:50].replace('\n', ' ')}...', " 
                f"metadata={metadata_preview}{'...' if len(self.metadata) > 2 else ''})")


class DocumentParser:
    def parse(self, path: Path) -> List[DocumentChunk]:
        raise NotImplementedError("Must implement parse()")