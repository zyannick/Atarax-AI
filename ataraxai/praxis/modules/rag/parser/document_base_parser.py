from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DocumentChunk:
    content: str
    source: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        metadata_preview = {k: v for k, v in list(self.metadata.items())[:2]} 
        content_preview = self.content[:50].replace('\n', ' ')
        return (f"DocumentChunk(source='{self.source}', "
                f"content='{content_preview}...', " 
                f"metadata={metadata_preview}{'...' if len(self.metadata) > 2 else ''})")


class DocumentParser:
    def parse(self, path: Path) -> List[DocumentChunk]:
        raise NotImplementedError("Must implement parse()")