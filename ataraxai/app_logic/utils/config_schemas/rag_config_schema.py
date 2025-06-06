from pydantic import BaseModel, Field
from typing import List


class RAGConfig(BaseModel):
    config_version: float = 1.0

    rag_model_path: str = ""
    rag_chunk_size: int = 400
    rag_chunk_overlap: int = 50
    rag_separators: List[str] | None = None
    rag_keep_separator: bool = True
    rag_model_name_for_tiktoken: str = "gpt-3.5-turbo"

    def is_setup_complete(self) -> bool:
        return bool(self.rag_model_path)