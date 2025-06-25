from pydantic import BaseModel
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

    def to_dict(self):
        return {
            "config_version": self.config_version,
            "rag_model_path": self.rag_model_path,
            "rag_chunk_size": self.rag_chunk_size,
            "rag_chunk_overlap": self.rag_chunk_overlap,
            "rag_separators": self.rag_separators,
            "rag_keep_separator": self.rag_keep_separator,
            "rag_model_name_for_tiktoken": self.rag_model_name_for_tiktoken,
        }
