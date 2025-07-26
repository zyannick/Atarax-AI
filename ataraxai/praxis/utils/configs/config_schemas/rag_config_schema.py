from pydantic import BaseModel, Field, field_validator
from typing import List
from pathlib import Path

class RAGConfig(BaseModel):
    config_version: float = Field(1.0, description="Version of the RAG configuration.")
    rag_model_path: str = Field("", description="Path to the RAG model.")
    rag_chunk_size: int = Field(400, description="Size of the chunks to be processed.")
    rag_chunk_overlap: int = Field(50, description="Overlap between chunks.")
    rag_watched_directories: List[str] | None = Field(None, description="Directories to watch for changes.")
    rag_time_out_update: float = Field(30.0, description="Timeout for updates.")
    rag_separators: List[str] | None = Field(None, description="List of separators.")
    rag_keep_separator: bool = Field(True, description="Whether to keep the separator.")
    rag_model_name_for_tiktoken: str = Field("gpt-3.5-turbo", description="Model name for tokenization.")
    rag_embedder_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedder model.")
    rag_use_reranking: bool = Field(False, description="Whether to use reranking.")
    rag_cross_encoder_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model.")
    cross_encoder_hits: int = Field(5, description="Number of hits for the cross-encoder.")

    def is_setup_complete(self) -> bool:
        return Path(self.rag_model_path).exists()
    
