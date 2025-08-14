from pydantic import BaseModel, Field, model_validator
from typing import List, Set
from pathlib import Path

class RAGConfig(BaseModel):
    rag_model_path: str = Field(default="", description="Path to the RAG model.")
    rag_chunk_size: int = Field(default=400, description="Size of the chunks to be processed.")
    rag_chunk_overlap: int = Field(default=50, description="Overlap between chunks.")
    rag_watched_directories: Set[str] = Field(default_factory=lambda : set(), description="Directories to watch for changes.")
    rag_time_out_update: float = Field(default=30.0, description="Timeout for updates.")
    rag_separators: List[str] | None = Field(default=None, description="List of separators.")
    rag_keep_separator: bool = Field(default=True, description="Whether to keep the separator.")
    rag_model_name_for_tiktoken: str = Field(default="gpt-3.5-turbo", description="Model name for tokenization.")
    rag_embedder_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedder model.")
    rag_use_reranking: bool = Field(default=False, description="Whether to use reranking.")
    rag_cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model.")
    rag_n_result: int = Field(default=5, description="Number of results to return.")
    rag_n_result_final: int = Field(default=3, description="Final number of results after processing.")
    rag_use_hyde: bool = Field(default=True, description="Whether to use HYDE for retrieval.")
    cross_encoder_hits: int = Field(default=5, description="Number of hits for the cross-encoder.")
    context_allocation_ratio: float = Field(
        default=0.5,
        description="Ratio of context allocated to RAG vs history in the prompt assembly.",
    )

    @model_validator(mode='after')
    def check_chunk_settings(self) -> 'RAGConfig':
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("rag_chunk_overlap must be smaller than rag_chunk_size")
        return self

    def is_setup_complete(self) -> bool:
        if not self.rag_model_path or not Path(self.rag_model_path).exists():
            return False
        
        if self.rag_watched_directories and not any(Path(p).is_dir() for p in self.rag_watched_directories):
            return False
            
        return True

    
