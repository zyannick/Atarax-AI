from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from ataraxai.routes.status import Status



class RagConfigAPI(BaseModel):
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
    
    @field_validator("rag_model_path")
    def validate_rag_model_path(cls, v: str) -> str:
        if not v:
            raise ValueError("RAG model path cannot be empty.")
        return v
    
class RagConfigResponse(BaseModel):
    status: Status = Field(..., description="Status of the RAG configuration operation.")
    message: str = Field(
        ..., description="Detailed message about the RAG configuration operation."
    )
    config: Optional[RagConfigAPI] = Field(
        default_factory=lambda: None, description="RAG configuration data."
    )

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v