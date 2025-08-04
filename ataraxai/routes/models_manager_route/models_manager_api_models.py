from datetime import datetime
from enum import Enum, auto
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from ataraxai.routes.status import Status


class ModelInfoResponse(BaseModel):
    organization: str = Field(..., description="The organization or user who owns the model.")
    repo_id: str = Field(..., description="The repository ID of the model.")
    filename: str = Field(..., description="The file name of the model.")
    local_path: str = Field(..., description="The local path where the model is stored.")
    file_size: int = Field(..., description="The size of the model file in bytes.", ge=0)
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The creation date of the model.",
    )
    downloads: int = Field(0, description="The number of times the model has been downloaded.", ge=0)
    likes: int = Field(0, description="The number of likes the model has received.", ge=0)
    quantization_bit: Optional[str] = Field(None, description="The bit quantization of the model, e.g., 'Q4', 'Q8'.")
    quantization_scheme: Optional[str] = Field(None, description="The quantization scheme used for the model.")
    quantization_modifier: Optional[str] = Field(None, description="The quantization modifier used for the model.")
    
class ModelInfoResponsePaginated(BaseModel):
    status: Status = Field(..., description="Status of the operation.")
    message: str = Field(..., description="Detailed message about the operation.")
    models: List[ModelInfoResponse] = Field(..., description="List of model information objects.")
    
    total_count: int = Field(..., description="Total number of models available.")
    page: int = Field(..., description="Current page number (1-based).")
    page_size: int = Field(..., description="Number of items per page.")
    total_pages: int = Field(..., description="Total number of pages.")
    has_next: bool = Field(..., description="Whether there are more pages available.")
    has_previous: bool = Field(..., description="Whether there are previous pages available.")
    
    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v
    
class DownloadModelRequest(BaseModel):
    organization: str = Field(
        ...,
        description="The organization or user who owns the model.",
    )
    repo_id: str = Field(
        ...,
        description="The repository ID of the model.",
    )
    filename: str = Field(
        ...,
        description="The file name of the model.",
    )
    local_path: Optional[str] = Field(
        None,
        description="The local path where the model is stored.",
    )
    file_size: int = Field(
        0,
        description="The size of the model file in bytes.",
    )
    create_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="The creation date of the model.",
    )
    downloads: int = Field(
        0,
        description="The number of times the model has been downloaded.",
    )
    likes: int = Field(
        0,
        description="The number of likes the model has received.",
    )
    quantization_bit: str = Field(
        "default",
        description="The bit quantization of the model, e.g., 'Q4', 'Q8'.",
    )
    quantization_scheme: str = Field(
        "default",
        description="The quantization scheme used for the model, e.g., 'A', 'B'.",
    )
    quantization_modifier: str = Field(
        "default",
        description="The quantization modifier, if any, used for the model.",
    )
    callback_url: Optional[str] = Field(
        None,
        description="Optional callback URL to notify when the download is complete.",
    )
    

class DownloadTaskStatus(str, Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    
class DownloadModelResponse(BaseModel):
    status: DownloadTaskStatus = Field(
        ...,
        description="The status of the download task.",
    )
    message: str = Field(
        ...,
        description="A message providing additional information about the download task.",
    )
    task_id: Optional[str] = Field(
        None,
        description="The unique identifier for the download task, if applicable.",
    )
    percentage: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="The percentage of the download task completed, if applicable.",
    )
    
class SearchModelsRequest(BaseModel):
    query : str = Field(
        default="",
        description="Search query to filter models by name or description.",
    )
    filters_tags: Optional[List[str]] = Field(
        default_factory=lambda: [],
        description="List of tags to apply when searching for models.",
    )
    limit: int = Field(
        10,
        ge=1,
        le=10000,
        description="The maximum number of models to return.",
    )
    
class SearchModelsResponsePaginated(BaseModel):
    status: Status = Field(..., description="Status of the search operation.")
    message: str = Field(
        ..., description="Detailed message about the search operation."
    )
    models: List[ModelInfoResponse] = Field(
        ...,
        description="List of model information objects returned by the search.",
    )
    total_count: int = Field(..., description="Total number of models available.")
    page: int = Field(..., description="Current page number (1-based).")
    page_size: int = Field(..., description="Number of items per page.")
    total_pages: int = Field(..., description="Total number of pages.")
    has_next: bool = Field(..., description="Whether there are more pages available.")
    has_previous: bool = Field(..., description="Whether there are previous pages available.")

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v

class SearchModelsManifestRequest(BaseModel):
    repo_id: Optional[str] = Field(
        default=None,
        description="The repository ID of the model to search for.",
    )
    filename : Optional[str] = Field(
        default=None,
        description="The file name of the model to search for.",
    )
    organization: Optional[str] = Field(
        default=None,
        description="The organization or user who owns the model.",
    )