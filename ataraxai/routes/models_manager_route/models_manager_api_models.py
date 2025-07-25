from enum import Enum, auto
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from ataraxai.routes.status import Status


class ModelInfoResponse(BaseModel):
    organization: str = Field(
        ...,
        description="The organization that owns the model.",
    )
    repo_id: str = Field(
        ...,
        description="The repository ID of the model.",
    )
    file_name: str = Field(
        ...,
        description="The file name of the model.",
    )
    local_path: str = Field(
        ...,
        description="The local path where the model is stored.",
    )
    file_size: int = Field(
        ...,
        description="The size of the model file in bytes.",
    )
    create_at: str = Field(
        ...,
        description="The creation date of the model.",
    )
    downloads = int = Field(
        ...,
        description="The number of times the model has been downloaded.",
    )
    likes = int = Field(
        ...,
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
    
class DownloadModelRequest(BaseModel):
    organization: str = Field(
        ...,
        description="The organization that owns the model.",
    )
    repo_id: str = Field(
        ...,
        description="The repository ID of the model.",
    )
    file_name: str = Field(
        ...,
        description="The file name of the model.",
    )
    local_path: str = Field(
        ...,
        description="The local path where the model is stored.",
    )
    file_size: int = Field(
        ...,
        description="The size of the model file in bytes.",
    )
    create_at: str = Field(
        ...,
        description="The creation date of the model.",
    )
    downloads = int = Field(
        ...,
        description="The number of times the model has been downloaded.",
    )
    likes = int = Field(
        ...,
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
        le=100,
        description="The maximum number of models to return.",
    )
    
class SearchModelsResponse(BaseModel):
    status: Status = Field(..., description="Status of the search operation.")
    message: str = Field(
        ..., description="Detailed message about the search operation."
    )
    models: List[ModelInfoResponse] = Field(
        ...,
        description="List of model information objects returned by the search.",
    )

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v
