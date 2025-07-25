from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from ataraxai.praxis.modules.models_manager.models_manager import ModelInfo
from ataraxai.routes.status import Status


class LlamaCPPConfigAPI(BaseModel):
    model_info : Optional[ModelInfo] = Field(
        None, description="Model information including local path and metadata."
    )
    n_ctx: int = Field(2048, description="Context size for the model.")
    n_gpu_layers: int = Field(0, description="Number of GPU layers to use.")
    main_gpu: int = Field(0, description="Main GPU to use.")
    tensor_split: bool = Field(False, description="Whether to use tensor splitting.")
    vocab_only: bool = Field(False, description="Whether to use vocabulary only.")
    use_map: bool = Field(False, description="Whether to use memory mapping.")
    use_mlock: bool = Field(False, description="Whether to use mlock.")

    @field_validator("model_info")
    @classmethod
    def validate_model_info(cls, v: ModelInfo) -> ModelInfo:
        if not v.local_path:
            raise ValueError("Model path cannot be empty.")
        return v
    
    @field_validator("n_ctx", "n_gpu_layers", "main_gpu")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Value must be non-negative.")
        return v
    
    
class LlamaCPPConfigResponse(BaseModel):
    status: Status = Field(..., description="Status of the Llama.cpp configuration operation.")
    message: str = Field(
        ..., description="Detailed message about the Llama.cpp configuration operation."
    )
    config: LlamaCPPConfigAPI = Field(
        ..., description="Llama.cpp configuration data."
    )

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v
    
        
class LlamaCPPGenerationParamsAPI(BaseModel):
    n_predict: int = Field(
        128, description="Number of tokens to predict."
    )
    temperature: float = Field(
        0.8, description="Temperature for sampling."
    )
    top_k: int = Field(
        40, description="Top-k sampling parameter."
    )
    top_p: float = Field(
        0.95, description="Top-p sampling parameter."
    )
    repeat_penalty: float = Field(
        1.2, description="Penalty for repeated tokens."
    )
    penalty_last_n: int = Field(
        64, description="Last N tokens to apply penalty."
    )
    penalty_freq: float = Field(
        0.7, description="Frequency penalty for repeated tokens."
    )
    penalty_present: float = Field(
        0.0, description="Present penalty for repeated tokens."
    )
    stop_sequences: List[str] = Field(
        default_factory=lambda: ["</s>", "\n\n", "User:"],
        description="Sequences that will stop generation."
    )
    n_batch: int = Field(
        1, description="Batch size for generation."
    )
    n_threads: int = Field(
        4, description="Number of threads to use for generation."
    )

    @field_validator("n_predict", "temperature", "top_k", "top_p", "repeat_penalty", "penalty_last_n", "penalty_freq", "penalty_present", "n_batch", "n_threads")
    @classmethod
    def validate_positive(cls, v: int | float) -> int | float:
        if v < 0:
            raise ValueError("Value must be non-negative.")
        return v
    
class LlamaCPPGenerationParamsResponse(BaseModel):
    status: Status = Field(..., description="Status of the Llama.cpp generation parameters operation.")
    message: str = Field(
        ..., description="Detailed message about the Llama.cpp generation parameters operation."
    )
    params: LlamaCPPGenerationParamsAPI = Field(
        ..., description="Llama.cpp generation parameters data."
    )

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v