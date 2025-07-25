from pydantic import BaseModel, Field, field_validator
from typing import List
from ataraxai.praxis.modules.models_manager.models_manager import ModelInfo
from ataraxai.routes.status import Status
from ataraxai.praxis.utils.configs.config_schemas.user_preferences_schema import AppTheme


class LlamaCPPConfigAPI(BaseModel):
    model_path: ModelInfo = Field(
        ..., description="Path to the Llama.cpp model file."
    )
    n_ctx: int = Field(
        2048, description="Context size for the Llama.cpp model."
    )
    n_batch: int = Field(
        512, description="Batch size for processing."
    )
    n_gpu_layers: int = Field(
        0, description="Number of GPU layers to use."
    )
    seed: int = Field(
        0, description="Random seed for reproducibility."
    )
    temperature: float = Field(
        0.7, description="Sampling temperature for text generation."
    )
    top_p: float = Field(
        0.9, description="Top-p sampling parameter."
    )
    top_k: int = Field(
        50, description="Top-k sampling parameter."
    )
    
        
class LlamaCPPGenerationParamsAPI(BaseModel):
    n_predict: int = Field(
        128, description="Number of tokens to predict."
    )
    temp: float = Field(
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