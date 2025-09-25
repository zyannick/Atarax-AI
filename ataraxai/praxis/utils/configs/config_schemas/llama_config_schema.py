from typing import List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator

from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo

from ataraxai.hegemonikon_py import HegemonikonLlamaModelParams, HegemonikonGenerationParams # type: ignore

class LlamaModelParams(BaseModel):
    config_version: str = Field(
        default="1.0", description="Version of the model configuration."
    )
    model_info: Optional[LlamaCPPModelInfo] = Field(
        None, description="Model information including local path and metadata."
    )
    n_ctx: int = Field(default=2048, description="Context size for the model.")
    n_gpu_layers: int = Field(default=0, description="Number of GPU layers to use.")
    main_gpu: int = Field(default=0, description="Main GPU to use.")
    tensor_split: bool = Field(default=False, description="Whether to use tensor splitting.")
    vocab_only: bool = Field(default=False, description="Whether to use vocabulary only.")
    use_map: bool = Field(default=False, description="Whether to use memory mapping.")
    use_mlock: bool = Field(default=False, description="Whether to use mlock.")

    @computed_field
    def model_path(self) -> str:
        return self.model_info.local_path if self.model_info else ""

    def is_setup_complete(self) -> bool:
        return bool(self.model_path)
    
    def to_hegemonikon(self) -> HegemonikonLlamaModelParams:
        return HegemonikonLlamaModelParams.from_dict(self.model_dump()) # type: ignore


class GenerationParams(BaseModel):
    config_version: str = Field(
        default="1.0", description="Version of the generation parameters schema."
    )
    n_predict: int = Field(default=128, description="Number of tokens to predict.")
    temperature: float = Field(default=0.8, description="Temperature for sampling.")
    top_k: int = Field(default=40, description="Top-k sampling parameter.")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter.")
    repeat_penalty: float = Field(default=1.2, description="Penalty for repeated tokens.")
    penalty_last_n: int = Field(default=64, description="Last N tokens to apply penalty.")
    penalty_freq: float = Field(
        default=0.7, description="Frequency penalty for repeated tokens."
    )
    penalty_present: float = Field(
        default=0.0, description="Present penalty for repeated tokens."
    )
    stop_sequences: List[str] = Field(
        default_factory=lambda: ["</s>", "\n\n", "User:"],
        description="Sequences that will stop generation.",
    )
    n_batch: int = Field(default=1, description="Batch size for generation.")
    n_threads: int = Field(default=4, description="Number of threads to use for generation.")

    def is_setup_complete(self) -> bool:
        return (
            self.n_predict > 0
            and self.temperature >= 0.0
            and self.top_k >= 0
            and self.top_p >= 0.0
        )

    @field_validator("temperature", "top_k", "top_p")
    def validate_positive(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Value must be non-negative.")
        return value
    
    def to_hegemonikon(self) -> HegemonikonGenerationParams:
        return HegemonikonGenerationParams.from_dict(self.model_dump()) # type: ignore


class LlamaConfig(BaseModel):
    config_version: str = Field(
        "1.0", description="Version of the model configuration."
    )
    llama_cpp_model_params: LlamaModelParams = Field(default_factory=lambda: LlamaModelParams())  # type: ignore
    generation_params: GenerationParams = Field(default_factory=lambda: GenerationParams())  # type: ignore

    def is_setup_complete(self) -> bool:
        return (
            self.llama_cpp_model_params.is_setup_complete()
            and self.generation_params.is_setup_complete()
        )
