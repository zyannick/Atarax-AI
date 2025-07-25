from pydantic import BaseModel, Field, computed_field
from typing import List, Optional
from ataraxai.praxis.modules.models_manager.models_manager import ModelInfo

from typing import Literal, Optional
from enum import Enum, auto


class LlamaModelParams(BaseModel):
    config_version: float = Field(1.0, description="Version of the model configuration.")
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

    @computed_field
    def model_path(self) -> str:
        return self.model_info.local_path if self.model_info else ""
    
    
    def is_setup_complete(self) -> bool:
        return bool(self.model_path)


class GenerationParams(BaseModel):
    config_version: float = Field(1.0, description="Version of the generation parameters schema.")
    n_predict: int = Field(128, description="Number of tokens to predict.")
    temperature: float = Field(0.8, description="Temperature for sampling.")
    top_k: int = Field(40, description="Top-k sampling parameter.")
    top_p: float = Field(0.95, description="Top-p sampling parameter.")
    repeat_penalty: float = Field(1.2, description="Penalty for repeated tokens.")
    penalty_last_n: int = Field(64, description="Last N tokens to apply penalty.")
    penalty_freq: float = Field(0.7, description="Frequency penalty for repeated tokens.")
    penalty_present: float = Field(0.0, description="Present penalty for repeated tokens.")
    stop_sequences: List[str] = Field(default_factory=lambda: ["</s>", "\n\n", "User:"], description="Sequences that will stop generation.")
    n_batch: int = Field(1, description="Batch size for generation.")
    n_threads: int = Field(4, description="Number of threads to use for generation.")

    def is_setup_complete(self) -> bool:
        return self.n_predict > 0 and self.temperature >= 0.0 and self.top_k >= 0 and self.top_p >= 0.0


class LlamaConfig(BaseModel):
    config_version: float = Field(1.0, description="Version of the model configuration.")
    llama_cpp_model_params: LlamaModelParams = Field(default_factory=lambda : LlamaModelParams()) # type: ignore
    generation_params: GenerationParams = Field(default_factory=lambda : GenerationParams()) # type: ignore

    def is_setup_complete(self) -> bool:
        return self.llama_cpp_model_params.is_setup_complete() and self.generation_params.is_setup_complete()
