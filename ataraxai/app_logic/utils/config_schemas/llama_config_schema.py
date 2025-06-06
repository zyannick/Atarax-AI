from pydantic import BaseModel, Field
from typing import List
import ml_collections


class LlamaModelParams(BaseModel):
    model_path: str = ""
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    main_gpu: int = 0
    tensor_split: bool = False
    vocab_only: bool = False
    use_map: bool = False
    use_mlock: bool = False
    



class GenerationParams(BaseModel):
    n_predict: int = 128
    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = Field(default_factory=list)
    n_batch: int = 512
    n_threads: int = 0


class LlamaConfig(BaseModel):
    config_version: float = 1.0
    llm_model_params: LlamaModelParams = LlamaModelParams()
    generation_params: GenerationParams = GenerationParams()
