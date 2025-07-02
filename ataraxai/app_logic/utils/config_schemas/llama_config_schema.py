from pydantic import BaseModel
from typing import List


class LlamaModelParams(BaseModel):
    config_version : float = 1.0
    model_path: str = ""
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    main_gpu: int = 0
    tensor_split: bool = False
    vocab_only: bool = False
    use_map: bool = False
    use_mlock: bool = False
    




class GenerationParams(BaseModel):
    config_version: float = 1.0
    n_predict: int = 128
    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.2
    penalty_last_n: int = 64
    penalty_freq: float = 0.7
    penalty_present: float = 0.0
    stop_sequences: List[str] = ["</s>", "\n\n", "User:"]
    n_batch: int = 1
    n_threads: int = 4
    



class LlamaConfig(BaseModel):
    config_version: float = 1.0
    llm_model_params: LlamaModelParams = LlamaModelParams()
    generation_params: GenerationParams = GenerationParams()
    

