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
    
    def to_dict(self):
        return {
            "config_version": self.config_version,
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "main_gpu": self.main_gpu,
            "tensor_split": self.tensor_split,
            "vocab_only": self.vocab_only,
            "use_map": self.use_map,
            "use_mlock": self.use_mlock
        }



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
    
    def to_dict(self):
        return {
            "config_version": self.config_version,
            "n_predict": self.n_predict,
            "temp": self.temp,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "stop_sequences": self.stop_sequences,
            "n_batch": self.n_batch,
            "n_threads": self.n_threads
        }


class LlamaConfig(BaseModel):
    config_version: float = 1.0
    llm_model_params: LlamaModelParams = LlamaModelParams()
    generation_params: GenerationParams = GenerationParams()
    
    def to_dict(self):
        return {
            "config_version": self.config_version,
            "llm_model_params": self.llm_model_params.to_dict(),
            "generation_params": self.generation_params.to_dict()
        }
