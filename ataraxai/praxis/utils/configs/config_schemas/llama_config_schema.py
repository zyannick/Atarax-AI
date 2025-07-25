from pydantic import BaseModel
from typing import List, Optional
from ataraxai.praxis.modules.models_manager.models_manager import ModelInfo

class LlamaModelParams(BaseModel):
    config_version: float = 1.0
    # model_path: str = "data/last_models/models/llama/Qwen3-30B-A3B-UD-IQ1_M.gguf"
    model_info : Optional[ModelInfo] = None   
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    main_gpu: int = 0
    tensor_split: bool = False
    vocab_only: bool = False
    use_map: bool = False
    use_mlock: bool = False
    
    @property
    def model_path(self) -> str:
        return self.model_info.local_path if self.model_info else ""
    
    
    def is_setup_complete(self) -> bool:
        return bool(self.model_path)


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
    
    def is_setup_complete(self) -> bool:
        return self.n_predict > 0 and self.temp >= 0.0 and self.top_k >= 0 and self.top_p >= 0.0


class LlamaConfig(BaseModel):
    config_version: float = 1.0
    llama_cpp_model_params: LlamaModelParams = LlamaModelParams()
    generation_params: GenerationParams = GenerationParams()
    
    def is_setup_complete(self) -> bool:
        return self.llama_cpp_model_params.is_setup_complete() and self.generation_params.is_setup_complete()
