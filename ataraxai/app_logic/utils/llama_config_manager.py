import yaml
from pathlib import Path
from ataraxai.app_logic.utils.config_schemas.llama_config_schema import (
    LlamaConfig,
    LlamaModelParams,
    GenerationParams,
)
from typing_extensions import Optional


LLAMA_CONFIG_FILENAME = "llama_config.yaml"


class LlamaConfigManager:
    def __init__(self, config_path: Path):
        if config_path.is_dir():
            self.config_path = config_path / LLAMA_CONFIG_FILENAME
        else:
            raise ValueError(f"Invalid config path: {config_path}")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config: LlamaConfig = self._load_or_initialize()

    def _load_or_initialize(self) -> LlamaConfig:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f)
                return LlamaConfig(**raw_data)
            except Exception as e:
                print(f"[ERROR] Failed to parse config: {e}")
        print(f"[INFO] Creating default LLM config at: {self.config_path}")
        config = LlamaConfig()
        self._save(config)
        return config

    def _save(self, config: Optional[LlamaConfig] = None):
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config.model_dump() if config else self.config.model_dump(),
                f,
                default_flow_style=False,
            )

    def get_llama_cpp_params(self) -> LlamaModelParams:
        return self.config.llama_cpp_model_params

    def set_llama_cpp_params(self, params: LlamaModelParams):
        self.config.llama_cpp_model_params = params
        self._save()

    def get_generation_params(self) -> GenerationParams:
        return self.config.generation_params

    def set_generation_params(self, params: GenerationParams):
        self.config.generation_params = params
        self._save()

    def set_param(self, section: str, key: str, value: str):
        if section == "llama_cpp_model_params" and hasattr(self.config.llama_cpp_model_params, key):
            setattr(self.config.llama_cpp_model_params, key, value)
        elif section == "generation_params" and hasattr(
            self.config.generation_params, key
        ):
            setattr(self.config.generation_params, key, value)
        else:
            raise ValueError(f"Invalid section '{section}' or key '{key}'")
        self._save()

    def reload(self):
        self.config = self._load_or_initialize()
