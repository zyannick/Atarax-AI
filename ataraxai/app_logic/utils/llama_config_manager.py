
import yaml
from pathlib import Path
from platformdirs import user_config_dir
from ataraxai.app_logic.utils.config_schemas.llama_config_schema import LlamaConfig, LlamaModelParams, GenerationParams

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"
LLAMA_CONFIG_FILENAME = "llm_config.yaml"


class LlamaConfigManager:
    def __init__(self):
        self.config_path = (
            Path(user_config_dir(APP_NAME, APP_AUTHOR)) / LLAMA_CONFIG_FILENAME
        )
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

    def _save(self, config: LlamaConfig = None):
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config.dict(), f, default_flow_style=False)

    def get_llm_params(self) -> LlamaModelParams:
        return self.config.llm_model_params

    def get_generation_params(self) -> GenerationParams:
        return self.config.generation_params

    def set_param(self, section: str, key: str, value):
        if section == "llm_model_params" and hasattr(self.config.llm_model_params, key):
            setattr(self.config.llm_model_params, key, value)
        elif section == "generation_params" and hasattr(
            self.config.generation_params, key
        ):
            setattr(self.config.generation_params, key, value)
        else:
            raise ValueError(f"Invalid section '{section}' or key '{key}'")
        self._save()

    def reload(self):
        self.config = self._load_or_initialize()
