import yaml
from pathlib import Path
from platformdirs import user_config_dir

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"
LLAMA_CONFIG_FILENAME = "llama_config.yaml"


class LlamaConfigManager:
    def __init__(self):
        self.config_path = (
            Path(user_config_dir(APP_NAME, APP_AUTHOR)) / LLAMA_CONFIG_FILENAME
        )
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_or_initialize()

    def _default_config(self):
        return {
            "config_version": 1.0,
            "llama_model_params": {
                "model_path": "",
                "n_ctx": 2048,
                "n_gpu_layers": 0,
                "main_gpu": 0,
                "tensor_split": False,
                "vocab_only": False,
                "use_map": False,
                "use_mlock": False,
            },
            "generation_params": {
                "n_predict": 128,
                "temp": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "repeat_penalty": 1.1,
                "stop_sequences": [],
                "n_batch": 512,
                "n_threads": 0,
            },
        }

    def _load_or_initialize(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load YAML config: {e}")
        print(f"[INFO] Creating default Llama config: {self.config_path}")
        default = self._default_config()
        self._save(default)
        return default

    def _save(self, config=None):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config or self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")

    def get_llama_params(self):
        return self.config["llama_model_params"]

    def get_generation_params(self):
        return self.config["generation_params"]

    def set_param(self, section, key, value):
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self._save()
        else:
            raise KeyError(f"Section '{section}' or key '{key}' not found in config.")

    def reload(self):
        self.config = self._load_or_initialize()
