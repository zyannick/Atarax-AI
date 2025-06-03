import yaml
from pathlib import Path
from platformdirs import user_config_dir

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"
WHISPER_CONFIG_FILENAME = "whisper_config.yaml"


class WhisperConfigManager:
    
    def __init__(self):
        self.config_path = Path(user_config_dir(APP_NAME, APP_AUTHOR)) / WHISPER_CONFIG_FILENAME
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_or_initialize()
        
    
    def _default_config(self):
        return {
            "whisper_model_params": {
                "model_path": "en",
                "use_gpu": True,
            },
            "whisper_transcription_params": {
                "n_threads": 0,
                "language": "en",
                "translate": False,
                "print_special": False,
                "print_progress": True,
                "no_context": True,
                "max_len": 0,
                "single_segment": False,
                "temperature": 0.0,
            }
        }
        
    def _load_or_initialize(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load YAML config: {e}")
        print(f"[INFO] Creating default Whisper config: {self.config_path}")
        default = self._default_config()
        self._save(default)
        return default
    
    def _save(self, config=None):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config or self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")
            
    def get_whisper_params(self):
        return self.config["whisper_model_params"]
    
    def get_transcription_params(self):
        return self.config["whisper_transcription_params"]
    
    def update_whisper_params(self, params):
        self.config["whisper_model_params"].update(params)
        self._save()
        
    def update_transcription_params(self, params):
        
        self.config["whisper_transcription_params"].update(params)
        self._save()
    
    def get_config(self):
        return self.config
    
    def set_param(self, section, key, value):
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self._save()
        else:
            raise KeyError(f"Section '{section}' or key '{key}' not found in config.")