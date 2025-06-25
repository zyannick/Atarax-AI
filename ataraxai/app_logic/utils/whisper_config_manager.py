import yaml
from pathlib import Path
from .config_schemas.whisper_config_schema import (
    WhisperConfig,
    WhisperModelParams,
    WhisperTranscriptionParams,
)


WHISPER_CONFIG_FILENAME = "whisper_config.yaml"


class WhisperConfigManager:

    def __init__(self, config_path: Path):
        self.config_path = config_path / WHISPER_CONFIG_FILENAME
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config: WhisperConfig = self._load_or_initialize()

    def _load_or_initialize(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return WhisperConfig(**yaml.safe_load(f))
            except Exception as e:
                print(f"[ERROR] Failed to load YAML config: {e}")
        print(f"[INFO] Creating default Whisper config: {self.config_path}")
        default = self._default_config()
        self._save(default)
        return default

    def _save(self, config=None):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config.dict() if config else self.config.dict(),
                    f,
                    default_flow_style=False,
                )
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")

    def get_whisper_params(self) -> WhisperModelParams:
        return self.config.whisper_model_params

    def get_transcription_params(self) -> WhisperTranscriptionParams:
        return self.config.whisper_transcription_params

    def update_whisper_params(self, params: WhisperModelParams):
        self.config.whisper_model_params = params
        self._save()

    def update_transcription_params(self, params: WhisperTranscriptionParams):
        self.config.whisper_transcription_params = params
        self._save()

    def get_config(self) -> WhisperConfig:
        return self.config

    def set_param(self, section: str, key: str, value):
        if (
            section == "whisper_model_params"
            and key in self.config.whisper_model_params.__dict__
        ):
            setattr(self.config.whisper_model_params, key, value)
            self._save()
        elif (
            section == "whisper_transcription_params"
            and key in self.config.whisper_transcription_params.__dict__
        ):
            setattr(self.config.whisper_transcription_params, key, value)
            self._save()
        else:
            raise KeyError(f"Section '{section}' or key '{key}' not found in config.")

    def reload(self):
        self.config = self._load_or_initialize()
