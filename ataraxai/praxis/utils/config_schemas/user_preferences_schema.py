from pydantic import BaseModel, Field
from typing import Dict


class UserPreferences(BaseModel):
    config_version: float = 1.0

    index_on_startup: bool = True
    realtime_monitoring: bool = False

    font_size: int = 12

    notifications_enabled: bool = True

    shortcuts: Dict[str, str] = Field(default_factory=dict)

    theme: str = "system_default"

    llm_model_path: str = ""
    whisper_model_path: str = ""
    language: str = "en"

    def is_setup_complete(self) -> bool:
        return bool(self.llm_model_path and self.whisper_model_path)
