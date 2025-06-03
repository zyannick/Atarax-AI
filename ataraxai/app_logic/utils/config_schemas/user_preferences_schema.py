
from pydantic import BaseModel, Field
from typing import List

class UserPreferences(BaseModel):
    config_version: float = 1.0
    
    watched_directories: List[str] = Field(default_factory=list)
    index_on_startup: bool = True
    realtime_monitoring: bool = False

    theme: str = "system_default"  

    llm_model_path: str = ""
    whisper_model_path: str = ""

    def is_setup_complete(self) -> bool:
        return bool(self.llm_model_path and self.whisper_model_path)
