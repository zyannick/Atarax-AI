
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
    language: str = "en"  

    def is_setup_complete(self) -> bool:
        return bool(self.llm_model_path and self.whisper_model_path)
    
    def to_dict(self):
        return {
            "config_version": self.config_version,
            "watched_directories": self.watched_directories,
            "index_on_startup": self.index_on_startup,
            "realtime_monitoring": self.realtime_monitoring,
            "theme": self.theme,
            "llm_model_path": self.llm_model_path,
            "whisper_model_path": self.whisper_model_path,  
            "language": self.language
        }
