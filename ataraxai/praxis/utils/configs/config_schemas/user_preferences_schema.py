from pydantic import BaseModel, Field, field_validator
from typing import Dict
from enum import Enum, auto


class AppTheme(str, Enum):
    SYSTEM_DEFAULT = auto()
    LIGHT = auto()
    DARK = auto()


class UserPreferences(BaseModel):
    config_version: float = 1.0

    index_on_startup: bool = True
    realtime_monitoring: bool = False
    font_size: int = 12
    notifications_enabled: bool = True
    shortcuts: Dict[str, str] = Field(default_factory=dict)
    theme: AppTheme = AppTheme.SYSTEM_DEFAULT
    language: str = "en"

    def is_setup_complete(self) -> bool:
        return True

    @field_validator("font_size")
    @classmethod
    def validate_font_size(cls, value: int) -> int:
        if not (8 <= value <= 32):
            raise ValueError("Font size must be between 8 and 32.")
        return value
