from pydantic import BaseModel, Field, field_validator
from typing import Dict
from enum import Enum

class AppTheme(str, Enum):
    SYSTEM_DEFAULT = "system_default"
    LIGHT = "light"
    DARK = "dark"

class UserPreferences(BaseModel):
    config_version: str = Field("1.0", description="Version of the user preferences configuration.")

    index_on_startup: bool = Field(True, description="Whether to index on startup.")
    realtime_monitoring: bool = Field(False, description="Whether to enable real-time monitoring.")
    font_size: int = Field(12, description="Font size for the application.")
    notifications_enabled: bool = Field(True, description="Whether to enable notifications.")
    shortcuts: Dict[str, str] = Field(default_factory=dict, description="Keyboard shortcuts.")
    theme: AppTheme = Field(AppTheme.SYSTEM_DEFAULT, description="Application theme.")
    language: str = Field("en", description="Language for the application.")

    def is_setup_complete(self) -> bool:
        return True

    @field_validator("font_size")
    @classmethod
    def validate_font_size(cls, value: int) -> int:
        if not (8 <= value <= 32):
            raise ValueError("Font size must be between 8 and 32.")
        return value
