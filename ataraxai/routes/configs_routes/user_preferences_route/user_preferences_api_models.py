from pydantic import BaseModel, Field, field_validator
from typing import List
from ataraxai.routes.status import Status
from ataraxai.praxis.utils.configs.config_schemas.user_preferences_schema import AppTheme


class UserPreferencesAPI(BaseModel):
    index_on_startup: bool = Field(
        True, description="Whether to index on application startup."
    )
    realtime_monitoring: bool = Field(
        False, description="Enable or disable real-time monitoring."
    )
    font_size: int = Field(12, description="Font size for the user interface.")
    notifications_enabled: bool = Field(
        True, description="Enable or disable notifications."
    )
    shortcuts: List[str] = Field(
        default_factory=list, description="List of user-defined shortcuts."
    )
    theme: AppTheme = Field(AppTheme.SYSTEM_DEFAULT, description="Application theme preference.")
    
    class Config:
        use_enum_values = True
        from_attributes = True
        
class UserPreferencesResponse(BaseModel):
    status: Status = Field(..., description="Status of the user preferences operation.")
    message: str = Field(
        ..., description="Detailed message about the user preferences operation."
    )
    preferences: UserPreferencesAPI = Field(
        ..., description="User preferences data."
    )

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v