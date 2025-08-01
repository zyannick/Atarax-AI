from typing import Any, Dict
from pydantic import BaseModel, Field, field_validator
from ataraxai.routes.status import Status


class CoreAiServiceInitializationResponse(BaseModel):
    status: Status = Field(..., description="Status of the Core AI Service initialization operation.")
    message: str = Field(
        ..., description="Detailed message about the Core AI Service initialization operation."
    )
    # data: Dict[str, Any] = Field(
    #     default_factory=dict,
    #     description="Additional data related to the Core AI Service initialization."
    # )

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v