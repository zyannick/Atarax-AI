import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List
import uuid


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique identifier for the project.")
    name: str = Field(..., description="Name of the project.")
    description: str = Field(..., description="Description of the project.")
    created_at: datetime.datetime = Field(..., description="Creation timestamp of the project.")
    updated_at: datetime.datetime = Field(..., description="Last update timestamp of the project.")
    chat_sessions: Optional[List["ChatSessionResponse"]] = Field(None, description="List of chat sessions associated with the project.")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Project name cannot be empty or contain only whitespace.")
        return v
    
    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Project description cannot be empty or contain only whitespace.")
        return v

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique identifier for the message.")
    session_id: uuid.UUID = Field(..., description="Unique identifier for the session.")
    role: str = Field(..., description="Role of the message sender.")
    content: str = Field(..., description="Content of the message.")
    date_time: datetime.datetime = Field(..., description="Date and time when the message was sent.")
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty or contain only whitespace.")
        return v

    @field_validator("date_time")
    @classmethod
    def validate_date_time(cls, v: datetime.datetime) -> datetime.datetime:
        if v > datetime.datetime.now():
            raise ValueError("Message timestamp cannot be in the future.")
        return v


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique identifier for the chat session.")
    project_id: uuid.UUID = Field(..., description="Unique identifier for the project.")
    title: str = Field(..., description="Title of the chat session.")
    created_at: datetime.datetime = Field(..., description="Creation timestamp of the chat session.")
    updated_at: datetime.datetime = Field(..., description="Last update timestamp of the chat session.")
    messages: Optional[List[MessageResponse]] = Field(default_factory=lambda: [], description="List of messages associated with the chat session.")
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Chat session title cannot be empty or contain only whitespace.")
        return v

    @field_validator("updated_at")
    @classmethod
    def validate_timestamps(cls, v: datetime.datetime, info) -> datetime.datetime:
        if hasattr(info, 'data') and 'created_at' in info.data:
            created_at = info.data['created_at']
            if v < created_at:
                raise ValueError("Updated timestamp cannot be earlier than created timestamp.")
        return v

ProjectResponse.model_rebuild()
ChatSessionResponse.model_rebuild()
