import datetime
from pydantic import BaseModel, Field, field_validator
import uuid


class CreateProjectRequestAPI(BaseModel):
    name: str = Field(..., description="The name of the new project.")
    description: str = Field(..., description="A brief description of the project.")

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("Project name cannot be empty.")
        return v

    @field_validator("description")
    def validate_description(cls, v: str) -> str:
        if not v:
            raise ValueError("Project description cannot be empty.")
        return v


class DeleteProjectRequestAPI(BaseModel):
    project_id: uuid.UUID = Field(..., description="The ID of the project to delete.")

    @field_validator("project_id")
    def validate_project_id(cls, v: uuid.UUID) -> uuid.UUID:
        if not v:
            raise ValueError("Project ID cannot be empty.")
        return v


class CreateSessionRequestAPI(BaseModel):
    project_id: uuid.UUID
    title: str = Field(..., description="The initial title for the new chat session.")

    @field_validator("title")
    def validate_title(cls, v: str) -> str:
        if not v:
            raise ValueError("Session title cannot be empty.")
        return v

    @field_validator("project_id")
    def validate_project_id(cls, v: uuid.UUID) -> uuid.UUID:
        if not v:
            raise ValueError("Project ID cannot be empty.")
        return v


class ChatMessageRequestAPI(BaseModel):
    user_query: str = Field(..., description="The user's message to the AI.")

    @field_validator("user_query")
    def validate_user_query(cls, v: str) -> str:
        if not v:
            raise ValueError("User query cannot be empty.")
        return v


class MessageResponseAPI(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    date_time: datetime.datetime

    class Config:
        from_attributes = True


class ProjectResponseAPI(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str


class DeleteProjectResponseAPI(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str
    status: str = Field(..., description="Status of the project deletion operation.")


class SessionResponseAPI(BaseModel):
    session_id: uuid.UUID
    title: str
    project_id: uuid.UUID


class DeleteSessionResponseAPI(BaseModel):
    session_id: uuid.UUID
    title: str
    project_id: uuid.UUID
    status: str = Field(..., description="Status of the session deletion operation.")
