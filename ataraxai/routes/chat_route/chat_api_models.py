import datetime
import uuid
from typing import List

from pydantic import BaseModel, Field, field_validator

from ataraxai.routes.status import Status


class CreateProjectRequestAPI(BaseModel):
    name: str = Field(..., description="The name of the new project.")
    description: str = Field(..., description="A brief description of the project.")

    @field_validator("name")
    def validate_name(cls, v: str) -> str: 
        if not v:
            raise ValueError("Project name cannot be empty.")
        if len(v) > 32:
            raise ValueError("Project name exceeds maximum length.")
        return v

    @field_validator("description")
    def validate_description(cls, v: str) -> str:
        if not v:
            raise ValueError("Project description cannot be empty.")
        if len(v) > 256:
            raise ValueError("Project description exceeds maximum length.")
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
    title: str = Field(
        default_factory=lambda: "New Session",
        description="The initial title for the new chat session.",
    )

    @field_validator("title")
    def validate_title(cls, v: str) -> str:
        if not v:
            raise ValueError("Session title cannot be empty.")
        if len(v) > 64:
            raise ValueError("Session title exceeds maximum length.")
        return v

    @field_validator("project_id")
    def validate_project_id(cls, v: uuid.UUID) -> uuid.UUID:
        if not v:
            raise ValueError("Project ID cannot be empty.")
        return v

class UpdateSessionRequestAPI(BaseModel):
    title: str = Field(..., description="The new title for the chat session.")

    @field_validator("title")
    def validate_title(cls, v: str) -> str:
        if not v:
            raise ValueError("Session title cannot be empty.")
        if len(v) > 64:
            raise ValueError("Session title exceeds maximum length.")
        return v


class ChatMessageRequestAPI(BaseModel):
    user_query: str = Field(..., description="The user's message to the AI.")

    @field_validator("user_query")
    def validate_user_query(cls, v: str) -> str:
        if not v:
            raise ValueError("User query cannot be empty.")
        return v


class MessageResponseAPI(BaseModel):
    status : Status = Field(default=Status.SUCCESS, description="Status of the message processing.")
    id: uuid.UUID = Field(..., description="Unique identifier for the message.")
    session_id: uuid.UUID = Field(..., description="Unique identifier for the session.")
    role: str = Field(..., description="Role of the message sender.")
    content: str = Field(..., description="Content of the message.")
    date_time: datetime.datetime = Field(
        default_factory=datetime.datetime.now, description="Date and time when the message was sent."
    )

    class Config:
        from_attributes = True


class ProjectResponseAPI(BaseModel):
    status : Status = Field(..., description="Status of the project.")
    project_id: uuid.UUID = Field(..., description="Unique identifier for the project.")
    name: str = Field(..., description="Name of the project.")
    description: str = Field(..., description="Description of the project.")
    created_at: datetime.datetime = Field(
        ..., description="Project creation timestamp."
    )
    updated_at: datetime.datetime = Field(
        ..., description="Project last update timestamp."
    )

class ListProjectsResponseAPI(BaseModel):
    status : Status = Field(..., description="Status of the projects listing.")
    projects: List[ProjectResponseAPI] = Field(
        ..., description="List of projects."
    )

class DeleteProjectResponseAPI(BaseModel):
    project_id: uuid.UUID = Field(..., description="Unique identifier for the project.")
    name: str = Field(..., description="Name of the project.")
    description: str = Field(..., description="Description of the project.")
    status: Status = Field(..., description="Status of the project deletion operation.")


class SessionResponseAPI(BaseModel):
    status : Status = Field(..., description="Status of the session.")
    session_id: uuid.UUID = Field(..., description="Unique identifier for the session.")
    title: str = Field(..., description="Title of the session.")
    project_id: uuid.UUID = Field(..., description="Unique identifier for the project.")
    created_at: datetime.datetime = Field(
        ..., description="Session creation timestamp."
    )
    updated_at: datetime.datetime = Field(
        ..., description="Session last update timestamp."
    )

class ListSessionsResponseAPI(BaseModel):
    status : Status = Field(..., description="Status of the sessions listing.")
    sessions: List[SessionResponseAPI] = Field(
        ..., description="List of sessions."
    )


class DeleteSessionResponseAPI(BaseModel):
    session_id: uuid.UUID = Field(..., description="Unique identifier for the session.")
    title: str = Field(..., description="Title of the session.")
    project_id: uuid.UUID = Field(..., description="Unique identifier for the project.")
    status: Status = Field(..., description="Status of the session deletion operation.")


class ProjectWithSessionsResponse(ProjectResponseAPI):
    status : Status = Field(..., description="Status of the project with sessions.")
    sessions: List[SessionResponseAPI] = Field(
        ..., description="List of sessions in this project."
    )
