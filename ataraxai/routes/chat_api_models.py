from pydantic import BaseModel, Field
import uuid


class CreateProjectRequest(BaseModel):
    name: str = Field(..., description="The name of the new project.")
    description: str = Field(..., description="A brief description of the project.")

class DeleteProjectRequest(BaseModel):
    project_id: uuid.UUID = Field(..., description="The ID of the project to delete.")
    

class CreateSessionRequest(BaseModel):
    project_id: uuid.UUID
    title: str = Field(..., description="The initial title for the new chat session.")


class ChatMessageRequest(BaseModel):
    session_id: uuid.UUID
    user_query: str = Field(..., description="The user's message to the AI.")


class MessageResponse(BaseModel):
    assistant_response: str = Field(..., description="The AI's generated response.")
    session_id: uuid.UUID = Field(..., description="The ID of the chat session.")


class ProjectResponse(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str
    
class DeleteProjectResponse(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str
    status: str = Field(..., description="Status of the project deletion operation.")


class SessionResponse(BaseModel):
    session_id: uuid.UUID
    title: str
    project_id: uuid.UUID

class DeleteSessionResponse(BaseModel):
    session_id: uuid.UUID
    title: str
    project_id: uuid.UUID
    status: str = Field(..., description="Status of the session deletion operation.")
