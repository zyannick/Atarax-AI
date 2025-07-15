from pydantic import BaseModel, ConfigDict
from typing import Optional, List
import uuid


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: Optional[str] = None

    chat_sessions: Optional[List["ChatSessionResponse"]] = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    timestamp: float


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    messages: List[MessageResponse]

    
ProjectResponse.model_rebuild()
ChatSessionResponse.model_rebuild()