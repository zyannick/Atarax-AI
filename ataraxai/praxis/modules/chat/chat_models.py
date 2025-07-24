import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
import uuid


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    chat_sessions: Optional[List["ChatSessionResponse"]] = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    date_time: datetime.datetime


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    messages: Optional[List[MessageResponse]] = None

ProjectResponse.model_rebuild()
ChatSessionResponse.model_rebuild()
