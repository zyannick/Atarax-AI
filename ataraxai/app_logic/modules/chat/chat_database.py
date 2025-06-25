import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from peewee import (
    SqliteDatabase,
    Model,
    UUIDField,
    CharField,
    TextField,
    DateTimeField,
    ForeignKeyField,
)

db = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = db


class Project(BaseModel):
    id = UUIDField(primary_key=True)
    name = CharField()
    description = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)


class ChatSession(BaseModel):
    id = UUIDField(primary_key=True)
    project = ForeignKeyField(Project, backref="sessions")
    title = CharField()
    created_at = DateTimeField(default=datetime.now)


class Message(BaseModel):
    id = UUIDField(primary_key=True)
    session = ForeignKeyField(ChatSession, backref="messages")
    role = CharField()
    content = TextField()
    timestamp = DateTimeField(default=datetime.now)


class ChatDatabaseManager:
    def __init__(self, db_path: Path):
        db.init(str(db_path))
        db.connect()
        db.create_tables([Project, ChatSession, Message], safe=True)

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        return Project.create(id=uuid.uuid4(), name=name, description=description)

    def list_projects(self):
        return list(Project.select())

    def create_session(self, project_id: uuid.UUID, title: str) -> ChatSession:
        project = Project.get(Project.id == project_id)
        return ChatSession.create(id=uuid.uuid4(), project=project, title=title)

    def get_sessions_for_project(self, project_id: uuid.UUID):
        return list(ChatSession.select().where(ChatSession.project == project_id))

    def add_message(self, session_id: uuid.UUID, role: str, content: str) -> Message:
        session = ChatSession.get(ChatSession.id == session_id)
        return Message.create(
            id=uuid.uuid4(), session=session, role=role, content=content
        )

    def delete_message(self, message_id: uuid.UUID):
        try:
            message = Message.get(Message.id == message_id)
            message.delete_instance()
        except Exception:
            print(f"Message with ID {message_id} does not exist.")

    def get_messages_for_session(self, session_id: uuid.UUID):
        return list(
            Message.select()
            .where(Message.session == session_id)
            .order_by(Message.timestamp)
        )

    def delete_session(self, session_id: uuid.UUID):
        Message.delete().where(Message.session == session_id).execute()
        ChatSession.delete_by_id(session_id)

    def delete_project(self, project_id: uuid.UUID):
        sessions = ChatSession.select().where(ChatSession.project == project_id)
        for s in sessions:
            self.delete_session(s.id)
        Project.delete_by_id(project_id)

    def close(self):
        if not db.is_closed():
            db.close()
