import asyncio
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from peewee import (
    BlobField,
    CharField,
    DateTimeField,
    DoesNotExist,
    ForeignKeyField,
    IntegrityError,
    Model,
    Select,
    SqliteDatabase,
    TextField,
    UUIDField,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = db


class Project(BaseModel):
    id = UUIDField(primary_key=True)
    name = CharField(null=False)
    description = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    sessions = List["ChatSession"]  # type: ignore

    def get_id(self) -> uuid.UUID:
        return uuid.UUID(str(self.id))

    def get_created_at(self) -> datetime:
        return datetime(
            self.created_at.year,
            self.created_at.month,
            self.created_at.day,
            self.created_at.hour,
            self.created_at.minute,
            self.created_at.second,
        )

    def get_updated_at(self) -> datetime:
        return datetime(
            self.updated_at.year,
            self.updated_at.month,
            self.updated_at.day,
            self.updated_at.hour,
            self.updated_at.minute,
            self.updated_at.second,
        )

    def get_name(self) -> str:
        return str(self.name)

    def get_description(self) -> Optional[str]:
        return str(self.description) if self.description else None

    def save(self, *args, **kwargs) -> int:  # type: ignore
        self.updated_at = datetime.now()  # type: ignore
        return super().save(*args, **kwargs)  # type: ignore

    def __str__(self):
        return f"Project(id={self.id}, name='{self.name}')"


class ChatSession(BaseModel):
    id = UUIDField(primary_key=True)
    project = ForeignKeyField(Project, backref="sessions", on_delete="CASCADE")
    title = CharField()
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    messages: List["Message"] = []  # type: ignore

    def get_id(self) -> uuid.UUID:
        return uuid.UUID(str(self.id))

    def get_created_at(self) -> datetime:
        return datetime(self.created_at.year, self.created_at.month, self.created_at.day, self.created_at.hour, self.created_at.minute, self.created_at.second)  # type: ignore

    def get_updated_at(self) -> datetime:
        return datetime(self.updated_at.year, self.updated_at.month, self.updated_at.day, self.updated_at.hour, self.updated_at.minute, self.updated_at.second)  # type: ignore

    def get_title(self) -> str:
        return str(self.title)

    def get_project_id(self) -> uuid.UUID:
        return uuid.UUID(str(self.project.get_id()))  # type: ignore

    def save(self, *args, **kwargs):  # type: ignore
        self.updated_at = datetime.now()
        return super().save(*args, **kwargs)  # type: ignore

    def __str__(self):
        return f"ChatSession(id={self.id}, title='{self.title}', project_id={self.project.id})"  # type: ignore


class Message(BaseModel):
    id = UUIDField(primary_key=True)
    session = ForeignKeyField(ChatSession, backref="messages", on_delete="CASCADE")
    role = CharField()
    content = BlobField()
    date_time = DateTimeField(default=datetime.now)

    def get_date_time(self) -> datetime:
        return datetime(
            self.date_time.year,
            self.date_time.month,
            self.date_time.day,
            self.date_time.hour,
            self.date_time.minute,
            self.date_time.second,
        )

    def get_id(self) -> uuid.UUID:
        return uuid.UUID(str(self.id))

    def get_session_id(self) -> uuid.UUID:
        return uuid.UUID(str(self.session.get_id()))  # type: ignore

    def get_role(self) -> str:
        return str(self.role)

    def get_content(self) -> bytes:
        return bytes(self.content)

    def get_timestamp(self) -> datetime:
        return self.date_time  # type: ignore


class DatabaseError(Exception):
    pass


class NotFoundError(DatabaseError):
    pass


class ValidationError(DatabaseError):
    pass


class BaseService:

    def __init__(self, model_class: type[Project] | type[ChatSession] | type[Message]):
        self.model = model_class

    def _handle_does_not_exist(self, operation: str, identifier: Any) -> None:
        raise NotFoundError(
            f"{self.model.__name__} {operation} failed: not found with identifier {identifier}"
        )


class ProjectService(BaseService):
    def __init__(self):
        super().__init__(Project)

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        try:
            if not name or not name.strip():
                raise ValidationError("Project name cannot be empty")

            return self.model.create(  # type: ignore
                id=uuid.uuid4(),
                name=name.strip(),
                description=description.strip() if description else None,
            )
        except IntegrityError:
            raise ValidationError(f"Project with name '{name}' already exists")
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise DatabaseError(f"Failed to create project: {e}")

    def get_project(self, project_id: uuid.UUID) -> Project:
        try:
            return self.model.get(self.model.id == project_id)  # type: ignore
        except DoesNotExist:
            self._handle_does_not_exist("get", project_id)
            raise NotFoundError(
                f"Project get failed: not found with identifier {project_id}"
            )

    def get_project_by_name(self, name: str) -> Project:
        try:
            return self.model.get(self.model.name == name)  # type: ignore
        except DoesNotExist:
            self._handle_does_not_exist("get", f"name='{name}'")
            raise NotFoundError(
                f"Project get by name failed: not found with name '{name}'"
            )

    def get_projects(self, limit: Optional[int] = None) -> List[Project]:
        try:
            query: Select = self.model.select().order_by(self.model.created_at.desc())  # type: ignore
            if limit:
                query = query.limit(limit)  # type: ignore
            return list(query)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to get projects: {e}")
            raise DatabaseError(f"Failed to get projects: {e}")

    def update_project(
        self,
        project_id: uuid.UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Project:
        try:
            project = self.get_project(project_id)

            if name is not None:
                if not name.strip():
                    raise ValidationError("Project name cannot be empty")
                setattr(project, "name", name.strip())

            if description is not None:
                setattr(
                    project, "description", description.strip() if description else None
                )

            project.save()  # type: ignore
            return project
        except NotFoundError:
            raise
        except IntegrityError:
            raise ValidationError(f"Project with name '{name}' already exists")
        except Exception as e:
            logger.error(f"Failed to update project: {e}")
            raise DatabaseError(f"Failed to update project: {e}")

    def delete_project(self, project_id: uuid.UUID) -> bool:
        try:
            with db.atomic():  # type: ignore
                project = self.get_project(project_id)
                sessions: Select = ChatSession.select().where(ChatSession.project == project_id)  # type: ignore
                for session in sessions:  # type: ignore
                    Message.delete().where(Message.session == session.id).execute()  # type: ignore
                ChatSession.delete().where(ChatSession.project == project_id).execute()  # type: ignore
                project.delete_instance()  # type: ignore
                return True
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            raise DatabaseError(f"Failed to delete project: {e}")

    def search_projects(self, query: str) -> List[Project]:
        try:
            return list(
                self.model.select()
                .where(  # type: ignore
                    (self.model.name.contains(query))  # type: ignore
                    | (self.model.description.contains(query))  # type: ignore
                )
                .order_by(self.model.created_at.desc())  # type: ignore
            )
        except Exception as e:
            logger.error(f"Failed to search projects: {e}")
            raise DatabaseError(f"Failed to search projects: {e}")


class ChatSessionService(BaseService):
    def __init__(self):
        super().__init__(ChatSession)

    def create_session(self, project_id: uuid.UUID, title: str) -> ChatSession:
        try:
            if not title or not title.strip():
                raise ValidationError("Session title cannot be empty")

            project = Project.get(Project.id == project_id)  # type: ignore

            return self.model.create(  # type: ignore
                id=uuid.uuid4(), project=project, title=title.strip()
            )
        except DoesNotExist:
            raise NotFoundError(f"Project not found with id {project_id}")
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise DatabaseError(f"Failed to create session: {e}")

    def get_session(self, session_id: uuid.UUID) -> ChatSession:
        try:
            return self.model.get(self.model.id == session_id)  # type: ignore
        except DoesNotExist:
            self._handle_does_not_exist("get", session_id)
            raise NotFoundError(
                f"Session get failed: not found with identifier {session_id}"
            )

    def get_sessions_for_project(
        self, project_id: uuid.UUID, limit: Optional[int] = None
    ) -> List[ChatSession]:
        try:
            query = (
                self.model.select()  # type: ignore
                .where(self.model.project == project_id)  # type: ignore
                .order_by(self.model.created_at.desc())  # type: ignore
            )

            if limit:
                query = query.limit(limit)  # type: ignore

            return list(query)
        except Exception as e:
            logger.error(f"Failed to get sessions for project: {e}")
            raise DatabaseError(f"Failed to get sessions for project: {e}")

    def update_session(self, session_id: uuid.UUID, title: str) -> ChatSession:
        try:
            if not title or not title.strip():
                raise ValidationError("Session title cannot be empty")

            session = self.get_session(session_id)
            setattr(session, "title", title.strip())
            session.save()  # type: ignore
            return session
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            raise DatabaseError(f"Failed to update session: {e}")

    def delete_session(self, session_id: uuid.UUID) -> bool:
        try:
            with db.atomic():  # type: ignore
                session = self.get_session(session_id)
                Message.delete().where(Message.session == session_id).execute()  # type: ignore
                session.delete_instance()  # type: ignore
                return True
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            raise DatabaseError(f"Failed to delete session: {e}")

    def get_session_stats(self, session_id: uuid.UUID) -> Dict[str, Any]:
        try:
            session = self.get_session(session_id)
            message_count: int = Message.select().where(Message.session == session_id).count()  # type: ignore
            return {
                "session_id": session_id,
                "title": session.title,
                "message_count": message_count,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            raise DatabaseError(f"Failed to get session stats: {e}")


class MessageService(BaseService):
    def __init__(self):
        super().__init__(Message)

    def create_message(
        self, session_id: uuid.UUID, role: str, content: Union[str, bytes]
    ) -> Message:
        try:

            if role not in ["user", "assistant", "system"]:
                raise ValidationError("Role must be 'user', 'assistant', or 'system'")

            session: ChatSession = ChatSession.get(ChatSession.id == session_id)  # type: ignore

            return self.model.create(  # type: ignore
                id=uuid.uuid4(), session=session, role=role, content=content
            )
        except DoesNotExist:
            raise NotFoundError(f"Session not found with id {session_id}")
        except Exception as e:
            logger.error(f"Failed to create message: {e}")
            raise DatabaseError(f"Failed to create message: {e}")

    def get_message(self, message_id: uuid.UUID) -> Message:
        try:
            return self.model.get(self.model.id == message_id)  # type: ignore
        except DoesNotExist:
            self._handle_does_not_exist("get", message_id)
            raise NotFoundError(
                f"Message get failed: not found with identifier {message_id}"
            )
        except Exception as e:
            logger.error(f"Failed to get message: {e}")
            raise DatabaseError(f"Failed to get message: {e}")

    def get_messages_for_session(
        self, session_id: uuid.UUID, limit: Optional[int] = None
    ) -> List[Message]:
        try:
            query = (
                self.model.select()  # type: ignore
                .where(self.model.session == session_id)  # type: ignore
                .order_by(self.model.date_time.asc())  # type: ignore
            )

            if limit:
                query = query.limit(limit)  # type: ignore

            return list(query)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to get messages for session: {e}")
            raise DatabaseError(f"Failed to get messages for session: {e}")

    def update_message(
        self,
        message_id: uuid.UUID,
        role: Optional[str] = None,
        content: Optional[bytes] = None,
    ) -> Message:
        try:
            message = self.get_message(message_id)

            if role is not None:
                if role not in ["user", "assistant", "system"]:
                    raise ValidationError(
                        "Role must be 'user', 'assistant', or 'system'"
                    )
                setattr(message, "role", role)

            if content is not None:
                setattr(message, "content", content.strip())

            message.save()  # type: ignore
            return message
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update message: {e}")
            raise DatabaseError(f"Failed to update message: {e}")

    def delete_message(self, message_id: uuid.UUID) -> bool:
        try:
            message = self.get_message(message_id)
            message.delete_instance()  # type: ignore
            return True
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
            raise DatabaseError(f"Failed to delete message: {e}")


class ChatDatabaseManager:

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._initialize_database()

        self.project_service = ProjectService()
        self.session_service = ChatSessionService()
        self.message_service = MessageService()

    def _initialize_database(self):
        try:
            db.init(str(self.db_path))
            db.connect()
            db.create_tables([Project, ChatSession, Message])
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}")

    @contextmanager
    def _db_connection(self):
        db = SqliteDatabase(self.db_path)
        try:
            db.connect()
            yield db
        finally:
            if not db.is_closed():
                db.close()

    async def create_project(
        self, name: str, description: Optional[str] = None
    ) -> Project:
        def _create():
            with self._db_connection():
                return self.project_service.create_project(name, description)

        return await asyncio.to_thread(_create)

    async def get_project(self, project_id: uuid.UUID) -> Project:
        def _get():
            with self._db_connection():
                return self.project_service.get_project(project_id)

        return await asyncio.to_thread(_get)

    async def list_projects(self, limit: Optional[int] = None) -> List[Project]:
        def _list():
            with self._db_connection():
                return self.project_service.get_projects(limit)

        return await asyncio.to_thread(_list)

    async def update_project(
        self,
        project_id: uuid.UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Project:
        def _update():
            with self._db_connection():
                return self.project_service.update_project(
                    project_id, name, description
                )

        return await asyncio.to_thread(_update)

    async def delete_project(self, project_id: uuid.UUID) -> bool:
        def _delete():
            with self._db_connection():
                return self.project_service.delete_project(project_id)

        return await asyncio.to_thread(_delete)

    async def create_session(self, project_id: uuid.UUID, title: str) -> ChatSession:
        def _create():
            with self._db_connection():
                return self.session_service.create_session(project_id, title)

        return await asyncio.to_thread(_create)

    async def get_session(self, session_id: uuid.UUID) -> ChatSession:
        def _get():
            with self._db_connection():
                return self.session_service.get_session(session_id)

        return await asyncio.to_thread(_get)

    async def get_sessions_for_project(
        self, project_id: uuid.UUID, limit: Optional[int] = None
    ) -> List[ChatSession]:
        def _get_sessions():
            with self._db_connection():
                return self.session_service.get_sessions_for_project(project_id, limit)

        return await asyncio.to_thread(_get_sessions)

    async def update_session(self, session_id: uuid.UUID, title: str) -> ChatSession:
        def _update():
            with self._db_connection():
                return self.session_service.update_session(session_id, title)

        return await asyncio.to_thread(_update)

    async def delete_session(self, session_id: uuid.UUID) -> bool:
        def _delete():
            with self._db_connection():
                return self.session_service.delete_session(session_id)

        return await asyncio.to_thread(_delete)

    async def add_message(
        self, session_id: uuid.UUID, role: str, encrypted_content: Union[str, bytes]
    ) -> Message:
        if not encrypted_content:
            raise ValidationError("Message content cannot be empty")

        def _add_message():
            with self._db_connection():
                return self.message_service.create_message(
                    session_id, role, encrypted_content
                )

        return await asyncio.to_thread(_add_message)

    async def get_message(self, message_id: uuid.UUID) -> Message:
        def _get_message():
            with self._db_connection():
                return self.message_service.get_message(message_id)

        return await asyncio.to_thread(_get_message)

    async def get_messages_for_session(
        self, session_id: uuid.UUID, limit: Optional[int] = None
    ) -> List[Message]:
        def _get_messages():
            with self._db_connection():
                return self.message_service.get_messages_for_session(session_id, limit)

        return await asyncio.to_thread(_get_messages)

    async def update_message(
        self,
        message_id: uuid.UUID,
        role: Optional[str] = None,
        content: Optional[bytes] = None,
    ) -> Message:

        if content is not None and not content:
            raise DatabaseError("Message content cannot be empty")

        def _update():
            with self._db_connection():
                return self.message_service.update_message(message_id, role, content)

        return await asyncio.to_thread(_update)

    async def delete_message(self, message_id: uuid.UUID) -> bool:
        def _delete():
            with self._db_connection():
                return self.message_service.delete_message(message_id)

        return await asyncio.to_thread(_delete)

    async def get_conversation_history(
        self, session_id: uuid.UUID, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        try:
            messages = await self.get_messages_for_session(session_id, limit)
            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.date_time.isoformat(),  # type: ignore
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            raise DatabaseError(f"Failed to get conversation history: {e}")

    async def get_project_summary(self, project_id: uuid.UUID) -> Dict[str, Any]:
        try:
            project = await self.get_project(project_id)
            sessions = await self.get_sessions_for_project(project_id)

            total_messages = 0
            for session in sessions:
                total_messages += await asyncio.to_thread(Message.select().where(Message.session == session.id).count)  # type: ignore

            return {
                "project": {
                    "id": str(project.id),
                    "name": project.name,
                    "description": project.description,
                    "created_at": project.created_at.isoformat(),  # type: ignore
                    "updated_at": project.updated_at.isoformat(),  # type: ignore
                },
                "stats": {
                    "session_count": len(sessions),
                    "message_count": total_messages,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get project summary: {e}")
            raise DatabaseError(f"Failed to get project summary: {e}")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.close()
