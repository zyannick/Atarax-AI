from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
import logging
from ataraxai.praxis.utils.input_validator import InputValidator
import uuid
from typing import List
from ataraxai.praxis.modules.chat.chat_models import (
    ChatSessionResponse,
    ProjectResponse,
    MessageResponse,
)
from ataraxai.praxis.utils.vault_manager import VaultManager


class ChatManager:

    def __init__(
        self,
        db_manager: ChatDatabaseManager,
        logger: logging.Logger,
        vault_manager: VaultManager,
    ):
        """
        Initializes the orchestrator with the provided database manager, logger, and an input validator.

        Args:
            db_manager (ChatDatabaseManager): Instance responsible for managing chat database operations.
            logger (ArataxAILogger): Logger instance for recording application events and errors.
        """
        self.db_manager = db_manager
        self.logger = logger
        self.validator = InputValidator()
        self.vault_manager = vault_manager

    async def create_project(self, name: str, description: str) -> ProjectResponse:
        """
        Creates a new project with the specified name and description.

        Validates the project name, attempts to create the project in the database,
        logs the operation, and returns a ProjectResponse object. If an error occurs
        during project creation, logs the error and re-raises the exception.

        Args:
            name (str): The name of the project to create.
            description (str): A description of the project.

        Returns:
            ProjectResponse: The response object containing the created project's details.

        Raises:
            Exception: If project creation fails for any reason.
        """
        self.validator.validate_string(name, "Project name")
        try:
            project = await self.db_manager.create_project(name=name, description=description)
            self.logger.info(f"Created project: {name}")
            return ProjectResponse.model_validate(project)
        except Exception as e:
            self.logger.error(f"Failed to create project {name}: {e}")
            raise

    async def get_project(self, project_id: uuid.UUID) -> ProjectResponse:
        """
        Retrieve a project by its unique identifier.

        Args:
            project_id (uuid.UUID): The unique identifier of the project to retrieve.

        Returns:
            ProjectResponse: The response model containing the project's details.

        Raises:
            Exception: If the project cannot be retrieved or an error occurs during the process.
        """
        self.validator.validate_uuid(project_id, "Project ID")
        try:
            project = await self.db_manager.get_project(project_id)
            return ProjectResponse.model_validate(project)
        except Exception as e:
            self.logger.error(f"Failed to get project {project_id}: {e}")
            raise

    async def list_projects(self) -> List[ProjectResponse]:
        """
        Retrieves a list of all projects from the database.

        Returns:
            List[ProjectResponse]: A list of validated ProjectResponse objects representing the projects.

        Raises:
            Exception: If an error occurs while retrieving the projects, the exception is logged and re-raised.
        """
        try:
            projects = await self.db_manager.list_projects()
            return [ProjectResponse.model_validate(project) for project in projects]
        except Exception as e:
            self.logger.error(f"Failed to list projects: {e}")
            raise

    async def delete_project(self, project_id: uuid.UUID) -> bool:
        """
        Deletes a project from the database by its UUID.

        Args:
            project_id (uuid.UUID): The unique identifier of the project to delete.

        Returns:
            bool: True if the project was successfully deleted, False otherwise.

        Raises:
            Exception: If an error occurs during the deletion process.

        Logs:
            - Info: When a project is successfully deleted.
            - Error: If the deletion fails.
        """
        self.validator.validate_uuid(project_id, "Project ID")
        try:
            result = await self.db_manager.delete_project(project_id)
            if result:
                self.logger.info(f"Deleted project: {project_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete project {project_id}: {e}")
            raise

    async def create_session(self, project_id: uuid.UUID, title: str) -> ChatSessionResponse:
        """
        Creates a new chat session for the specified project.

        Args:
            project_id (uuid.UUID): The unique identifier of the project for which the session is being created.
            title (str): The title of the new chat session.

        Returns:
            ChatSessionResponse: The response object containing details of the created chat session.

        Raises:
            Exception: If session creation fails due to a database or validation error.
        """
        self.validator.validate_uuid(project_id, "Project ID")
        self.validator.validate_string(title, "Session title")
        try:
            session = await self.db_manager.create_session(project_id=project_id, title=title)
            self.logger.info(f"Created session: {title} for project {project_id}")
            return ChatSessionResponse.model_validate(session)
        except Exception as e:
            self.logger.error(f"Failed to create session {title}: {e}")
            raise

    async def get_session(self, session_id: uuid.UUID) -> ChatSessionResponse:
        """
        Retrieves a chat session by its unique identifier.

        Args:
            session_id (uuid.UUID): The unique identifier of the chat session to retrieve.

        Returns:
            ChatSessionResponse: The response model containing the session's details.

        Raises:
            Exception: If the session cannot be retrieved or an error occurs during the process.

        Logs:
            Logs an error message if session retrieval fails.
        """
        self.validator.validate_uuid(session_id, "Session ID")
        try:
            db_session = await self.db_manager.get_session(session_id)
            if not db_session:
                raise ValueError(f"Session not found with identifier {session_id}")
            decrypted_messages: List[MessageResponse] = []
            for msg in db_session.messages:
                decrypted_content: str = self.vault_manager.decrypt(
                    bytes(msg.content)
                ).decode("utf-8")
                decrypted_messages.append(
                    MessageResponse(
                        id=msg.get_id(),
                        session_id=db_session.get_id(),
                        role=msg.get_role(),
                        content=decrypted_content,
                        date_time=msg.get_date_time(),
                    )
                )

            return ChatSessionResponse(
                id=db_session.get_id(),
                project_id=db_session.get_project_id(),
                title=db_session.get_title(),
                messages=decrypted_messages,
                created_at=db_session.get_created_at(),
                updated_at=db_session.get_updated_at(),
            )
        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {e}")
            raise

    async def list_sessions(self, project_id: uuid.UUID) -> List[ChatSessionResponse]:
        """
        Retrieve a list of chat sessions associated with a given project.

        Args:
            project_id (uuid.UUID): The unique identifier of the project for which to list chat sessions.

        Returns:
            List[ChatSessionResponse]: A list of chat session response objects corresponding to the specified project.

        Raises:
            Exception: If an error occurs while retrieving the sessions from the database.

        Logs:
            Logs an error message if session retrieval fails.
        """
        self.validator.validate_uuid(project_id, "Project ID")
        try:
            sessions_db = await self.db_manager.get_sessions_for_project(project_id)
            return [
                ChatSessionResponse.model_validate(session) for session in sessions_db
            ]
        except Exception as e:
            self.logger.error(f"Failed to list sessions for project {project_id}: {e}")
            raise

    async def delete_session(self, session_id: uuid.UUID) -> bool:
        """
        Deletes a session with the specified session ID.

        Args:
            session_id (uuid.UUID): The unique identifier of the session to delete.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.

        Raises:
            Exception: If an error occurs during the deletion process.

        Logs:
            - Info: When a session is successfully deleted.
            - Error: If deletion fails, logs the exception details.
        """
        self.validator.validate_uuid(session_id, "Session ID")
        try:
            result = await self.db_manager.delete_session(session_id)
            if result:
                self.logger.info(f"Deleted session: {session_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            raise

    async def add_message(
        self, session_id: uuid.UUID, role: str, content: str
    ) -> MessageResponse:
        """
        Adds a new message to the specified session.

        Validates the session ID and message content before attempting to add the message
        to the database. If successful, returns a validated MessageResponse object.
        Logs and re-raises any exceptions encountered during the process.

        Args:
            session_id (uuid.UUID): The unique identifier of the session to which the message will be added.
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.

        Returns:
            MessageResponse: The response object containing the added message details.

        Raises:
            Exception: If adding the message to the database fails.
        """
        self.validator.validate_uuid(session_id, "Session ID")
        self.validator.validate_string(content, "Message content")
        try:
            encrypted_content_bytes = self.vault_manager.encrypt(
                content.encode("utf-8")
            )
            db_message = await self.db_manager.add_message(
                session_id=session_id,
                role=role,
                encrypted_content=encrypted_content_bytes,
            )
            response = MessageResponse(
                id=uuid.UUID(str(db_message.id)),
                session_id=uuid.UUID(str(db_message.session.id)),
                role=str(db_message.role),
                content=content,
                date_time=db_message.get_date_time(),  # Assuming timestamp is a datetime object,
            )

            return response
        except Exception as e:
            self.logger.error(f"Failed to add message to session {session_id}: {e}")
            raise

    async def get_message(self, message_id: uuid.UUID) -> MessageResponse:
        """
        Retrieves a message by its unique identifier.

        Args:
            message_id (uuid.UUID): The unique identifier of the message to retrieve.

        Returns:
            MessageResponse: The response model containing the message's details.

        Raises:
            Exception: If the message cannot be retrieved or an error occurs during the process.

        Logs:
            Logs an error message if message retrieval fails.
        """
        self.logger.info(f"Retrieving message with ID: {message_id}")
        self.validator.validate_uuid(message_id, "Message ID")
        try:
            db_message = await self.db_manager.get_message(message_id)
            if not db_message:
                raise ValueError(f"Message with ID {message_id} not found.")

            decrypted_content: str = self.vault_manager.decrypt(
                bytes(db_message.content)
            ).decode("utf-8")

            return MessageResponse(
                id=uuid.UUID(str(db_message.id)),
                session_id=uuid.UUID(str(db_message.session.id)),
                role=str(db_message.role),
                content=decrypted_content,
                date_time=db_message.get_date_time(),
            )
        except Exception as e:
            self.logger.error(f"Failed to get message {message_id}: {e}")
            raise

    async def get_messages_for_session(self, session_id: uuid.UUID) -> List[MessageResponse]:
        """
        Retrieves encrypted messages from the database and returns them decrypted.
        """
        self.validator.validate_uuid(session_id, "Session ID")
        try:
            encrypted_messages = await self.db_manager.get_messages_for_session(session_id)
            decrypted_responses = []
            for msg in encrypted_messages:
                decrypted_content: str = self.vault_manager.decrypt(
                    bytes(msg.content)
                ).decode("utf-8")
                decrypted_responses.append(
                    MessageResponse(
                        id=msg.get_id(),
                        session_id=msg.get_session_id(),
                        role=msg.get_role(),
                        content=decrypted_content,
                        date_time=msg.get_date_time(),
                    )
                )
            return decrypted_responses
        except Exception as e:
            self.logger.error(
                f"Failed to get and decrypt messages for session {session_id}: {e}"
            )
            raise

    async def delete_message(self, message_id: uuid.UUID) -> bool:
        """
        Deletes a message with the specified UUID from the database.

        Args:
            message_id (uuid.UUID): The unique identifier of the message to delete.

        Returns:
            bool: True if the message was successfully deleted, False otherwise.

        Raises:
            Exception: If an error occurs during the deletion process.

        Logs:
            - Info: When a message is successfully deleted.
            - Error: If deletion fails, logs the exception.
        """
        self.validator.validate_uuid(message_id, "Message ID")
        try:
            result = await self.db_manager.delete_message(message_id)
            if result:
                self.logger.info(f"Deleted message: {message_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete message {message_id}: {e}")
            raise
