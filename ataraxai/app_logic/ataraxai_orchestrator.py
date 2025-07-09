from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
from enum import Enum
from ataraxai.app_logic.utils.security_manager import SecurityManager

import uuid

from ataraxai import __version__, core_ai_py  # type: ignore
from ataraxai.app_logic.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.app_logic.preferences_manager import PreferencesManager
from ataraxai.app_logic.utils.ataraxai_logger import ArataxAILogger
from ataraxai.app_logic.utils.config_schemas.llama_config_schema import LlamaModelParams
from ataraxai.app_logic.utils.llama_config_manager import LlamaConfigManager
from ataraxai.app_logic.utils.rag_config_manager import RAGConfigManager
from ataraxai.app_logic.utils.whisper_config_manager import WhisperConfigManager
from platformdirs import user_config_dir, user_data_dir, user_cache_dir, user_log_dir

from ataraxai.app_logic.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
from ataraxai.app_logic.modules.prompt_engine.context_manager import ContextManager
from ataraxai.app_logic.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.app_logic.modules.prompt_engine.task_manager import TaskManager
from ataraxai.app_logic.modules.prompt_engine.chain_runner import ChainRunner
from ataraxai.app_logic.utils.config_schemas.whisper_config_schema import (
    WhisperModelParams,
)
from ataraxai.app_logic.modules.chat.chat_models import (
    ChatSessionResponse,
    ProjectResponse,
    MessageResponse,
)
import time
from prometheus_client import Counter


CHAINS_EXECUTED_COUNTER = Counter(
    "ataraxai_chains_executed_total",
    "Total number of task chains executed",
    ["chain_name"],
)

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"


class ServiceStatus(Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"


class AppState(Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    FIRST_LAUNCH = "first_launch"


class AtaraxAIError(Exception):
    pass


class CoreAIServiceError(AtaraxAIError):
    pass


class ServiceInitializationError(AtaraxAIError):
    pass


class ValidationError(AtaraxAIError):
    pass


@dataclass
class AppDirectories:
    config: Path
    data: Path
    cache: Path
    logs: Path

    @classmethod
    def create_default(cls) -> "AppDirectories":
        """
        Creates an instance of AppDirectories with default paths for config, data, cache, and logs
        using the user's operating system conventions. The directories are created if they do not exist.

        Returns:
            AppDirectories: An instance with initialized and created directory paths.
        """
        dirs = cls(
            config=Path(user_config_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
            data=Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
            cache=Path(user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
            logs=Path(user_log_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
        )
        dirs.create_directories()
        return dirs

    def create_directories(self) -> None:
        """
        Creates the necessary directories for configuration, data, cache, and logs.

        This method iterates over the predefined directory paths (self.config, self.data, self.cache, self.logs)
        and ensures that each directory exists by creating it if it does not already exist. Parent directories
        are also created as needed.

        Returns:
            None
        """
        for directory in [self.config, self.data, self.cache, self.logs]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    database_filename: str = "chat_history.sqlite"
    prompts_directory: str = "./prompts"
    setup_marker_filename: str = ".ataraxai_app_{version}_setup_complete"

    def get_setup_marker_filename(self, version: str) -> str:
        """
        Generate the setup marker filename for a specific version.

        Args:
            version (str): The version string to include in the filename.

        Returns:
            str: The formatted setup marker filename for the given version.
        """
        return self.setup_marker_filename.format(version=version)


class InputValidator:

    @staticmethod
    def validate_uuid(uuid_value: Optional[uuid.UUID], param_name: str) -> None:
        """
        Validates that the provided UUID value is not empty.

        Args:
            uuid_value (Optional[uuid.UUID]): The UUID value to validate.
            param_name (str): The name of the parameter being validated, used in the error message.

        Raises:
            ValidationError: If the uuid_value is None or empty.
        """
        if not uuid_value:
            raise ValidationError(f"{param_name} cannot be empty.")

    @staticmethod
    def validate_string(string_value: Optional[str], param_name: str) -> None:
        """
        Validates that the provided string is not None, empty, or only whitespace.

        Args:
            string_value (Optional[str]): The string value to validate.
            param_name (str): The name of the parameter being validated, used in the error message.

        Raises:
            ValidationError: If the string_value is None, empty, or contains only whitespace.
        """
        if not string_value or not string_value.strip():
            raise ValidationError(f"{param_name} cannot be empty.")

    @staticmethod
    def validate_path(
        path_value: Optional[str], param_name: str, must_exist: bool = True
    ) -> None:
        """
        Validates a file or directory path.

        Args:
            path_value (Optional[str]): The path to validate.
            param_name (str): The name of the parameter (used in error messages).
            must_exist (bool, optional): If True, the path must exist. Defaults to True.

        Raises:
            ValidationError: If the path is empty or, when must_exist is True, does not exist.
        """
        if not path_value:
            raise ValidationError(f"{param_name} cannot be empty.")

        path = Path(path_value)
        if must_exist and not path.exists():
            raise ValidationError(f"{param_name} path does not exist: {path_value}")

    @staticmethod
    def validate_directory(directory_path: Optional[str], param_name: str) -> None:
        """
        Validates that the provided directory path is a non-empty string and points to an existing directory.

        Args:
            directory_path (Optional[str]): The path to the directory to validate.
            param_name (str): The name of the parameter being validated, used in error messages.

        Raises:
            ValidationError: If the directory_path is empty or does not point to a valid directory.
        """
        if not directory_path:
            raise ValidationError(f"{param_name} cannot be empty.")

        path = Path(directory_path)
        if not path.is_dir():
            raise ValidationError(
                f"{param_name} is not a valid directory: {directory_path}"
            )


class ConfigurationManager:

    def __init__(self, config_dir: Path, logger: ArataxAILogger):
        """
        Initializes the orchestrator with the specified configuration directory and logger.

        Args:
            config_dir (Path): The directory containing configuration files.
            logger (ArataxAILogger): Logger instance for logging orchestrator activities.
        """
        self.config_dir = config_dir
        self.logger = logger
        self._init_config_managers()

    def _init_config_managers(self) -> None:
        """
        Initializes configuration manager instances for preferences, Llama, Whisper, and RAG components.

        Attempts to create and assign configuration manager objects using the provided configuration directory.
        Logs a success message upon successful initialization. If any exception occurs during initialization,
        logs the error and raises a ServiceInitializationError with details.

        Raises:
            ServiceInitializationError: If any configuration manager fails to initialize.
        """
        try:
            self.preferences = PreferencesManager(config_path=self.config_dir)
            self.llama_config = LlamaConfigManager(config_path=self.config_dir)
            self.whisper_config = WhisperConfigManager(config_path=self.config_dir)
            self.rag_config = RAGConfigManager(config_path=self.config_dir)
            self.logger.info("Configuration managers initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration managers: {e}")
            raise ServiceInitializationError(
                f"Configuration initialization failed: {e}"
            )

    def get_watched_directories(self) -> Optional[List[str]]:
        """
        Retrieves the list of directories being watched for changes as specified in the RAG configuration.

        Returns:
            Optional[List[str]]: A list of directory paths being watched, or None if not specified in the configuration.
        """
        config = self.rag_config.get_config()
        return getattr(config, "rag_watched_directories", None)

    def add_watched_directory(self, directory: str) -> None:
        """
        Adds a directory to the list of watched directories if it is not already present.

        Args:
            directory (str): The path of the directory to add to the watched list.

        Side Effects:
            Updates the "rag_watched_directories" configuration with the new directory if it was not already being watched.
        """
        watched_dirs = self.get_watched_directories()
        if watched_dirs is None:
            watched_dirs = []
        if directory not in watched_dirs:
            watched_dirs.append(directory)
            self.rag_config.set("rag_watched_directories", watched_dirs)


class CoreAIServiceManager:

    def __init__(self, config_manager: ConfigurationManager, logger: ArataxAILogger):
        """
        Initializes the orchestrator with the provided configuration manager and logger.

        Args:
            config_manager (ConfigurationManager): The configuration manager instance to handle configuration settings.
            logger (ArataxAILogger): The logger instance for logging orchestrator activities.

        Attributes:
            service (Optional[Any]): The service instance managed by the orchestrator, initialized as None.
            status (ServiceStatus): The current status of the service, initialized as NOT_INITIALIZED.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.service: Optional[Any] = None
        self.status = ServiceStatus.NOT_INITIALIZED

    def get_service(self) -> Any:
        """
        Retrieves the core AI service instance, initializing it if necessary.
        If the service has not been initialized, this method will initialize it.
        If the previous initialization attempt failed, raises a ServiceInitializationError.
        Returns:S
            Any: The initialized core AI service instance.
        Raises:
            ServiceInitializationError: If the service failed to initialize previously.
        """
        if self.status == ServiceStatus.NOT_INITIALIZED:
            self.initialize()
        elif self.status == ServiceStatus.FAILED:
            raise ServiceInitializationError(
                "Core AI service initialization previously failed"
            )

        return self.service

    def initialize(self) -> None:
        """
        Initializes the core AI services if they are not already initialized.
        This method performs the following steps:
            1. Checks if the services are already initialized and logs a message if so.
            2. Sets the status to INITIALIZING and logs the initialization process.
            3. Validates model paths and initializes required services.
            4. Updates the status to INITIALIZED upon successful completion and logs success.
            5. Handles exceptions by setting the status to FAILED, logging the error, and raising a ServiceInitializationError.
        Raises:
            ServiceInitializationError: If initialization of core AI services fails.
        """
        if self.status == ServiceStatus.INITIALIZED:
            self.logger.info("Core AI services already initialized")
            return

        self.status = ServiceStatus.INITIALIZING
        self.logger.info("Initializing Core AI services...")

        try:
            self._validate_model_paths()
            self._initialize_services()
            self.status = ServiceStatus.INITIALIZED
            self.logger.info("Core AI services initialized successfully")
        except Exception as e:
            self.status = ServiceStatus.FAILED
            self.logger.error(f"Failed to initialize core AI services: {e}")
            raise ServiceInitializationError(
                f"Core AI service initialization failed: {e}"
            )

    def is_configured(self) -> bool:
        """
        Checks if the current configuration is valid by validating model paths.

        Returns:
            bool: True if the model paths are valid and configuration is correct, False otherwise.
        """
        try:
            self._validate_model_paths()
            return True
        except ValidationError:
            return False

    def get_configuration_status(self) -> Dict[str, Any]:
        """
        Retrieves the current configuration and initialization status for Llama and Whisper models.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "llama_configured" (bool): Whether the Llama model is configured.
                - "whisper_configured" (bool): Whether the Whisper model is configured.
                - "llama_model_path" (Optional[str]): Path to the Llama model, if configured.
                - "whisper_model_path" (Optional[str]): Path to the Whisper model, if configured.
                - "llama_path_exists" (bool): Whether the Llama model path exists on disk.
                - "whisper_path_exists" (bool): Whether the Whisper model path exists on disk.
                - "initialization_status" (Any): The current initialization status value.

        Logs warnings if configuration retrieval fails for either model.
        """
        status: Dict[str, Any] = {
            "llama_configured": False,
            "whisper_configured": False,
            "llama_model_path": None,
            "whisper_model_path": None,
            "llama_path_exists": False,
            "whisper_path_exists": False,
            "initialization_status": self.status.value,
        }

        try:
            llama_params = self.config_manager.llama_config.get_llama_cpp_params()
            status["llama_model_path"] = llama_params.model_path
            status["llama_configured"] = bool(llama_params.model_path)
            if llama_params.model_path:
                status["llama_path_exists"] = Path(llama_params.model_path).exists()
        except Exception as e:
            self.logger.warning(f"Could not get Llama configuration: {e}")

        try:
            whisper_params = self.config_manager.whisper_config.get_whisper_params()
            status["whisper_model_path"] = whisper_params.model
            status["whisper_configured"] = bool(whisper_params.model)
            if whisper_params.model:
                status["whisper_path_exists"] = Path(whisper_params.model).exists()
        except Exception as e:
            self.logger.warning(f"Could not get Whisper configuration: {e}")

        return status

    def _validate_model_paths(self) -> None:
        """
        Validates the existence and configuration of model paths for Llama and Whisper models.

        Raises:
            ValidationError: If the Llama or Whisper model path is not configured or does not exist.
        """
        llama_params = self.config_manager.llama_config.get_llama_cpp_params()
        whisper_params = self.config_manager.whisper_config.get_whisper_params()

        if not llama_params.model_path:
            raise ValidationError("Llama model path not configured")

        if not Path(llama_params.model_path).exists():
            raise ValidationError(
                f"Llama model path does not exist: {llama_params.model_path}"
            )

        if not whisper_params.model:
            raise ValidationError("Whisper model path not configured")

        if not Path(whisper_params.model).exists():
            raise ValidationError(
                f"Whisper model path does not exist: {whisper_params.model}"
            )

    def _initialize_services(self) -> None:
        """
        Initializes core AI services by retrieving and converting configuration parameters.

        This method performs the following steps:
            1. Retrieves Llama and Whisper model parameters from the configuration manager.
            2. Converts these parameters into the required formats for model and generation/transcription.
            3. Creates and assigns the core AI service instance using the processed parameters.

        Returns:
            None
        """
        llama_params = self.config_manager.llama_config.get_llama_cpp_params()
        whisper_params = self.config_manager.whisper_config.get_whisper_params()

        (
            llama_model_params_cc,
            llama_generation_params_cc,
            whisper_model_params_cc,
            whisper_transcription_params_cc,
        ) = self._convert_params(llama_params, whisper_params)

        self.service = self._create_core_ai_service(
            llama_model_params_cc, whisper_model_params_cc
        )

    def _convert_params(
        self, llama_params: LlamaModelParams, whisper_params: WhisperModelParams
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Converts Llama and Whisper model parameter objects into their corresponding core_ai_py representations.

        Args:
            llama_params (LlamaModelParams): The Llama model parameters to convert.
            whisper_params (WhisperModelParams): The Whisper model parameters to convert.

        Returns:
            Tuple[Any, Any, Any, Any]: A tuple containing:
                - Converted Llama model parameters (core_ai_py.LlamaModelParams)
                - Converted Llama generation parameters (core_ai_py.GenerationParams)
                - Converted Whisper model parameters (core_ai_py.WhisperModelParams)
                - Converted Whisper transcription parameters (core_ai_py.WhisperGenerationParams)
        """
        llama_model_params_cc: Any = core_ai_py.LlamaModelParams.from_dict(  # type: ignore
            llama_params.model_dump()
        )

        llama_generation_params_cc: Any = core_ai_py.GenerationParams.from_dict(  # type: ignore
            self.config_manager.llama_config.get_generation_params().model_dump()
        )

        whisper_model_params_cc: Any = core_ai_py.WhisperModelParams.from_dict(  # type: ignore
            whisper_params.model_dump()
        )

        whisper_transcription_params_cc: Any = core_ai_py.WhisperGenerationParams.from_dict(  # type: ignore
            self.config_manager.whisper_config.get_transcription_params().model_dump()
        )

        return (
            llama_model_params_cc,
            llama_generation_params_cc,
            whisper_model_params_cc,
            whisper_transcription_params_cc,
        )  # type: ignore

    def _create_core_ai_service(self, llama_params: Any, whisper_params: Any) -> Any:
        """
        Initializes and returns a CoreAIService instance with specified model parameters.

        Args:
            llama_params (Any): Parameters for initializing the Llama model.
            whisper_params (Any): Parameters for initializing the Whisper model.

        Returns:
            Any: An instance of CoreAIService with the Llama and Whisper models initialized.
        """
        service = core_ai_py.CoreAIService()  # type: ignore
        service.initialize_llama_model(llama_params)  # type: ignore
        service.initialize_whisper_model(whisper_params)  # type: ignore
        return service  # type: ignore

    def shutdown(self) -> None:
        """
        Shuts down the core AI services managed by this orchestrator.

        If a service is currently running, attempts to shut it down gracefully.
        Logs the outcome of the shutdown process, including any errors encountered.
        Resets the service reference and updates the service status to NOT_INITIALIZED.
        """
        if self.service:
            try:
                # self.service.shutdown()
                self.logger.info("Core AI services shut down successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down core AI services: {e}")
            finally:
                self.service = None
                self.status = ServiceStatus.NOT_INITIALIZED

    @property
    def is_initialized(self) -> bool:
        """
        Checks if the service has been initialized.

        Returns:
            bool: True if the service status is INITIALIZED, False otherwise.
        """
        return self.status == ServiceStatus.INITIALIZED


class ChatManager:

    def __init__(self, db_manager: ChatDatabaseManager, logger: ArataxAILogger):
        """
        Initializes the orchestrator with the provided database manager, logger, and an input validator.

        Args:
            db_manager (ChatDatabaseManager): Instance responsible for managing chat database operations.
            logger (ArataxAILogger): Logger instance for recording application events and errors.
        """
        self.db_manager = db_manager
        self.logger = logger
        self.validator = InputValidator()

    def create_project(self, name: str, description: str) -> ProjectResponse:
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
            project = self.db_manager.create_project(name=name, description=description)
            self.logger.info(f"Created project: {name}")
            return ProjectResponse.model_validate(project)
        except Exception as e:
            self.logger.error(f"Failed to create project {name}: {e}")
            raise

    def get_project(self, project_id: uuid.UUID) -> ProjectResponse:
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
            project = self.db_manager.get_project(project_id)
            return ProjectResponse.model_validate(project)
        except Exception as e:
            self.logger.error(f"Failed to get project {project_id}: {e}")
            raise

    def list_projects(self) -> List[ProjectResponse]:
        """
        Retrieves a list of all projects from the database.

        Returns:
            List[ProjectResponse]: A list of validated ProjectResponse objects representing the projects.

        Raises:
            Exception: If an error occurs while retrieving the projects, the exception is logged and re-raised.
        """
        try:
            projects = self.db_manager.list_projects()
            return [ProjectResponse.model_validate(project) for project in projects]
        except Exception as e:
            self.logger.error(f"Failed to list projects: {e}")
            raise

    def delete_project(self, project_id: uuid.UUID) -> bool:
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
            result = self.db_manager.delete_project(project_id)
            if result:
                self.logger.info(f"Deleted project: {project_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete project {project_id}: {e}")
            raise

    def create_session(self, project_id: uuid.UUID, title: str) -> ChatSessionResponse:
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
            session = self.db_manager.create_session(project_id=project_id, title=title)
            self.logger.info(f"Created session: {title} for project {project_id}")
            return ChatSessionResponse.model_validate(session)
        except Exception as e:
            self.logger.error(f"Failed to create session {title}: {e}")
            raise

    def list_sessions(self, project_id: uuid.UUID) -> List[ChatSessionResponse]:
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
            sessions = self.db_manager.get_sessions_for_project(project_id)
            return [ChatSessionResponse.model_validate(session) for session in sessions]
        except Exception as e:
            self.logger.error(f"Failed to list sessions for project {project_id}: {e}")
            raise

    def delete_session(self, session_id: uuid.UUID) -> bool:
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
            result = self.db_manager.delete_session(session_id)
            if result:
                self.logger.info(f"Deleted session: {session_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            raise

    def add_message(
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
            message = self.db_manager.add_message(
                session_id=session_id, role=role, content=content
            )
            return MessageResponse.model_validate(message)
        except Exception as e:
            self.logger.error(f"Failed to add message to session {session_id}: {e}")
            raise

    def get_messages_for_session(self, session_id: uuid.UUID) -> List[MessageResponse]:
        """
        Retrieves all messages associated with a given session ID.

        Args:
            session_id (uuid.UUID): The unique identifier of the session.

        Returns:
            List[MessageResponse]: A list of MessageResponse objects corresponding to the session.

        Raises:
            Exception: If there is an error retrieving messages from the database.

        Logs:
            An error message if message retrieval fails.
        """
        self.validator.validate_uuid(session_id, "Session ID")
        try:
            messages = self.db_manager.get_messages_for_session(session_id)
            return [MessageResponse.model_validate(message) for message in messages]
        except Exception as e:
            self.logger.error(f"Failed to get messages for session {session_id}: {e}")
            raise

    def delete_message(self, message_id: uuid.UUID) -> bool:
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
            result = self.db_manager.delete_message(message_id)
            if result:
                self.logger.info(f"Deleted message: {message_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete message {message_id}: {e}")
            raise


class SetupManager:

    def __init__(
        self, directories: AppDirectories, config: AppConfig, logger: ArataxAILogger
    ):
        """
        Initializes the orchestrator with application directories, configuration, and logger.

        Args:
            directories (AppDirectories): Object containing paths to application directories.
            config (AppConfig): Application configuration object.
            logger (ArataxAILogger): Logger instance for logging application events.
        """
        self.directories = directories
        self.config = config
        self.logger = logger
        self.version = __version__
        self._marker_file = (
            self.directories.config
            / self.config.get_setup_marker_filename(self.version)
        )

    def is_first_launch(self) -> bool:
        """
        Checks if this is the first launch by verifying the existence of a marker file.

        Returns:
            bool: True if the marker file does not exist (indicating first launch), False otherwise.
        """
        return not self._marker_file.exists()

    def perform_first_launch_setup(self) -> None:
        """
        Performs the initial setup required during the application's first launch.

        This method checks if the application is being launched for the first time.
        If so, it executes the necessary setup steps and creates a marker to indicate
        that the setup has been completed. If the setup has already been performed,
        it skips the process. Logs the progress and any errors encountered.

        Raises:
            Exception: If any error occurs during the setup process.
        """
        if not self.is_first_launch():
            self.logger.info("Skipping first launch setup - already completed")
            return

        self.logger.info("Performing first launch setup...")
        try:
            self._create_setup_marker()
            self.logger.info("First launch setup completed successfully")
        except Exception as e:
            self.logger.error(f"First launch setup failed: {e}")
            raise

    def _create_setup_marker(self) -> None:
        """
        Creates a marker file to indicate that the setup process has been completed.

        This method attempts to create the marker file specified by `self._marker_file`.
        If the file already exists, a FileExistsError will be raised due to `exist_ok=False`.
        """
        self._marker_file.touch(exist_ok=False)


class AtaraxAIOrchestrator:

    def __init__(self, app_config: Optional[AppConfig] = None):
        """
        Initializes the orchestrator with application configuration, logging, directory structure, and core managers.

        Args:
            app_config (Optional[AppConfig]): Optional application configuration. If not provided, a default AppConfig is used.

        Attributes:
            app_config (AppConfig): The application configuration instance.
            logger (logging.Logger): Logger instance for the application.
            directories (AppDirectories): Handles application directory paths.
            security_manager (SecurityManager): Manages security-related operations, such as salt and check files.
            state (AppState): The current state of the application (e.g., FIRST_LAUNCH or LOCKED).
            setup_manager (SetupManager): Handles setup procedures for the application.
            config_manager (ConfigurationManager): Manages configuration files and settings.
            core_ai_manager (CoreAIServiceManager): Manages core AI services.

        Raises:
            Any exceptions raised by underlying manager initializations or file operations.

        Side Effects:
            Initializes application state and managers, creates directories and files as needed, and sets up logging.
        """
        self.app_config = app_config or AppConfig()
        self.logger = self._init_logger()
        self.directories = AppDirectories.create_default()

        salt_file = self.directories.data / "vault.salt"
        check_file = self.directories.data / "vault.check"
        self.security_manager = SecurityManager(
            salt_path=str(salt_file), check_path=str(check_file)
        )

        if not Path(salt_file).exists():
            self.state = AppState.FIRST_LAUNCH
        else:
            self.state = AppState.LOCKED

        self.setup_manager = SetupManager(
            self.directories, self.app_config, self.logger
        )
        self.config_manager = ConfigurationManager(self.directories.config, self.logger)
        self.core_ai_manager = CoreAIServiceManager(self.config_manager, self.logger)

        self._initialize_application()

    def _init_logger(self) -> ArataxAILogger:
        """
        Initializes and returns an instance of the ArataxAILogger.

        Returns:
            ArataxAILogger: A new logger instance for AtaraxAI operations.
        """
        return ArataxAILogger()

    def _initialize_application(self) -> None:
        try:
            self.logger.info(f"Starting AtaraxAI v{__version__}")

            if self.setup_manager.is_first_launch():
                self.setup_manager.perform_first_launch_setup()

            self._init_database()
            self._init_rag_manager()
            self._init_prompt_engine()

            config_status = self.core_ai_manager.get_configuration_status()
            if (
                config_status["llama_configured"]
                and config_status["whisper_configured"]
            ):
                self.logger.info(
                    "AI services are properly configured and ready for use"
                )
            else:
                self.logger.warning(
                    "AI services are not fully configured - some features may be unavailable"
                )

            self._finalize_setup()

            self.logger.info("AtaraxAI initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize AtaraxAI: {e}")
            raise

    def _init_database(self) -> None:
        """
        Initializes the application's database and related managers.

        This method sets up the database path, creates an instance of the ChatDatabaseManager
        with the appropriate security manager, and initializes the chat context and chat manager.
        Logs a message upon successful initialization.
        """
        db_path = self.directories.data / self.app_config.database_filename
        self.db_manager = ChatDatabaseManager(db_path=db_path, security_manager=self.security_manager)
        self.chat_context = ChatContextManager(db_manager=self.db_manager)
        self.chat_manager = ChatManager(self.db_manager, self.logger)
        self.logger.info("Database initialized successfully")

    def set_master_password(self, password: str):
        """
        Sets the master password for the vault during the first launch.

        This method derives a cryptographic key from the provided password,
        creates a vault check to verify the integrity of the vault, updates
        the application state to UNLOCKED, and initializes services that
        require the vault to be unlocked.

        Args:
            password (str): The master password to set for the vault.

        Raises:
            RuntimeError: If the method is called when the application is not in the FIRST_LAUNCH state.
        """
        if self.state != AppState.FIRST_LAUNCH:
            raise RuntimeError("Cannot set master password on an existing vault.")

        self.security_manager.derive_key(password)
        self.security_manager.create_vault_check()
        self.state = AppState.UNLOCKED
        self._initialize_unlocked_services()

    def unlock(self, password: str):
        """
        Unlocks the application vault using the provided password.

        This method attempts to unlock the application by deriving a key from the given password
        and verifying its validity. If the password is correct, the application state is set to
        UNLOCKED and necessary services are initialized. If the password is invalid or an error
        occurs, the application remains in the LOCKED state and an error is logged.

        Args:
            password (str): The password used to unlock the vault.

        Raises:
            ValueError: If the provided password is invalid.
            Exception: If any other error occurs during the unlocking process.
        """
        if self.state != AppState.LOCKED:
            self.logger.warning("Application is already unlocked.")
            return

        try:
            self.security_manager.derive_key(password)

            if not self.security_manager.verify_password():
                self.security_manager.lock()
                raise ValueError("Invalid password.")

            self.state = AppState.UNLOCKED
            self._initialize_unlocked_services()
            self.logger.info("Vault unlocked successfully.")
        except Exception as e:
            self.state = AppState.LOCKED
            self.logger.error(f"Failed to unlock vault: {e}")
            raise

    def _initialize_unlocked_services(self):
        pass

    def lock(self):
        """
        Locks the application by invoking the security manager's lock mechanism, shutting down operations,
        updating the application state to LOCKED, and logging the action.
        """
        self.security_manager.lock()
        self.shutdown()
        self.state = AppState.LOCKED
        self.logger.info("Vault locked.")

    def _init_rag_manager(self) -> None:
        """
        Initializes the RAG (Retrieval-Augmented Generation) manager for the application.

        This method creates an instance of `AtaraxAIRAGManager` using the current RAG configuration,
        the application's data root path, and sets the core AI service to `None`. It assigns the
        instance to `self.rag_manager` and logs a message indicating successful initialization.
        """
        self.rag_manager = AtaraxAIRAGManager(
            rag_config_manager=self.config_manager.rag_config,
            app_data_root_path=self.directories.data,
            core_ai_service=None,
        )
        self.logger.info("RAG manager initialized successfully")

    def _init_prompt_engine(self) -> None:
        """
        Initializes the prompt engine and its related components.

        This method performs the following actions:
        - Ensures the prompts directory exists.
        - Initializes the PromptManager with the prompts directory.
        - Initializes the ContextManager with the current RAG configuration and RAG manager.
        - Initializes the TaskManager.
        - Initializes the ChainRunner with the task manager, context manager, prompt manager, chat context, and RAG manager.
        - Logs the successful initialization of the prompt engine.
        """
        prompts_dir = Path(self.app_config.prompts_directory)
        prompts_dir.mkdir(exist_ok=True)

        self.prompt_manager = PromptManager(prompts_directory=prompts_dir)
        self.context_manager = ContextManager(
            config=self.config_manager.rag_config.get_config().model_dump(),
            rag_manager=self.rag_manager,
        )
        self.task_manager = TaskManager()
        self.chain_runner = ChainRunner(
            task_manager=self.task_manager,
            context_manager=self.context_manager,
            prompt_manager=self.prompt_manager,
            core_ai_service=None,
            chat_context=self.chat_context,
            rag_manager=self.rag_manager,
        )
        self.logger.info("Prompt engine initialized successfully")

    def _finalize_setup(self) -> None:
        """
        Finalizes the setup process by validating and initializing the RAG index, 
        performing an initial scan or rebuilding the index as necessary, and 
        starting file monitoring on the configured watched directories.

        Steps:
        1. Retrieves the list of directories to watch from the configuration manager.
        2. Checks if the RAG index is valid:
            - If invalid, rebuilds the index using the watched directories.
            - If valid, performs an initial scan of the watched directories.
        3. Starts monitoring the watched directories for file changes.
        4. Logs the completion of the setup finalization process.
        """
        watched_dirs = self.config_manager.get_watched_directories()

        is_valid = self.rag_manager.manifest.is_valid(self.rag_manager.rag_store)
        if not is_valid:
            self.logger.info("RAG index is invalid, rebuilding...")
            self.rag_manager.rebuild_index(watched_dirs)
        else:
            self.logger.info("RAG index is valid, performing initial scan...")
            self.rag_manager.perform_initial_scan(watched_dirs)

        self.rag_manager.start_file_monitoring(watched_dirs)
        self.logger.info("Setup finalization completed")

    def run_task_chain(
        self, chain_definition: List[Dict[str, Any]], initial_user_query: str
    ) -> Any:
        """
        Executes a sequence of tasks defined in a chain for a given user query.

        Args:
            chain_definition (List[Dict[str, Any]]): A list of dictionaries, each representing a task in the chain.
            initial_user_query (str): The initial query provided by the user to start the chain.

        Returns:
            Any: The result of executing the task chain.

        Raises:
            ValidationError: If the initial user query is invalid or the chain definition is empty.
            ServiceInitializationError: If the core AI service cannot be initialized.
            Exception: If any error occurs during chain execution.

        Logs:
            - Information about the start and successful completion of the chain execution.
            - Errors encountered during service initialization or chain execution.
        """
        InputValidator.validate_string(initial_user_query, "Initial user query")

        if not chain_definition:
            raise ValidationError("Chain definition cannot be empty")

        try:
            core_ai_service = self.core_ai_manager.get_service()
        except ServiceInitializationError as e:
            self.logger.error(f"Cannot run task chain: {e}")
            raise

        if self.chain_runner.core_ai_service is None:
            self.chain_runner.core_ai_service = core_ai_service

        if self.rag_manager.core_ai_service is None:  # type: ignore
            self.rag_manager.core_ai_service = core_ai_service

        self.logger.info(f"Executing chain for query: '{initial_user_query}'")
        try:
            chain_name = chain_definition[0].get("task_id", "unknown")
            CHAINS_EXECUTED_COUNTER.labels(chain_name=chain_name).inc()
            result = self.chain_runner.run_chain(
                chain_definition=chain_definition, initial_user_query=initial_user_query
            )
            self.logger.info("Chain execution completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Chain execution failed: {e}")
            raise

    def add_watch_directory(self, directory: str) -> None:
        """
        Adds a directory to the list of watched directories and starts monitoring it for file changes.

        Args:
            directory (str): The path to the directory to be watched.

        Raises:
            ValueError: If the provided directory path is invalid.

        Side Effects:
            - Updates the configuration with the new watched directory.
            - Starts or updates file monitoring for all watched directories.
            - Logs the addition of the new watch directory.
        """
        InputValidator.validate_directory(directory, "Directory path")

        self.config_manager.add_watched_directory(directory)
        watched_dirs = self.config_manager.get_watched_directories()
        self.rag_manager.start_file_monitoring(watched_dirs)
        self.logger.info(f"Added watch directory: {directory}")

    def create_project(self, name: str, description: str) -> ProjectResponse:
        """
        Creates a new project with the given name and description.

        Args:
            name (str): The name of the project to create.
            description (str): A brief description of the project.

        Returns:
            ProjectResponse: An object containing details about the newly created project.
        """
        return self.chat_manager.create_project(name, description)

    def get_project(self, project_id: uuid.UUID) -> ProjectResponse:
        """
        Retrieve a project by its unique identifier.

        Args:
            project_id (uuid.UUID): The unique identifier of the project to retrieve.

        Returns:
            ProjectResponse: The response object containing project details.

        Raises:
            Any exceptions raised by chat_manager.get_project.
        """
        return self.chat_manager.get_project(project_id)

    def get_project_summary(self, project_id: uuid.UUID) -> Dict[str, Any]:
        """
        Retrieves a summary of the specified project.

        Args:
            project_id (uuid.UUID): The unique identifier of the project.

        Returns:
            Dict[str, Any]: A dictionary containing the project's summary information.

        Raises:
            Exception: If an error occurs while retrieving the project summary.
        """
        InputValidator.validate_uuid(project_id, "Project ID")
        try:
            return self.db_manager.get_project_summary(project_id)
        except Exception as e:
            self.logger.error(f"Failed to get project summary {project_id}: {e}")
            raise

    def list_projects(self) -> List[ProjectResponse]:
        """
        Retrieves a list of available projects.

        Returns:
            List[ProjectResponse]: A list containing project response objects representing the available projects.
        """
        return self.chat_manager.list_projects()

    def delete_project(self, project_id: uuid.UUID) -> bool:
        """
        Deletes a project with the specified project ID.

        Args:
            project_id (uuid.UUID): The unique identifier of the project to delete.

        Returns:
            bool: True if the project was successfully deleted, False otherwise.
        """
        return self.chat_manager.delete_project(project_id)

    def create_session(self, project_id: uuid.UUID, title: str) -> ChatSessionResponse:
        """
        Creates a new chat session for the specified project.

        Args:
            project_id (uuid.UUID): The unique identifier of the project for which the chat session is to be created.
            title (str): The title of the new chat session.

        Returns:
            ChatSessionResponse: An object containing details about the newly created chat session.
        """
        return self.chat_manager.create_session(project_id, title)

    def list_sessions(self, project_id: uuid.UUID) -> List[ChatSessionResponse]:
        """
        Retrieve a list of chat sessions associated with a given project.

        Args:
            project_id (uuid.UUID): The unique identifier of the project for which to list chat sessions.

        Returns:
            List[ChatSessionResponse]: A list of chat session response objects corresponding to the specified project.
        """
        return self.chat_manager.list_sessions(project_id)

    def delete_session(self, session_id: uuid.UUID) -> bool:
        """
        Deletes a chat session with the specified session ID.

        Args:
            session_id (uuid.UUID): The unique identifier of the session to be deleted.

        Returns:
            bool: True if the session was successfully deleted, False otherwise.
        """
        return self.chat_manager.delete_session(session_id)

    def add_message(
        self, session_id: uuid.UUID, role: str, content: str
    ) -> MessageResponse:
        """
        Adds a message to the chat session.

        Args:
            session_id (uuid.UUID): The unique identifier of the chat session.
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message to be added.

        Returns:
            MessageResponse: The response object containing details of the added message.
        """
        return self.chat_manager.add_message(session_id, role, content)

    def get_messages_for_session(self, session_id: uuid.UUID) -> List[MessageResponse]:
        """
        Retrieve all messages associated with a given chat session.

        Args:
            session_id (uuid.UUID): The unique identifier of the chat session.

        Returns:
            List[MessageResponse]: A list of message responses for the specified session.
        """
        return self.chat_manager.get_messages_for_session(session_id)

    def delete_message(self, message_id: uuid.UUID) -> bool:
        """
        Deletes a message with the specified message ID.

        Args:
            message_id (uuid.UUID): The unique identifier of the message to be deleted.

        Returns:
            bool: True if the message was successfully deleted, False otherwise.
        """
        return self.chat_manager.delete_message(message_id)

    def shutdown(self) -> None:
        """
        Shuts down the AtaraxAI application by stopping file monitoring, closing the database connection,
        and shutting down the core AI manager. Logs the shutdown process and handles any exceptions that occur.
        """
        self.logger.info("Shutting down AtaraxAI...")
        try:
            self.rag_manager.stop_file_monitoring()
            self.db_manager.close()
            self.core_ai_manager.shutdown()
            self.logger.info("AtaraxAI shutdown completed successfully")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def __enter__(self) -> "AtaraxAIOrchestrator":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.shutdown()


if __name__ == "__main__":
    with AtaraxAIOrchestrator() as orchestrator:
        projects = orchestrator.list_projects()
        current_user_home = Path.home()
        orchestrator.add_watch_directory(str(current_user_home / "Documents"))
        while True:
            time.sleep(1)
        print(f"Found {len(projects)} projects")
