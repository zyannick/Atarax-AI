from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
from enum import Enum
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
    'ataraxai_chains_executed_total',
    'Total number of task chains executed',
    ['chain_name']  
)

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"


class ServiceStatus(Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"


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
        dirs = cls(
            config=Path(user_config_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
            data=Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
            cache=Path(user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
            logs=Path(user_log_dir(appname=APP_NAME, appauthor=APP_AUTHOR)),
        )
        dirs.create_directories()
        return dirs

    def create_directories(self) -> None:
        for directory in [self.config, self.data, self.cache, self.logs]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    database_filename: str = "chat_history.sqlite"
    prompts_directory: str = "./prompts"
    setup_marker_filename: str = ".ataraxai_app_{version}_setup_complete"

    def get_setup_marker_filename(self, version: str) -> str:
        return self.setup_marker_filename.format(version=version)


class InputValidator:

    @staticmethod
    def validate_uuid(uuid_value: Optional[uuid.UUID], param_name: str) -> None:
        if not uuid_value:
            raise ValidationError(f"{param_name} cannot be empty.")

    @staticmethod
    def validate_string(string_value: Optional[str], param_name: str) -> None:
        if not string_value or not string_value.strip():
            raise ValidationError(f"{param_name} cannot be empty.")

    @staticmethod
    def validate_path(
        path_value: Optional[str], param_name: str, must_exist: bool = True
    ) -> None:
        if not path_value:
            raise ValidationError(f"{param_name} cannot be empty.")

        path = Path(path_value)
        if must_exist and not path.exists():
            raise ValidationError(f"{param_name} path does not exist: {path_value}")

    @staticmethod
    def validate_directory(directory_path: Optional[str], param_name: str) -> None:
        if not directory_path:
            raise ValidationError(f"{param_name} cannot be empty.")

        path = Path(directory_path)
        if not path.is_dir():
            raise ValidationError(
                f"{param_name} is not a valid directory: {directory_path}"
            )


class ConfigurationManager:

    def __init__(self, config_dir: Path, logger: ArataxAILogger):
        self.config_dir = config_dir
        self.logger = logger
        self._init_config_managers()

    def _init_config_managers(self) -> None:
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
        config = self.rag_config.get_config()
        return getattr(config, "rag_watched_directories", None)

    def add_watched_directory(self, directory: str) -> None:
        watched_dirs = self.get_watched_directories()
        if watched_dirs is None:
            watched_dirs = []
        if directory not in watched_dirs:
            watched_dirs.append(directory)
            self.rag_config.set("rag_watched_directories", watched_dirs)


class CoreAIServiceManager:

    def __init__(self, config_manager: ConfigurationManager, logger: ArataxAILogger):
        self.config_manager = config_manager
        self.logger = logger
        self.service: Optional[Any] = None
        self.status = ServiceStatus.NOT_INITIALIZED
    
    def get_service(self) -> Any:
        if self.status == ServiceStatus.NOT_INITIALIZED:
            self.initialize()
        elif self.status == ServiceStatus.FAILED:
            raise ServiceInitializationError("Core AI service initialization previously failed")
        
        return self.service
    
    def initialize(self) -> None:
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
            raise ServiceInitializationError(f"Core AI service initialization failed: {e}")
    
    def is_configured(self) -> bool:
        try:
            self._validate_model_paths()
            return True
        except ValidationError:
            return False
    
    def get_configuration_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "llama_configured": False,
            "whisper_configured": False,
            "llama_model_path": None,
            "whisper_model_path": None,
            "llama_path_exists": False,
            "whisper_path_exists": False,
            "initialization_status": self.status.value
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
        llama_params = self.config_manager.llama_config.get_llama_cpp_params()
        whisper_params = self.config_manager.whisper_config.get_whisper_params()

        if not llama_params.model_path:
            raise ValidationError("Llama model path not configured")

        if not Path(llama_params.model_path).exists():
            raise ValidationError(f"Llama model path does not exist: {llama_params.model_path}")

        if not whisper_params.model:
            raise ValidationError("Whisper model path not configured")
        
        if not Path(whisper_params.model).exists():
            raise ValidationError(f"Whisper model path does not exist: {whisper_params.model}")
    
    def _initialize_services(self) -> None:
        llama_params = self.config_manager.llama_config.get_llama_cpp_params()
        whisper_params = self.config_manager.whisper_config.get_whisper_params()
        
        (
            llama_model_params_cc,
            llama_generation_params_cc,
            whisper_model_params_cc,
            whisper_transcription_params_cc,
        ) = self._convert_params(llama_params, whisper_params)

        self.service = self._create_core_ai_service(
            llama_model_params_cc, 
            whisper_model_params_cc
        )

    def _convert_params(self, llama_params: LlamaModelParams, whisper_params: WhisperModelParams) -> Tuple[Any, Any, Any, Any]:
        llama_model_params_cc : Any = core_ai_py.LlamaModelParams.from_dict(  # type: ignore
            llama_params.model_dump()
        )

        llama_generation_params_cc : Any = core_ai_py.GenerationParams.from_dict( # type: ignore
            self.config_manager.llama_config.get_generation_params().model_dump()
        )

        whisper_model_params_cc : Any = core_ai_py.WhisperModelParams.from_dict( # type: ignore
            whisper_params.model_dump()
        )

        whisper_transcription_params_cc : Any = core_ai_py.WhisperGenerationParams.from_dict( # type: ignore
            self.config_manager.whisper_config.get_transcription_params().model_dump()
        )
        
        return (
            llama_model_params_cc,
            llama_generation_params_cc,
            whisper_model_params_cc,
            whisper_transcription_params_cc,
        ) # type: ignore
    
    def _create_core_ai_service(self, llama_params: Any, whisper_params: Any) -> Any:
        service = core_ai_py.CoreAIService() # type: ignore
        service.initialize_llama_model(llama_params) # type: ignore
        service.initialize_whisper_model(whisper_params) # type: ignore
        return service # type: ignore
    
    def shutdown(self) -> None:
        """Shutdown core AI services"""
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
        """Check if services are initialized"""
        return self.status == ServiceStatus.INITIALIZED



class ChatManager:

    def __init__(self, db_manager: ChatDatabaseManager, logger: ArataxAILogger):
        self.db_manager = db_manager
        self.logger = logger
        self.validator = InputValidator()

    def create_project(self, name: str, description: str) -> ProjectResponse:
        self.validator.validate_string(name, "Project name")
        try:
            project = self.db_manager.create_project(name=name, description=description)
            self.logger.info(f"Created project: {name}")
            return ProjectResponse.model_validate(project)
        except Exception as e:
            self.logger.error(f"Failed to create project {name}: {e}")
            raise

    def get_project(self, project_id: uuid.UUID) -> ProjectResponse:
        self.validator.validate_uuid(project_id, "Project ID")
        try:
            project = self.db_manager.get_project(project_id)
            return ProjectResponse.model_validate(project)
        except Exception as e:
            self.logger.error(f"Failed to get project {project_id}: {e}")
            raise

    def list_projects(self) -> List[ProjectResponse]:
        try:
            projects = self.db_manager.list_projects()
            return [ProjectResponse.model_validate(project) for project in projects]
        except Exception as e:
            self.logger.error(f"Failed to list projects: {e}")
            raise

    def delete_project(self, project_id: uuid.UUID) -> bool:
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
        self.validator.validate_uuid(project_id, "Project ID")
        try:
            sessions = self.db_manager.get_sessions_for_project(project_id)
            return [ChatSessionResponse.model_validate(session) for session in sessions]
        except Exception as e:
            self.logger.error(f"Failed to list sessions for project {project_id}: {e}")
            raise

    def delete_session(self, session_id: uuid.UUID) -> bool:
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
        self.validator.validate_uuid(session_id, "Session ID")
        try:
            messages = self.db_manager.get_messages_for_session(session_id)
            return [MessageResponse.model_validate(message) for message in messages]
        except Exception as e:
            self.logger.error(f"Failed to get messages for session {session_id}: {e}")
            raise

    def delete_message(self, message_id: uuid.UUID) -> bool:
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
        self.directories = directories
        self.config = config
        self.logger = logger
        self.version = __version__
        self._marker_file = self.directories.config / self.config.get_setup_marker_filename(
            self.version
        )

    def is_first_launch(self) -> bool:
        return not self._marker_file.exists()

    def perform_first_launch_setup(self) -> None:
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
        self._marker_file.touch(exist_ok=False)


class AtaraxAIOrchestrator:

    def __init__(self, app_config: Optional[AppConfig] = None):
        self.app_config = app_config or AppConfig()
        self.logger = self._init_logger()
        self.directories = AppDirectories.create_default()

        self.setup_manager = SetupManager(
            self.directories, self.app_config, self.logger
        )
        self.config_manager = ConfigurationManager(self.directories.config, self.logger)
        self.core_ai_manager = CoreAIServiceManager(self.config_manager, self.logger)

        self._initialize_application()

    def _init_logger(self) -> ArataxAILogger:
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
            if config_status["llama_configured"] and config_status["whisper_configured"]:
                self.logger.info("AI services are properly configured and ready for use")
            else:
                self.logger.warning("AI services are not fully configured - some features may be unavailable")
            
            self._finalize_setup()
            
            self.logger.info("AtaraxAI initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AtaraxAI: {e}")
            raise

    def _init_database(self) -> None:
        db_path = self.directories.data / self.app_config.database_filename
        self.db_manager = ChatDatabaseManager(db_path=db_path)
        self.chat_context = ChatContextManager(db_manager=self.db_manager)
        self.chat_manager = ChatManager(self.db_manager, self.logger)
        self.logger.info("Database initialized successfully")

    def _init_rag_manager(self) -> None:
        self.rag_manager = AtaraxAIRAGManager(
            rag_config_manager=self.config_manager.rag_config,
            app_data_root_path=self.directories.data,
            core_ai_service=None,
        )
        self.logger.info("RAG manager initialized successfully")

    def _init_prompt_engine(self) -> None:
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
                chain_definition=chain_definition, 
                initial_user_query=initial_user_query
            )
            self.logger.info("Chain execution completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Chain execution failed: {e}")
            raise

    def add_watch_directory(self, directory: str) -> None:
        InputValidator.validate_directory(directory, "Directory path")

        self.config_manager.add_watched_directory(directory)
        watched_dirs = self.config_manager.get_watched_directories()
        self.rag_manager.start_file_monitoring(watched_dirs)
        self.logger.info(f"Added watch directory: {directory}")

    def create_project(self, name: str, description: str) -> ProjectResponse:
        return self.chat_manager.create_project(name, description)

    def get_project(self, project_id: uuid.UUID) -> ProjectResponse:
        return self.chat_manager.get_project(project_id)

    def get_project_summary(self, project_id: uuid.UUID) -> Dict[str, Any]:
        InputValidator.validate_uuid(project_id, "Project ID")
        try:
            return self.db_manager.get_project_summary(project_id)
        except Exception as e:
            self.logger.error(f"Failed to get project summary {project_id}: {e}")
            raise

    def list_projects(self) -> List[ProjectResponse]:
        return self.chat_manager.list_projects()

    def delete_project(self, project_id: uuid.UUID) -> bool:
        return self.chat_manager.delete_project(project_id)

    def create_session(self, project_id: uuid.UUID, title: str) -> ChatSessionResponse:
        return self.chat_manager.create_session(project_id, title)

    def list_sessions(self, project_id: uuid.UUID) -> List[ChatSessionResponse]:
        return self.chat_manager.list_sessions(project_id)

    def delete_session(self, session_id: uuid.UUID) -> bool:
        return self.chat_manager.delete_session(session_id)

    def add_message(
        self, session_id: uuid.UUID, role: str, content: str
    ) -> MessageResponse:
        return self.chat_manager.add_message(session_id, role, content)

    def get_messages_for_session(self, session_id: uuid.UUID) -> List[MessageResponse]:
        return self.chat_manager.get_messages_for_session(session_id)

    def delete_message(self, message_id: uuid.UUID) -> bool:
        return self.chat_manager.delete_message(message_id)

    def shutdown(self) -> None:
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
