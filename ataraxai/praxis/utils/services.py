from pathlib import Path
from typing import Any, Dict, List
from ataraxai.praxis.utils.vault_manager import VaultManager
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.utils.ataraxai_logger import ArataxAILogger
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
from ataraxai.praxis.modules.prompt_engine.context_manager import ContextManager
from ataraxai.praxis.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.praxis.modules.prompt_engine.task_manager import TaskManager
from ataraxai.praxis.modules.prompt_engine.chain_runner import ChainRunner
from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.input_validator import InputValidator
from ataraxai.praxis.utils.exceptions import (
    ValidationError,
    ServiceInitializationError,
)
from ataraxai.praxis.utils.chat_manager import ChatManager
from ataraxai.praxis.utils.app_config import AppConfig
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager


class Services:

    def __init__(
        self,
        directories: AppDirectories,
        logger: ArataxAILogger,
        db_manager: ChatDatabaseManager,
        chat_context: ChatContextManager,
        chat_manager: ChatManager,
        config_manager: ConfigurationManager,
        app_config: AppConfig,
        vault_manager: VaultManager,
    ):
        """
        Initializes the service with required managers, configuration, and logging utilities.

        Args:
            directories (AppDirectories): Handles application directory paths.
            logger (ArataxAILogger): Logger instance for logging events and errors.
            db_manager (ChatDatabaseManager): Manages chat-related database operations.
            chat_context (ChatContextManager): Maintains chat context and state.
            chat_manager (ChatManager): Handles chat session management.
            config_manager (ConfigurationManager): Manages application configuration.
            app_config (AppConfig): Application configuration settings.
            vault_manager (VaultManager): Manages secure storage and retrieval of secrets.
        """
        self.directories = directories
        self.logger = logger
        self.db_manager = db_manager
        self.chat_context = chat_context
        self.chat_manager = chat_manager
        self.config_manager = config_manager
        self.app_config = app_config
        self.core_ai_service = None
        self.vault_manager = vault_manager

    def initialize(self) -> None:
        """
        Initializes the core services required for the application.

        This method sequentially initializes the database, RAG manager, and prompt engine,
        then finalizes the setup. Logs a success message upon completion. If any step fails,
        an error is logged and the exception is propagated.

        Raises:
            Exception: If any of the service initialization steps fail.
        """
        try:
            self._init_database()
            self._init_rag_manager()
            self._init_prompt_engine()
            self._finalize_setup()
            self.logger.info("Services initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise

    def set_core_ai_manager(self, core_ai_manager: Any) -> None:
        """
        Sets the core AI manager instance for the current object.

        Args:
            core_ai_manager (Any): The core AI manager to be assigned.

        Returns:
            None

        Logs:
            Info message indicating that the core AI manager has been set for the chat manager and chain runner.
        """
        self.core_ai_manager = core_ai_manager
        self.logger.info("Core AI manager set for chat manager and chain runner")

    def add_watched_directory(self, directory: str) -> None:
        """
        Adds a new directory to the list of watched directories and starts monitoring it.

        Validates the provided directory path, updates the configuration to include the new directory,
        restarts file monitoring for all watched directories, and logs the addition.

        Args:
            directory (str): The path of the directory to be watched.

        Raises:
            ValueError: If the provided directory path is invalid.
        """
        InputValidator.validate_directory(directory, "Directory path")

        self.config_manager.add_watched_directory(directory)
        watched_dirs = self.config_manager.get_watched_directories()
        self.rag_manager.start_file_monitoring(watched_dirs)
        self.logger.info(f"Added watch directory: {directory}")

    def _init_database(self) -> None:
        """
        Initializes the application's database and related managers.

        This method sets up the database path, creates an instance of the ChatDatabaseManager
        with the appropriate security manager, and initializes the chat context and chat manager.
        It also logs a message upon successful initialization.
        """
        db_path = self.directories.data / self.app_config.database_filename
        self.db_manager = ChatDatabaseManager(db_path=db_path)
        self.chat_context = ChatContextManager(db_manager=self.db_manager)
        self.chat_manager = ChatManager(self.db_manager, self.logger, self.vault_manager)
        self.logger.info("Database initialized successfully")

    def _init_rag_manager(self) -> None:
        """
        Initializes the RAG (Retrieval-Augmented Generation) manager for the application.

        This method creates an instance of `AtaraxAIRAGManager` using the current configuration and data directory,
        and assigns it to `self.rag_manager`. It also logs a message indicating successful initialization.
        """
        self.rag_manager = AtaraxAIRAGManager(
            rag_config_manager=self.config_manager.rag_config,
            app_data_root_path=self.directories.data,
            core_ai_service=None,
        )
        self.logger.info("RAG manager initialized successfully")

    def _init_prompt_engine(self) -> None:
        """
        Initializes the prompt engine and its related managers.

        This method sets up the prompt, context, and task managers, as well as the chain runner,
        using the application's configuration. It ensures the prompts directory exists, initializes
        the PromptManager, ContextManager, and TaskManager, and then creates a ChainRunner with
        the necessary dependencies. Logs a message upon successful initialization.
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

    def run_task_chain(
        self, chain_definition: List[Dict[str, Any]], initial_user_query: str
    ) -> Any:
        """
        Executes a sequence of tasks defined in a chain for a given user query.

        Args:
            chain_definition (List[Dict[str, Any]]): A list of dictionaries, each representing a task in the chain.
            initial_user_query (str): The initial query provided by the user to start the task chain.

        Returns:
            Any: The result of executing the task chain.

        Raises:
            ValidationError: If the chain definition is empty or the initial user query is invalid.
            ServiceInitializationError: If the core AI service cannot be initialized.
            Exception: If any error occurs during the execution of the task chain.

        Logs:
            - Info: When chain execution starts and completes successfully.
            - Error: If chain execution fails or the core AI service cannot be initialized.
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

        self.logger.info(f"Executing chain for query: '{initial_user_query}'")
        try:
            # _ = chain_definition[0].get("task_id", "unknown")
            result = self.chain_runner.run_chain(
                chain_definition=chain_definition, initial_user_query=initial_user_query
            )
            self.logger.info("Chain execution completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Chain execution failed: {e}")
            raise

    def shutdown(self) -> None:
        """
        Shuts down the AtaraxAI service by stopping file monitoring, closing the database connection,
        and shutting down the core AI manager. Logs the shutdown process and handles any exceptions
        that occur during shutdown.

        Raises:
            Exception: If an error occurs during the shutdown process, it is logged and re-raised.
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

    def _finalize_setup(self) -> None:
        """
        Finalizes the setup process by validating the RAG (Retrieval-Augmented Generation) index and initializing file monitoring.

        This method performs the following steps:
        1. Retrieves the list of directories to watch from the configuration manager.
        2. Checks if the RAG manifest is valid for the current RAG store.
        3. If the manifest is invalid, rebuilds the RAG index for the watched directories.
        4. If the manifest is valid, performs an initial scan of the watched directories.
        5. Starts monitoring the watched directories for file changes.
        """
        watched_dirs = self.config_manager.get_watched_directories()
        is_valid = self.rag_manager.manifest.is_valid(self.rag_manager.rag_store)
        if not is_valid:
            self.rag_manager.rebuild_index(watched_dirs)
        else:
            self.rag_manager.perform_initial_scan(watched_dirs)
        self.rag_manager.start_file_monitoring(watched_dirs)
