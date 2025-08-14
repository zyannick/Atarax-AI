import asyncio
from pathlib import Path
from typing import Any, Dict, List
import logging
from typing import Optional
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager
from ataraxai.praxis.utils.vault_manager import VaultManager
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
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
)
from ataraxai.praxis.modules.models_manager.models_manager import ModelsManager
from ataraxai.praxis.utils.chat_manager import ChatManager
from ataraxai.praxis.utils.app_config import AppConfig
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager


class Services:

    def __init__(
        self,
        directories: AppDirectories,
        logger: logging.Logger,
        db_manager: ChatDatabaseManager,
        chat_context: ChatContextManager,
        chat_manager: ChatManager,
        config_manager: ConfigurationManager,
        app_config: AppConfig,
        vault_manager: VaultManager,
        models_manager: ModelsManager,
        core_ai_service_manager: CoreAIServiceManager,
    ):
        """
        Initializes the service with required managers, configuration, and logging utilities.

        Args:
            directories (AppDirectories): Handles application directory paths.
            logger (logging.Logger): Logger instance for logging events and errors.
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
        self.models_manager = models_manager
        self.core_ai_service_manager = core_ai_service_manager
        self.vault_manager = vault_manager

    async def initialize(self) -> None:
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
            await self._finalize_setup()
            self.logger.info("Services initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise

    def set_core_ai_manager(self, core_ai_manager: CoreAIServiceManager) -> None:
        """
        Sets the core AI manager instance for the current object.

        Args:
            core_ai_manager (CoreAIServiceManager): The core AI manager to be assigned.

        Returns:
            None

        Logs:
            Info message indicating that the core AI manager has been set for the chat manager and chain runner.
        """
        self.core_ai_manager = core_ai_manager
        self.logger.info("Core AI manager set for chat manager and chain runner")

    async def add_watched_directory(self, directory: str) -> None:
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
        await self.rag_manager.start()
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
        self.chat_context = ChatContextManager(
            db_manager=self.db_manager, vault_manager=self.vault_manager
        )
        self.chat_manager = ChatManager(
            self.db_manager, self.logger, self.vault_manager
        )
        self.logger.info("Database initialized successfully")

    def _init_rag_manager(self) -> None:
        """
        Initializes the RAG (Retrieval-Augmented Generation) manager for the application.

        This method creates an instance of `AtaraxAIRAGManager` using the current configuration and data directory,
        and assigns it to `self.rag_manager`. It also logs a message indicating successful initialization.
        """
        self.rag_manager = AtaraxAIRAGManager(
            rag_config_manager=self.config_manager.rag_config_manager,
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
            config=self.config_manager.rag_config_manager.get_config().model_dump(),
            rag_manager=self.rag_manager,
        )
        self.task_manager = TaskManager()
        self.chain_runner = ChainRunner(
            task_manager=self.task_manager,
            context_manager=self.context_manager,
            prompt_manager=self.prompt_manager,
            core_ai_service_manager=self.core_ai_service_manager,
            chat_context=self.chat_context,
            rag_manager=self.rag_manager,
        )
        self.logger.info("Prompt engine initialized successfully")

    async def run_task_chain(
        self, chain_definition: List[Dict[str, Any]], initial_user_query: str
    ) -> Any:
        InputValidator.validate_string(initial_user_query, "Initial user query")

        if not chain_definition:
            raise ValidationError("Chain definition cannot be empty")


        self.logger.info(f"Executing chain for query: '{initial_user_query}'")
        try:
            result = await self.chain_runner.run_chain(
                chain_definition=chain_definition, initial_user_query=initial_user_query
            )
            self.logger.info("Chain execution completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Chain execution failed: {e}")
            raise

    async def shutdown(self) -> None:
        self.logger.info("Shutting down AtaraxAI...")
        try:
            if hasattr(self, "rag_manager"):
                await self.rag_manager.stop()
            if hasattr(self, "db_manager"):
                await asyncio.to_thread(self.db_manager.close)
            if hasattr(self, "core_ai_manager"):
                await asyncio.to_thread(self.core_ai_service_manager.shutdown)
            self.logger.info("AtaraxAI shutdown completed successfully")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    async def _finalize_setup(self) -> None:
        await self.rag_manager.start()
