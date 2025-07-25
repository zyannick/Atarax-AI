from pathlib import Path
from typing import List, Optional
from ataraxai.praxis.utils.exceptions import ServiceInitializationError
from ataraxai.praxis.utils.user_preferences_manager import UserPreferencesManager
from ataraxai.praxis.utils.configs.llama_config_manager import LlamaConfigManager
from ataraxai.praxis.utils.configs.whisper_config_manager import WhisperConfigManager
from ataraxai.praxis.utils.configs.rag_config_manager import RAGConfigManager
import logging

class ConfigurationManager:

    def __init__(self, config_dir: Path, logger: logging.Logger):
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
            self.preferences_manager = UserPreferencesManager(config_path=self.config_dir)
            self.llama_config_manager = LlamaConfigManager(config_path=self.config_dir)
            self.whisper_config_manager = WhisperConfigManager(config_path=self.config_dir)
            self.rag_config_manager = RAGConfigManager(config_path=self.config_dir)
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
        config = self.rag_config_manager.get_config()
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
            self.rag_config_manager.set("rag_watched_directories", watched_dirs)
