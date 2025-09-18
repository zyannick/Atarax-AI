from typing import Union, Any, Dict
import yaml
from pathlib import Path
from .config_schemas.rag_config_schema import (
    RAGConfig,
)
from typing_extensions import Optional


RAG_CONFIG_FILENAME = "rag_config.yaml"


class RAGConfigManager:
    """
    Manages the configuration for the RAG (Retrieval-Augmented Generation) system.
    Loads, saves, and updates the RAG configuration settings.
    """

    def __init__(self, config_path: Path):
        """
        Initializes the RAGConfigManager with the specified configuration path.

        Args:
            config_path (Path): The directory path where the RAG configuration file will be stored.

        Side Effects:
            - Ensures that the parent directory for the configuration file exists, creating it if necessary.
            - Loads an existing RAG configuration or initializes a new one.

        Attributes:
            config_path (Path): Full path to the RAG configuration file.
            config (RAGConfig): The loaded or newly initialized RAG configuration.
        """
        self.config_path = config_path / RAG_CONFIG_FILENAME
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config: RAGConfig = self._load_or_initialize()

    def _load_or_initialize(self) -> RAGConfig:
        """
        Loads the RAG configuration from the specified file path if it exists; otherwise,
        initializes a default configuration and saves it to the file.

        Returns:
            RAGConfig: The loaded or newly created RAG configuration object.

        Side Effects:
            - Prints error messages if loading fails.
            - Prints info messages when creating a default config.
            - Saves a default configuration file if none exists or loading fails.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return RAGConfig(**yaml.safe_load(f))
            except Exception as e:
                print(f"Failed to load YAML config: {e}")
        print(f"Creating default RAG config at: {self.config_path}")
        default = RAGConfig()
        self._save(default)
        return default

    def _save(self, config: Optional[RAGConfig] = None):
        """
        Saves the current RAG configuration to a YAML file.

        If a `config` object is provided, it will be serialized and saved.
        Otherwise, the instance's current configuration (`self.config`) will be saved.

        Args:
            config (Optional[RAGConfig]): The configuration object to save. If None, saves the instance's current config.

        Raises:
            OSError: If the file cannot be written.
        """
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config.model_dump() if config else self.config.model_dump(),
                f,
                default_flow_style=False,
            )

    def __eq__(self, value: object) -> bool:
        """
        Determine if this RAGConfigManager instance is equal to another.

        Args:
            value (object): The object to compare with this instance.

        Returns:
            bool: True if the other object is a RAGConfigManager and has the same config, False otherwise.
        """
        if not isinstance(value, RAGConfigManager):
            return NotImplemented
        return self.config == value.config

    def set(self, key: str, value: Optional[object]):
        """
        Sets the value of a configuration attribute and saves the updated configuration.

        Args:
            key (str): The name of the configuration attribute to set.
            value (Optional[object]): The value to assign to the configuration attribute.

        """
        setattr(self.config, key, value)
        self._save()

    def get(
        self, key: str, default: Optional[Union[str, int, bool, Dict[str, Any]]] = None
    ) -> Optional[Union[str, int, bool, Dict[str, Any]]]:
        """
        Retrieve the value associated with the given key from the configuration.

        Args:
            key (str): The name of the configuration attribute to retrieve.
            default (Optional[Union[str, int, bool, Dict[str, Any]]], optional): 
                The value to return if the key is not found in the configuration. Defaults to None.

        Returns:
            Optional[Union[str, int, bool, Dict[str, Any]]]: 
                The value of the configuration attribute if it exists, otherwise the default value.
        """
        return getattr(self.config, key, default)

    def get_config(self) -> RAGConfig:
        """
        Returns the current RAGConfig instance.

        Returns:
            RAGConfig: The current configuration object.
        """
        return self.config

    def update_config(self, new_config: RAGConfig):
        """
        Update the current RAG configuration with a new configuration.

        Args:
            new_config (RAGConfig): The new configuration to be set.

        Side Effects:
            - Updates the internal configuration state.
            - Persists the new configuration by calling the _save method.
        """
        self.config = new_config
        self._save(new_config)

    def reload(self) -> RAGConfig:
        """
        Reloads the configuration by reloading or initializing it.

        This method updates the `config` attribute by calling the internal
        `_load_or_initialize` method, ensuring the latest configuration is loaded.
        """
        self.config = self._load_or_initialize()
        return self.config
