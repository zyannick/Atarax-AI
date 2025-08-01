import yaml
from pathlib import Path
from .config_schemas.whisper_config_schema import (
    WhisperConfig,
    WhisperModelParams,
    WhisperTranscriptionParams,
)
from typing_extensions import Optional

WHISPER_CONFIG_FILENAME = "whisper_config.yaml"


class WhisperConfigManager:

    def __init__(self, config_path: Path):
        """
        Initializes the WhisperConfigManager with the specified configuration path.

        Args:
            config_path (Path): The base directory where the configuration file will be stored.

        Side Effects:
            - Ensures that the parent directory for the configuration file exists, creating it if necessary.
            - Loads an existing Whisper configuration or initializes a new one.

        Attributes:
            config_path (Path): Full path to the configuration file.
            config (WhisperConfig): The loaded or newly initialized Whisper configuration.
        """
        self.config_path = config_path / WHISPER_CONFIG_FILENAME
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config: WhisperConfig = self._load_or_initialize()

    def _load_or_initialize(self):
        """
        Loads the Whisper configuration from the specified config path if it exists; 
        otherwise, initializes and saves a default configuration.

        Returns:
            WhisperConfig: The loaded or newly created Whisper configuration object.

        Side Effects:
            - Prints error messages if loading fails.
            - Prints info messages when creating a default config.
            - Saves a default configuration file if none exists or loading fails.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return WhisperConfig(**yaml.safe_load(f))
            except Exception as e:
                print(f"Failed to load YAML config: {e}")
        default = self._default_config()
        self._save(default)
        return default

    def _save(self, config: Optional[WhisperConfig] = None):
        """
        Saves the Whisper configuration to a YAML file.

        If a `config` object is provided, it will be saved; otherwise, the current instance's configuration is saved.
        Handles exceptions and prints an error message if saving fails.

        Args:
            config (Optional[WhisperConfig]): The configuration object to save. If None, saves the current instance's config.
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config.model_dump() if config else self.config.model_dump(),
                    f,
                    default_flow_style=False,
                )
        except Exception as e:
            print(f"Failed to save config: {e}")

    def _default_config(self) -> WhisperConfig:
        """
        Creates and returns a default WhisperConfig instance with default model and transcription parameters.

        Returns:
            WhisperConfig: A configuration object initialized with default WhisperModelParams and WhisperTranscriptionParams.
        """
        return WhisperConfig(
            config_version="1.0",
            whisper_model_params=WhisperModelParams(),
            whisper_transcription_params=WhisperTranscriptionParams(),
        )

    def get_whisper_params(self) -> WhisperModelParams:
        """
        Retrieves the current Whisper model parameters from the configuration.

        Returns:
            WhisperModelParams: The parameters used for configuring the Whisper model.
        """
        return self.config.whisper_model_params

    def update_whisper_model_params(self, params: WhisperModelParams):
        """
        Update the Whisper model parameters in the configuration and save the changes.

        Args:
            params (WhisperModelParams): The new parameters to set for the Whisper model.
        """
        self.config.whisper_model_params = params
        self._save()

    def get_transcription_params(self) -> WhisperTranscriptionParams:
        """
        Retrieve the current Whisper transcription parameters from the configuration.

        Returns:
            WhisperTranscriptionParams: The transcription parameters used by Whisper.
        """
        return self.config.whisper_transcription_params

    def update_whisper_transcription_params(self, params: WhisperTranscriptionParams):
        """
        Update the Whisper transcription parameters in the configuration.

        Args:
            params (WhisperTranscriptionParams): The new transcription parameters to set.

        Side Effects:
            Updates the internal configuration and persists the changes by saving to storage.
        """
        self.config.whisper_transcription_params = params
        self._save()

    def update_transcription_params(self, params: WhisperTranscriptionParams):
        """
        Update the Whisper transcription parameters in the configuration.

        Args:
            params (WhisperTranscriptionParams): The new transcription parameters to be set.

        Side Effects:
            Updates the `whisper_transcription_params` attribute in the configuration and saves the updated configuration to persistent storage.
        """
        self.config.whisper_transcription_params = params
        self._save()

    def get_config(self) -> WhisperConfig:
        """
        Retrieve the current WhisperConfig instance.

        Returns:
            WhisperConfig: The current configuration object for Whisper.
        """
        return self.config

    def set_param(self, section: str, key: str, value: str):
        """
        Sets a configuration parameter for the specified section and key.

        Args:
            section (str): The section of the configuration to update. 
                Must be either "whisper_model_params" or "whisper_transcription_params".
            key (str): The parameter key to set within the section.
            value (str): The value to assign to the parameter.

        Raises:
            KeyError: If the section or key does not exist in the configuration.
        """
        if (
            section == "whisper_model_params"
            and key in self.config.whisper_model_params.__dict__
        ):
            setattr(self.config.whisper_model_params, key, value)
            self._save()
        elif (
            section == "whisper_transcription_params"
            and key in self.config.whisper_transcription_params.__dict__
        ):
            setattr(self.config.whisper_transcription_params, key, value)
            self._save()
        else:
            raise KeyError(f"Section '{section}' or key '{key}' not found in config.")

    def reload(self):
        """
        Reloads the configuration by reloading or initializing it.

        This method updates the `config` attribute by calling the internal
        method `_load_or_initialize`, ensuring the latest configuration
        is loaded into the instance.
        """
        self.config = self._load_or_initialize()
