import yaml
from pathlib import Path
from ataraxai.app_logic.utils.config_schemas.llama_config_schema import (
    LlamaConfig,
    LlamaModelParams,
    GenerationParams,
)
from typing_extensions import Optional


LLAMA_CONFIG_FILENAME = "llama_config.yaml"


class LlamaConfigManager:
    def __init__(self, config_path: Path):
        """
        Initializes the configuration manager with the given configuration directory path.

        Args:
            config_path (Path): The path to the configuration directory. Must be a directory.

        Raises:
            ValueError: If the provided config_path is not a directory.

        Side Effects:
            - Sets self.config_path to the configuration file path within the directory.
            - Creates the parent directories for the configuration file if they do not exist.
            - Loads or initializes the LlamaConfig instance and assigns it to self.config.
        """
        if config_path.is_dir():
            self.config_path = config_path / LLAMA_CONFIG_FILENAME
        else:
            raise ValueError(f"Invalid config path: {config_path}")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config: LlamaConfig = self._load_or_initialize()

    def _load_or_initialize(self) -> LlamaConfig:
        """
        Loads the Llama configuration from the specified file path if it exists and is valid.
        If the configuration file does not exist or cannot be parsed, creates a default LlamaConfig,
        saves it to the file path, and returns it.

        Returns:
            LlamaConfig: The loaded or newly created Llama configuration.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f)
                return LlamaConfig(**raw_data)
            except Exception as e:
                print(f"[ERROR] Failed to parse config: {e}")
        print(f"[INFO] Creating default LLM config at: {self.config_path}")
        config = LlamaConfig()
        self._save(config)
        return config

    def _save(self, config: Optional[LlamaConfig] = None):
        """
        Saves the current LlamaConfig instance to a YAML file.

        If a config is provided, it serializes and saves that config; otherwise, it saves the instance's current config.
        The configuration is written to the file specified by `self.config_path` in UTF-8 encoding using YAML format.

        Args:
            config (Optional[LlamaConfig]): An optional LlamaConfig instance to save. If None, saves `self.config`.

        Raises:
            OSError: If the file cannot be opened or written to.
        """
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config.model_dump() if config else self.config.model_dump(),
                f,
                default_flow_style=False,
            )

    def get_llama_cpp_params(self) -> LlamaModelParams:
        """
        Retrieve the Llama C++ model parameters from the current configuration.

        Returns:
            LlamaModelParams: The parameters used to configure the Llama C++ model.
        """
        return self.config.llama_cpp_model_params

    def set_llama_cpp_params(self, params: LlamaModelParams):
        """
        Set the parameters for the llama.cpp model.

        Args:
            params (LlamaModelParams): The parameters to configure the llama.cpp model.

        Side Effects:
            Updates the internal configuration with the provided parameters and saves the configuration.
        """
        self.config.llama_cpp_model_params = params
        self._save()

    def get_generation_params(self) -> GenerationParams:
        """
        Retrieve the current generation parameters from the configuration.

        Returns:
            GenerationParams: The generation parameters stored in the configuration.
        """
        return self.config.generation_params

    def set_generation_params(self, params: GenerationParams):
        """
        Set the generation parameters for the configuration.

        Args:
            params (GenerationParams): The generation parameters to be set.

        Side Effects:
            Updates the `generation_params` attribute of the configuration and saves the updated configuration.
        """
        self.config.generation_params = params
        self._save()

    def set_param(self, section: str, key: str, value: str):
        """
        Sets the value of a configuration parameter within a specified section.

        Args:
            section (str): The section of the configuration to update. 
                Must be either "llama_cpp_model_params" or "generation_params".
            key (str): The name of the parameter to set within the section.
            value (str): The new value to assign to the parameter.

        Raises:
            ValueError: If the section or key is invalid.

        Side Effects:
            Updates the configuration and saves the changes to persistent storage.
        """
        if section == "llama_cpp_model_params" and hasattr(self.config.llama_cpp_model_params, key):
            setattr(self.config.llama_cpp_model_params, key, value)
        elif section == "generation_params" and hasattr(
            self.config.generation_params, key
        ):
            setattr(self.config.generation_params, key, value)
        else:
            raise ValueError(f"Invalid section '{section}' or key '{key}'")
        self._save()

    def reload(self):
        """
        Reloads the configuration by re-initializing it from the source.

        This method updates the `config` attribute by loading the latest configuration,
        either from a persistent storage or by initializing a new configuration if none exists.
        """
        self.config = self._load_or_initialize()
