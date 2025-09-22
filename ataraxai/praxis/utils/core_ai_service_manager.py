import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ataraxai import hegemonikon_py  # type: ignore

# from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
)
from ataraxai.praxis.utils.configs.config_schemas.whisper_config_schema import (
    WhisperModelParams,
)
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.exceptions import ServiceInitializationError, ValidationError
from ataraxai.praxis.utils.service_status import ServiceStatus


class CoreAIServiceManager:

    def __init__(self, config_manager: ConfigurationManager, logger: logging.Logger):
        """
        Initializes the core AI service manager with the provided configuration manager and logger.

        Args:
            config_manager (ConfigurationManager): The configuration manager instance to manage service configurations.
            logger (ArataxAILogger): The logger instance for logging service events and errors.

        Attributes:
            service (Optional[Any]): The AI service instance, initialized as None.
            status (ServiceStatus): The current status of the service, initialized as NOT_INITIALIZED.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.core_ai_service: Optional[Any] = None
        self.llama_cpp_status = ServiceStatus.NOT_INITIALIZED
        self.llama_cpp_params_cc = None
        self.llama_cpp_generation_params_cc = None

    def get_service(self) -> Any:
        """
        Retrieves the core AI service instance, initializing it if necessary.

        If the service has not been initialized, this method will attempt to initialize it.
        If the previous initialization attempt failed, raises a ServiceInitializationError.
        Otherwise, returns the initialized service instance.

        Returns:
            Any: The initialized core AI service instance.

        Raises:
            ServiceInitializationError: If the service failed to initialize previously.
        """
        # in the current implementation, we only have one service
        if self.llama_cpp_status == ServiceStatus.NOT_INITIALIZED:
            self.initialize()
        elif self.llama_cpp_status == ServiceStatus.FAILED:
            raise ServiceInitializationError(
                "Core AI service initialization previously failed"
            )

        return self.core_ai_service

    async def process_prompt(self, prompt: str) -> str:
        """
        Processes a prompt using the core AI service.

        Args:
            prompt (str): The prompt to process.
            generation_params (Any): The parameters for generation.

        Returns:
            str: The processed response from the core AI service.
        """
        if not self.core_ai_service:
            raise ServiceInitializationError("Core AI service is not initialized")

        response = await asyncio.to_thread(
            self.core_ai_service.process_prompt,
            prompt.encode("utf-8"),
            self.llama_cpp_generation_params_cc,
        )

        return response

    def get_llama_cpp_model_context_size(self) -> int:
        """
        Retrieves the context size of the Llama model from the core AI service.

        Returns:
            int: The context size of the Llama model.

        Raises:
            ServiceInitializationError: If the core AI service is not initialized.
        """
        if not self.core_ai_service:
            raise ServiceInitializationError("Core AI service is not initialized")

        return self.config_manager.llama_config_manager.get_llama_cpp_params().n_ctx

    def initialize(self) -> None:
        """
        Initializes the core AI services if they have not already been initialized.

        This method performs the following steps:
            1. Checks if the services are already initialized and returns early if so.
            2. Sets the status to INITIALIZING and logs the start of the initialization process.
            3. Validates model paths and initializes required services.
            4. Updates the status to INITIALIZED and logs success upon completion.
            5. If any exception occurs during initialization, sets the status to FAILED,
               logs the error, and raises a ServiceInitializationError.

        Raises:
            ServiceInitializationError: If initialization of core AI services fails.
        """
        if self.llama_cpp_status == ServiceStatus.INITIALIZED:
            self.logger.info("Core AI services already initialized")
            return

        self.llama_cpp_status = ServiceStatus.INITIALIZING
        self.logger.info("Initializing Core AI services...")

        try:
            self._validate_model_paths()
            self._initialize_services()
        except Exception as e:
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
            "initialization_status": self.llama_cpp_status.value,
        }

        try:
            llama_params = (
                self.config_manager.llama_config_manager.get_llama_cpp_params()
            )
            status["llama_model_path"] = llama_params.model_path
            status["llama_configured"] = bool(llama_params.model_path)
            if len(str(llama_params.model_path)) > 0:
                status["llama_path_exists"] = Path(
                    str(llama_params.model_path)
                ).exists()
        except Exception as e:
            self.logger.warning(f"Could not get Llama configuration: {e}")

        try:
            whisper_params = (
                self.config_manager.whisper_config_manager.get_whisper_params()
            )
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
        self.logger.info("Validating model paths for Llama and Whisper models...")
        llama_params = self.config_manager.llama_config_manager.get_llama_cpp_params()
        _ = self.config_manager.whisper_config_manager.get_whisper_params()

        if llama_params.model_path is None:
            raise ValidationError("Llama model path not configured")

        if not Path(str(llama_params.model_path)).exists():
            raise ValidationError(
                f"Llama model path does not exist: {llama_params.model_path}"
            )

        # if not whisper_params.model:
        #     raise ValidationError("Whisper model path not configured")

        # if not Path(whisper_params.model).exists():
        #     raise ValidationError(
        #         f"Whisper model path does not exist: {whisper_params.model}"
        #     )

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
        llama_params = self.config_manager.llama_config_manager.get_llama_cpp_params()
        whisper_params = self.config_manager.whisper_config_manager.get_whisper_params()

        self.llama_cpp_params_cc, self.llama_cpp_generation_params_cc = (
            self._convert_params(llama_params, whisper_params)
        )

        self.core_ai_service = self._create_core_ai_service(
            self.llama_cpp_params_cc, None
        )

    def _convert_params(
        self, llama_params: LlamaModelParams, whisper_params: WhisperModelParams
    ) -> Tuple[Any, Any]:
        """
        Converts Llama and Whisper model parameter objects into their corresponding hegemonikon_py representations.

        Args:
            llama_params (LlamaModelParams): The Llama model parameters to convert.
            whisper_params (WhisperModelParams): The Whisper model parameters to convert.

        Returns:
            Tuple[Any, Any, Any, Any]: A tuple containing:
                - Converted Llama model parameters (hegemonikon_py.LlamaModelParams)
                - Converted Llama generation parameters (hegemonikon_py.GenerationParams)
                - Converted Whisper model parameters (hegemonikon_py.WhisperModelParams)
                - Converted Whisper transcription parameters (hegemonikon_py.WhisperGenerationParams)
        """
        llama_model_params_cc: Any = hegemonikon_py.LlamaModelParams.from_dict(  # type: ignore
            {
                "model_path": str(llama_params.model_info.local_path),  # type: ignore
                "n_ctx": llama_params.n_ctx,
                "n_gpu_layers": llama_params.n_gpu_layers,
                "main_gpu": llama_params.main_gpu,
                "tensor_split": llama_params.tensor_split,
                "vocab_only": llama_params.vocab_only,
                "use_map": llama_params.use_map,
                "use_mlock": llama_params.use_mlock,
            }
        )

        llama_generation_params_cc: Any = hegemonikon_py.GenerationParams.from_dict(  # type: ignore
            self.config_manager.llama_config_manager.get_generation_params().model_dump()
        )

        # whisper_model_params_cc: Any = hegemonikon_py.WhisperModelParams.from_dict(  # type: ignore
        #     whisper_params.model_dump()
        # )

        # whisper_transcription_params_cc: Any = hegemonikon_py.WhisperGenerationParams.from_dict(  # type: ignore
        #     self.config_manager.whisper_config_manager.get_transcription_params().model_dump()
        # )

        return (
            llama_model_params_cc,
            llama_generation_params_cc,
            # whisper_model_params_cc,
            # whisper_transcription_params_cc,
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
        service = hegemonikon_py.CoreAIService()  # type: ignore
        service.initialize_llama_model(llama_params)  # type: ignore
        # try:
        #     service.initialize_whisper_model(whisper_params)  # type: ignore
        # except Exception as e:
        #     self.logger.error(f"Error initializing Whisper model: {e}")
        return service  # type: ignore

    def tokenize(self, text: str) -> List[int]:
        if not self.core_ai_service:
            raise ServiceInitializationError("Core AI service is not initialized")

        return self.core_ai_service.tokenization(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens into a string using the core AI service.

        Args:
            tokens (List[int]): The list of tokens to decode.

        Returns:
            str: The decoded string.
        """
        if not self.core_ai_service:
            raise ServiceInitializationError("Core AI service is not initialized")

        return str(self.core_ai_service.detokenization(tokens))

    def shutdown(self) -> None:
        """
        Shuts down the core AI services managed by this orchestrator.

        If a service is currently running, attempts to shut it down gracefully.
        Logs the outcome of the shutdown process, including any errors encountered.
        Resets the service reference and updates the service status to NOT_INITIALIZED.
        """
        if self.core_ai_service:
            try:
                # self.service.shutdown()
                self.logger.info("Core AI services shut down successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down core AI services: {e}")
            finally:
                self.core_ai_service = None
                self.status = ServiceStatus.NOT_INITIALIZED

    @property
    def is_initialized(self) -> bool:
        """
        Checks if the service has been initialized.

        Returns:
            bool: True if the service status is INITIALIZED, False otherwise.
        """
        return self.status == ServiceStatus.INITIALIZED
