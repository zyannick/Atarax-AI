import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Type
import shutil
from ataraxai import __version__  # type: ignore
from ataraxai.hegemonikon_py import SecureString  # type: ignore

from ataraxai.praxis.modules.models_manager.model_manager import ModelManager
from ataraxai.praxis.utils.vault_manager import VaultManager
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
import threading
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.exceptions import (
    AtaraxAIError,
    # ValidationError,
)
from ataraxai.praxis.utils.ataraxai_settings import AtaraxAISettings
from ataraxai.praxis.utils.vault_manager import (
    VaultUnlockStatus,
)
from ataraxai.praxis.utils.setup_manager import SetupManager
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager
from ataraxai.praxis.utils.chat_manager import ChatManager
from ataraxai.praxis.utils.app_config import AppConfig
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.services import Services



class AtaraxAIOrchestrator:

    EXPECTED_RESET_CONFIRMATION_PHRASE = "reset ataraxai vault"

    def __init__(
        self,
        app_config: AppConfig,
        settings: AtaraxAISettings,
        logger: logging.Logger,
        directories: AppDirectories,
        vault_manager: VaultManager,
        setup_manager: SetupManager,
        config_manager: ConfigurationManager,
        core_ai_manager: CoreAIServiceManager,
        services: Services,
    ):
        
        self.app_config = app_config
        self.logger = logger
        self.settings = settings
        self.directories = directories

        self.vault_manager = vault_manager
        self.setup_manager = setup_manager
        self.config_manager = config_manager
        self.core_ai_manager = core_ai_manager
        self.services = services

        self._state_lock = threading.RLock()
        self._state: AppState = AppState.LOCKED
        self.initialize()

    def initialize(self):
        """
        Initializes the orchestrator by setting up base components if this is the first launch.
        Logs the current state after initialization.

        Acquires a lock to ensure thread-safe state checking and initialization.

        Raises:
            Any exceptions raised by the underlying initialization methods.
        """
        self._set_state(self._determine_initial_state())
        with self._state_lock:
            if self._state == AppState.FIRST_LAUNCH:
                self._initialize_base_components()

        self.logger.info(f"Orchestrator initialized. Current state: {self._state.name}")

    @property
    def state(self) -> AppState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: AppState) -> None:
        """
        Safely updates the application's state to the specified new_state.

        Acquires a lock to ensure thread-safe modification of the internal state.
        Only updates the state if the new state differs from the current state.

        Args:
            new_state (AppState): The new state to set for the application.

        Returns:
            None
        """
        with self._state_lock:
            if self._state != new_state:
                self._state = new_state

    def _init_logger(self) -> "AtaraxAILogger":
        return AtaraxAILogger()

    def _init_security_manager(self) -> VaultManager:
        """
        Initializes and returns a VaultManager instance using the specified salt and check files.

        Returns:
            VaultManager: An instance of VaultManager configured with the paths to the salt and check files
                          located in the data directory.
        """
        salt_file = self.directories.data / "vault.salt"
        check_file = self.directories.data / "vault.check"
        return VaultManager(salt_path=str(salt_file), check_path=str(check_file))

    def _determine_initial_state(self) -> AppState:
        """
        Determines the initial application state based on the existence of the vault check path.

        Returns:
            AppState: Returns AppState.LOCKED if the vault check path exists,
                      otherwise returns AppState.FIRST_LAUNCH.
        """
        if Path(self.vault_manager.check_path).exists():
            return AppState.LOCKED
        else:
            return AppState.FIRST_LAUNCH

    def _initialize_base_components(self):
        """
        Initializes the base components required for the application.

        Logs the start of the application with its version. If this is the first launch,
        it triggers the first launch setup process via the setup manager. Handles and logs
        any exceptions that occur during initialization, sets the application state to ERROR,
        and re-raises the exception.
        """
        try:
            self.logger.info(f"Starting AtaraxAI v{__version__}")
            if self.setup_manager.is_first_launch():
                self.setup_manager.perform_first_launch_setup()
        except Exception as e:
            self.logger.error(f"Failed during base initialization: {e}")
            self._set_state(AppState.ERROR)
            raise
        
    # def _clear_internal_refs(self):
    #     """
    #     Clears internal references to services and managers to help with garbage collection.

    #     This method is called when the orchestrator is shutting down to ensure that all internal
    #     references are cleared, allowing Python's garbage collector to reclaim memory.
    #     """
    #     self.services = None
    #     self.vault_manager = None
    #     self.setup_manager = None
    #     self.config_manager = None
    #     self.core_ai_manager = None
    
    def _reset_state(self):
        """
        Resets the internal state of the orchestrator to its initial state.

        This method is called when the orchestrator is shutting down to ensure that the state
        is reset, allowing for a clean restart of the application.
        """
        with self._state_lock:
            self._state = AppState.LOCKED
            self.logger.info("Orchestrator state reset to LOCKED.")

    def initialize_new_vault(self, master_password: SecureString) -> bool:
        """
        Initializes a new vault with the provided master password.

        This method should only be called during the application's first launch.
        If the application is not in the FIRST_LAUNCH state, initialization will not proceed.

        Args:
            master_password (SecureString): The master password to secure the new vault.

        Returns:
            bool: True if the vault was successfully initialized and unlocked, False otherwise.

        Side Effects:
            - Changes the application state to UNLOCKED on success, or ERROR on failure.
            - Logs relevant information and errors.
            - Initializes services required for the unlocked state.
        """

        if self._state != AppState.FIRST_LAUNCH:
            self.logger.error("Attempted to initialize an existing vault.")
            return False

        try:
            self.vault_manager.create_and_initialize_vault(master_password)
            self._set_state(AppState.UNLOCKED)
            self.logger.info("Vault successfully initialized and unlocked.")
            self._initialize_unlocked_services()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize new vault: {e}")
            self._set_state(AppState.ERROR)
            return False

    def reinitialize_vault(self, confirmation_phrase: str) -> bool:
        """
        Re-initializes the vault by performing a factory reset, deleting all user data and resetting the application state.

        Args:
            confirmation_phrase (str): The confirmation phrase required to authorize the factory reset.

        Returns:
            bool: True if the vault was successfully re-initialized, False otherwise.

        Behavior:
            - Checks if the provided confirmation phrase matches the expected phrase.
            - Ensures the vault is in the UNLOCKED state before proceeding.
            - Deletes the data directory and all user data.
            - Recreates necessary directories and re-initializes the security manager.
            - Sets the application state to FIRST_LAUNCH upon successful completion.
            - Logs all major actions and errors during the process.
        """

        if confirmation_phrase != self.EXPECTED_RESET_CONFIRMATION_PHRASE:
            self.logger.error(
                "Vault re-initialization failed: incorrect confirmation phrase."
            )
            return False
            # raise ValidationError(
            #     "Incorrect confirmation phrase. Factory reset has been aborted."
            # )

        with self._state_lock:
            if self._state != AppState.UNLOCKED:
                self.logger.error(
                    "Vault must be unlocked to perform re-initialization."
                )
                return False

        self.logger.warning("PERFORMING FACTORY RESET: All user data will be deleted.")

        self.lock()

        try:
            self.logger.info(f"Deleting data directory: {self.directories.data}")
            shutil.rmtree(self.directories.data)
            self.logger.info("Data directory successfully deleted.")
        except Exception as e:
            self.logger.critical(
                f"Failed to delete data directory during re-initialization: {e}"
            )
            self._set_state(AppState.ERROR)
            return False

        self.directories.create_directories()

        self.vault_manager = self._init_security_manager()

        self._set_state(AppState.FIRST_LAUNCH)
        self.logger.info(
            "Vault re-initialization complete. Application is now in FIRST_LAUNCH state."
        )
        return True

    def _initialize_unlocked_services(self):
        """
        Initializes all unlocked services managed by the orchestrator.

        Attempts to initialize the services and logs the outcome. If initialization fails,
        logs the error, sets the application state to ERROR, locks the orchestrator, and re-raises the exception.

        Raises:
            Exception: Propagates any exception encountered during service initialization.
        """
        try:
            self.services.initialize()
            self.logger.info("All unlocked services initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize unlocked services: {e}")
            self._set_state(AppState.ERROR)
            self.lock()
            raise

    def unlock(self, password: SecureString) -> bool:
        """
        Attempts to unlock the application using the provided password.

        Args:
            password (SecureString): The password to unlock the application.

        Returns:
            bool: True if the application was successfully unlocked, False otherwise.

        Behavior:
            - If the application is not in the LOCKED state, logs a warning and returns whether it is already UNLOCKED.
            - If in the LOCKED state, attempts to unlock the vault with the given password.
            - On successful unlock, updates the application state, logs the event, initializes unlocked services, and returns True.
            - On failure, logs a warning and returns False.
        """
        if self._state != AppState.LOCKED:
            self.logger.warning(
                f"Unlock attempt in non-locked state: {self._state.name}"
            )
            return self._state == AppState.UNLOCKED

        unlock_result = self.vault_manager.unlock_vault(password) # type: ignore
        if unlock_result.status == VaultUnlockStatus.SUCCESS:
            self._state = AppState.UNLOCKED
            self.logger.info("Application unlocked successfully.")
            self._initialize_unlocked_services()
            return True
        else:
            self.logger.warning("Unlock failed: incorrect password.")
            return False

    def lock(self):
        """
        Locks the vault, initiates application shutdown, updates the application state to LOCKED, and logs the action.

        This method ensures that sensitive resources are secured by locking the vault, gracefully shuts down the application,
        sets the internal state to indicate the locked status, and records the event in the application logs.
        """
        self.vault_manager.lock()
        self.shutdown()
        self._set_state(AppState.LOCKED)
        self.logger.info("Vault locked.")

    def run_task_chain(
        self, chain_definition: List[Dict[str, Any]], initial_user_query: str
    ) -> Any:
        """
        Executes a sequence of tasks defined in a chain, starting with an initial user query.

        Args:
            chain_definition (List[Dict[str, Any]]): A list of dictionaries, each representing a task in the chain with its configuration and parameters.
            initial_user_query (str): The initial input or query provided by the user to start the task chain.

        Returns:
            Any: The result produced after executing the entire task chain.

        """
        return self.services.run_task_chain(
            chain_definition=chain_definition, initial_user_query=initial_user_query
        )


    @property
    def chat(self) -> ChatManager:
        with self._state_lock:
            if self.state != AppState.UNLOCKED or self.services is None:
                raise AtaraxAIError(
                    "Application is locked. CRUD operations are not available."
                )
            return self.services.chat_manager

    @property
    def rag(self) -> AtaraxAIRAGManager:
        with self._state_lock:
            if self.state != AppState.UNLOCKED or self.services is None:
                raise AtaraxAIError(
                    "Application is locked. RAG operations are not available."
                )
            return self.services.rag_manager
        
    @property
    def model_manager(self) -> ModelManager:
        with self._state_lock:
            if self.state != AppState.UNLOCKED or self.services is None:
                raise AtaraxAIError(
                    "Application is locked. Model management operations are not available."
                )
            return self.services.model_manager

    def shutdown(self) -> None:
        self.services.shutdown()

    def __enter__(self) -> "AtaraxAIOrchestrator":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.shutdown()


class AtaraxAIOrchestratorFactory:

    @staticmethod
    def create_orchestrator() -> AtaraxAIOrchestrator:
        app_config = AppConfig()
        logger : logging.Logger = AtaraxAILogger().get_logger()
        settings = AtaraxAISettings()
        directories = AppDirectories.create_default(settings)

        vault_manager = VaultManager(
            salt_path=str(directories.data / "vault.salt"),
            check_path=str(directories.data / "vault.check"),
        )

        setup_manager = SetupManager(directories, app_config, logger)
        config_manager = ConfigurationManager(directories.config, logger)
        core_ai_manager = CoreAIServiceManager(config_manager, logger)

        db_manager = ChatDatabaseManager(
            db_path=directories.data / app_config.database_filename
        )

        chat_context = ChatContextManager(db_manager=db_manager)
        chat_manager = ChatManager(
            db_manager=db_manager, logger=logger, vault_manager=vault_manager
        )
        
        model_manager = ModelManager(
            directories=directories,
            logger=logger
        )

        services = Services(
            directories=directories,
            logger=logger,
            db_manager=db_manager,
            chat_context=chat_context,
            chat_manager=chat_manager,
            config_manager=config_manager,
            app_config=app_config,
            vault_manager=vault_manager,
            model_manager=model_manager
        )

        orchestrator = AtaraxAIOrchestrator(
            app_config=app_config,
            settings=settings,
            logger=logger,
            directories=directories,
            vault_manager=vault_manager,
            setup_manager=setup_manager,
            config_manager=config_manager,
            core_ai_manager=core_ai_manager,
            services=services,
        )

        return orchestrator
