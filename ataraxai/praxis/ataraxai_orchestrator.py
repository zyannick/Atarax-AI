import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
from contextlib import asynccontextmanager
from ataraxai import __version__  # type: ignore
from ataraxai.hegemonikon_py import SecureString  # type: ignore

from ataraxai.praxis.modules.models_manager.models_manager import ModelsManager
from ataraxai.praxis.modules.prompt_engine.task_manager import TaskManager
from ataraxai.praxis.utils.vault_manager import (
    UnlockResult,
    VaultInitializationStatus,
    VaultManager,
)
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
import threading
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.exceptions import (
    AtaraxAILockError,
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
from ataraxai.praxis.utils.user_preferences_manager import UserPreferencesManager
from ataraxai.praxis.utils.services import Services


class AtaraxAIOrchestrator:

    EXPECTED_RESET_CONFIRMATION_PHRASE = "reset ataraxai vault"

    def __init__(
        self,
        settings: AtaraxAISettings,
        setup_manager: SetupManager,
        services: Services,
        logger: logging.Logger,
    ):
        self.settings = settings
        self.setup_manager = setup_manager
        self.services = services
        self.logger = logger

        self._state_lock = asyncio.Lock()
        self._state: AppState = AppState.LOCKED
        self._initialized = False
        self._shutdown = False

    async def __aenter__(self) -> "AtaraxAIOrchestrator":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()

    async def initialize(self) -> None:
        if self._initialized:
            self.logger.warning("Orchestrator already initialized")
            return

        try:
            async with self._state_lock:
                initial_state = await self._determine_initial_state()
                self._state = initial_state

                if self._state == AppState.FIRST_LAUNCH:
                    await self._initialize_base_components()

                self._initialized = True
                self.logger.info(
                    f"Orchestrator initialized. Current state: {self._state.name}"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
            await self._set_state(AppState.ERROR)
            raise

    async def get_state(self) -> AppState:
        async with self._state_lock:
            return self._state

    async def _set_state(self, new_state: AppState) -> None:
        async with self._state_lock:
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                self.logger.info(
                    f"State transition: {old_state.name} -> {new_state.name}"
                )

    async def _determine_initial_state(self) -> AppState:
        try:
            if self.services and self.services.vault_manager:
                check_path = Path(self.services.vault_manager.check_path)
                path_exists = await asyncio.to_thread(check_path.exists)
                return AppState.LOCKED if path_exists else AppState.FIRST_LAUNCH
            else:
                self.logger.error(
                    "Vault manager not initialized, defaulting to FIRST_LAUNCH state"
                )
                return AppState.FIRST_LAUNCH
        except Exception as e:
            self.logger.error(f"Error determining initial state: {e}", exc_info=True)
            return AppState.ERROR

    async def _initialize_base_components(self) -> None:
        try:
            self.logger.info(f"Starting AtaraxAI v{__version__}")
            if self.setup_manager:
                if await asyncio.to_thread(self.setup_manager.is_first_launch):
                    await asyncio.to_thread(
                        self.setup_manager.perform_first_launch_setup
                    )
            else:
                self.logger.error(
                    "Setup manager is not initialized, cannot perform first launch setup"
                )
                raise RuntimeError("Setup manager is not initialized")
        except Exception as e:
            self.logger.error(f"Failed during base initialization: {e}", exc_info=True)
            self._state = AppState.ERROR
            raise

    async def initialize_new_vault(
        self, master_password: SecureString
    ) -> VaultInitializationStatus:
        current_state = await self.get_state()
        if current_state != AppState.FIRST_LAUNCH:
            self.logger.error(
                f"Attempted to initialize vault in state: {current_state.name}"
            )
            return VaultInitializationStatus.ALREADY_INITIALIZED

        try:
            if self.services is None or self.services.vault_manager is None:
                self.logger.error("Vault manager is not initialized.")
                return VaultInitializationStatus.FAILED

            await asyncio.to_thread(
                self.services.vault_manager.create_and_initialize_vault, master_password
            )
            await self._set_state(AppState.UNLOCKED)
            await self._initialize_unlocked_services()
            self.logger.info("Vault successfully initialized and unlocked.")
            return VaultInitializationStatus.SUCCESS

        except Exception as e:
            self.logger.error(f"Failed to initialize new vault: {e}", exc_info=True)
            await self._set_state(AppState.ERROR)
            return VaultInitializationStatus.FAILED

    async def reinitialize_vault(self, confirmation_phrase: str) -> bool:
        if confirmation_phrase != self.EXPECTED_RESET_CONFIRMATION_PHRASE:
            self.logger.error(
                "Vault re-initialization failed: incorrect confirmation phrase."
            )
            return False

        current_state = await self.get_state()
        if current_state != AppState.UNLOCKED:
            self.logger.error(
                f"Vault must be unlocked to perform re-initialization. Current state: {current_state.name}"
            )
            return False

        self.logger.warning("PERFORMING FACTORY RESET: All user data will be deleted.")

        try:
            await self.lock()

            directories = await self.get_directories()
            data_dir = directories.data
            self.logger.info(f"Deleting data directory: {data_dir}")
            await asyncio.to_thread(shutil.rmtree, data_dir)
            self.logger.info("Data directory successfully deleted.")

            await asyncio.to_thread(directories.create_directories)

            await self._reinitialize_vault_manager()

            await self._set_state(AppState.FIRST_LAUNCH)
            self.logger.info(
                "Vault re-initialization complete. Application is now in FIRST_LAUNCH state."
            )
            return True

        except Exception as e:
            self.logger.critical(
                f"Failed during vault re-initialization: {e}", exc_info=True
            )
            await self._set_state(AppState.ERROR)
            return False

    async def _reinitialize_vault_manager(self) -> None:
        directories = await self.get_directories()
        salt_file = directories.data / "vault.salt"
        check_file = directories.data / "vault.check"
        new_vault_manager = VaultManager(
            salt_path=str(salt_file), check_path=str(check_file)
        )
        if self.services is None:
            self.logger.error(
                "Services are not initialized, cannot set new vault manager."
            )
            return
        self.services.vault_manager = new_vault_manager

    async def _initialize_unlocked_services(self) -> None:
        try:
            if self.services is None:
                self.logger.error(
                    "Services are not initialized, cannot initialize unlocked services."
                )
                return
            await self.services.initialize()
            self.logger.info("All unlocked services initialized successfully.")
        except Exception as e:
            self.logger.error(
                f"Failed to initialize unlocked services: {e}", exc_info=True
            )
            await self._set_state(AppState.ERROR)
            await self.lock()
            raise

    async def unlock(self, password: SecureString) -> UnlockResult:
        current_state = await self.get_state()
        if current_state != AppState.LOCKED:
            self.logger.warning(
                f"Unlock attempt in non-locked state: {current_state.name}"
            )
            return UnlockResult(
                status=VaultUnlockStatus.ALREADY_UNLOCKED,
                error="Application is already unlocked.",
            )

        try:
            if self.services is None or self.services.vault_manager is None:
                self.logger.error("Vault manager is not initialized, cannot unlock.")
                return UnlockResult(
                    status=VaultUnlockStatus.ERROR,
                    error="Vault manager is not initialized.",
                )
            unlock_result = await asyncio.to_thread(
                self.services.vault_manager.unlock_vault, password
            )

            if unlock_result.status == VaultUnlockStatus.SUCCESS:
                await self._set_state(AppState.UNLOCKED)
                await self._initialize_unlocked_services()
                self.logger.info("Application unlocked successfully.")
            else:
                self.logger.warning(f"Unlock failed: {unlock_result.error}")

            return unlock_result

        except Exception as e:
            self.logger.error(f"Exception during unlock: {e}", exc_info=True)
            await self._set_state(AppState.ERROR)
            return UnlockResult(
                status=VaultUnlockStatus.ERROR,
                error=f"Internal error during unlock: {str(e)}",
            )

    async def lock(self) -> bool:
        try:
            if self.services is None or self.services.vault_manager is None:
                self.logger.error("Vault manager is not initialized, cannot lock.")
                return False
            await asyncio.to_thread(self.services.vault_manager.lock)
            await self._shutdown_services()
            await self._set_state(AppState.LOCKED)
            self.logger.info("Vault locked successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to lock vault: {e}", exc_info=True)
            await self._set_state(AppState.ERROR)
            return False

    async def run_task_chain(
        self, chain_definition: List[Dict[str, Any]], initial_user_query: str
    ) -> Any:
        current_state = await self.get_state()
        if current_state != AppState.UNLOCKED:
            raise AtaraxAILockError(
                f"Cannot run task chain in state: {current_state.name}"
            )
        if self.services is None:
            raise RuntimeError("Services are not initialized, cannot run task chain.")
        return await self.services.run_task_chain(
            chain_definition=chain_definition, initial_user_query=initial_user_query
        )

    async def _shutdown_services(self) -> None:
        if hasattr(self.services, "shutdown") and callable(
            getattr(self.services, "shutdown")
        ):
            if self.services is not None:
                self.logger.info("Shutting down services...")
                await self.services.shutdown()
            else:
                self.logger.warning("Services are not initialized, skipping shutdown.")

    async def shutdown(self) -> None:
        if self._shutdown:
            return

        self._shutdown = True

        try:
            await self._shutdown_services()
            self.logger.info("Orchestrator shutdown complete.")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            self.services = None
            self.setup_manager = None

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        if self.services is None:
            raise RuntimeError("Services container is not available.")

    async def get_directories(self) -> AppDirectories:
        async with self._state_lock:
            current_state = await self.get_state()
            if current_state != AppState.UNLOCKED or self.services is None:
                raise AtaraxAILockError(
                    "Application is locked. Directory operations are not available."
                )
            return self.services.directories

    async def get_rag_manager(self) -> AtaraxAIRAGManager:
        async with self._state_lock:
            current_state = await self.get_state()
            if current_state != AppState.UNLOCKED or self.services is None:
                raise AtaraxAILockError(
                    "Application is locked. RAG operations are not available."
                )
            return self.services.rag_manager

    async def get_vault_manager(self) -> VaultManager:
        async with self._state_lock:
            if self.services is None or self.services.vault_manager is None:
                raise RuntimeError("Vault manager is not initialized.")
            return self.services.vault_manager

    async def get_config_manager(self) -> ConfigurationManager:
        async with self._state_lock:
            if self.services is None or self.services.config_manager is None:
                raise RuntimeError("Config manager is not initialized.")
            return self.services.config_manager

    async def get_core_ai_service_manager(self) -> CoreAIServiceManager:
        async with self._state_lock:
            if self.services is None or self.services.core_ai_service_manager is None:
                raise RuntimeError("Core AI service manager is not initialized.")
            return self.services.core_ai_service_manager

    async def get_app_config(self) -> AppConfig:
        async with self._state_lock:
            if self.services is None or self.services.app_config is None:
                raise RuntimeError("App config is not initialized.")
            return self.services.app_config

    async def get_chat_context(self) -> ChatContextManager:
        async with self._state_lock:
            if self.services is None or self.services.chat_context is None:
                raise RuntimeError("Chat context is not initialized.")
            return self.services.chat_context

    async def get_chat_manager(self) -> ChatManager:
        async with self._state_lock:
            if self.services is None or self.services.chat_manager is None:
                raise RuntimeError("Chat manager is not initialized.")
            return self.services.chat_manager

    async def get_models_manager(self) -> ModelsManager:
        async with self._state_lock:
            if self.services is None or self.services.models_manager is None:
                raise RuntimeError("Models manager is not initialized.")
            return self.services.models_manager

    async def get_task_manager(self) -> TaskManager:
        async with self._state_lock:
            if self.services is None or self.services.task_manager is None:
                raise RuntimeError("Task manager is not initialized.")
            return self.services.task_manager
            raise RuntimeError(
                "Services are not initialized, cannot access task manager."
            )
        return self.services.task_manager


class AtaraxAIOrchestratorFactory:

    @staticmethod
    async def create_orchestrator() -> AtaraxAIOrchestrator:
        try:
            app_config = AppConfig()
            settings = AtaraxAISettings()
            directories = AppDirectories.create_default(settings)

            await asyncio.to_thread(directories.create_directories)

            logger = AtaraxAILogger(log_dir=directories.logs).get_logger()
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

            chat_context = ChatContextManager(
                db_manager=db_manager, vault_manager=vault_manager
            )

            chat_manager = ChatManager(
                db_manager=db_manager, logger=logger, vault_manager=vault_manager
            )

            models_manager = ModelsManager(directories=directories, logger=logger)

            services = Services(
                directories=directories,
                logger=logger,
                db_manager=db_manager,
                chat_context=chat_context,
                chat_manager=chat_manager,
                config_manager=config_manager,
                app_config=app_config,
                vault_manager=vault_manager,
                models_manager=models_manager,
                core_ai_service_manager=core_ai_manager,
            )

            orchestrator = AtaraxAIOrchestrator(
                settings=settings,
                setup_manager=setup_manager,
                services=services,
                logger=logger,
            )

            await orchestrator.initialize()

            return orchestrator

        except Exception as e:
            raise


@asynccontextmanager
async def create_orchestrator():
    orchestrator = await AtaraxAIOrchestratorFactory.create_orchestrator()
    try:
        yield orchestrator
    finally:
        await orchestrator.shutdown()
