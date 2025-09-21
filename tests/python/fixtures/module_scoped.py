import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api import app
from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestratorFactory,
)
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.models_manager.models_manager import (
    ModelsManager,
)
from ataraxai.praxis.utils.app_config import AppConfig
from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.ataraxai_settings import AtaraxAISettings
from ataraxai.praxis.utils.background_task_manager import BackgroundTaskManager
from ataraxai.praxis.utils.chat_manager import ChatManager
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager
from ataraxai.praxis.utils.services import Services
from ataraxai.praxis.utils.setup_manager import SetupManager
from ataraxai.praxis.utils.vault_manager import VaultManager
from ataraxai.routes.dependency_api import verify_token
from tests.python.fixtures.async_orch import setup_async_orchestrator

# module scope fixtures


@pytest.fixture(scope="module")
def module_app_config() -> Generator[AppConfig, None, None]:
    """
    Yields an instance of AppConfig for use in tests.

    Returns:
        Generator[AppConfig, None, None]: A generator that yields an AppConfig instance.
    """
    yield AppConfig()


@pytest.fixture(scope="module")
def module_settings() -> Generator[AtaraxAISettings, None, None]:
    """
    Pytest fixture that provides an instance of AtaraxAISettings for use in tests.

    Yields:
        AtaraxAISettings: An instance of the settings class for Atarax-AI.
    """
    yield AtaraxAISettings()


@pytest.fixture(scope="module")
def module_app_directories(
    tmp_path_factory: Path,
) -> Generator[AppDirectories, None, None]:
    """
    Creates temporary application directories for configuration, data, cache, and logs within a given base path.
    Args:
        tmp_path (Path): The base temporary path where directories will be created.
    Yields:
        AppDirectories: An object containing paths to the created config, data, cache, and logs directories.
    """
    module_tmp_dir: Path = tmp_path_factory.mktemp("integration_module")  # type: ignore
    config = module_tmp_dir / "config"
    data = module_tmp_dir / "data"
    cache = module_tmp_dir / "cache"
    logs = module_tmp_dir / "logs"

    config.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    directories = AppDirectories(config=config, data=data, cache=cache, logs=logs)
    # await asyncio.to_thread(directories.create_directories)
    yield directories


@pytest.fixture(scope="module")
def module_logger(
    module_app_directories: AppDirectories,
) -> Generator[logging.Logger, None, None]:
    """
    Creates and yields a configured logger for the application.

    Args:
        app_directories (AppDirectories): An object containing application directory paths, including the log directory.

    Yields:
        logging.Logger: A logger instance configured to write logs to the specified log directory.
    """
    logger = AtaraxAILogger(log_dir=module_app_directories.logs).get_logger()
    yield logger


@pytest.fixture(scope="module")
def module_vault_manager(
    module_app_directories: AppDirectories,
) -> Generator[VaultManager, None, None]:
    """
    Creates and yields a VaultManager instance using provided application directories.
    This function initializes a VaultManager with paths for salt and check files located in the application's cache directory.
    After yielding the VaultManager for use (typically in a test fixture context), it ensures cleanup by deleting the salt and check files if they exist.
    Args:
        app_directories (AppDirectories): An object containing application directory paths, specifically the cache directory.
    Yields:
        VaultManager: An instance of VaultManager initialized with the appropriate salt and check file paths.
    """
    salt_path = module_app_directories.cache / "vault.salt"
    check_path = module_app_directories.cache / "vault.check"

    vault_manager = VaultManager(salt_path=str(salt_path), check_path=str(check_path))

    yield vault_manager

    for path in [salt_path, check_path]:
        if path.exists():
            path.unlink()


@pytest.fixture(scope="module")
def module_setup_manager(
    module_app_directories: AppDirectories,
    module_app_config: AppConfig,
    module_logger: logging.Logger,
) -> Generator[SetupManager, None, None]:
    """
    Pytest fixture that initializes and yields a SetupManager instance.

    Args:
        app_directories (AppDirectories): The application directories configuration.
        app_config (AppConfig): The application configuration object.
        logger (logging.Logger): Logger instance for logging setup events.

    Yields:
        SetupManager: An initialized SetupManager object for use in tests.
    """
    setup_manager = SetupManager(
        module_app_directories, module_app_config, logger=module_logger
    )

    yield setup_manager


@pytest.fixture(scope="module")
def module_config_manager(
    module_app_directories: AppDirectories,
    module_logger: logging.Logger,
) -> Generator[ConfigurationManager, None, None]:
    """
    Pytest fixture that provides a ConfigurationManager instance.
    Args:
        app_directories (AppDirectories): Object containing application directory paths.
        logger (logging.Logger): Logger instance for logging configuration events.
    Yields:
        ConfigurationManager: An instance of ConfigurationManager initialized with the provided config directory and logger.
    """
    config_manager = ConfigurationManager(
        config_dir=module_app_directories.config, logger=module_logger
    )

    yield config_manager


@pytest.fixture(scope="module")
def module_core_ai_manager(
    module_config_manager: ConfigurationManager,
    module_logger: logging.Logger,
) -> Generator[CoreAIServiceManager, None, None]:
    """
    Pytest fixture that provides an instance of CoreAIServiceManager for use in tests.
    Args:
        config_manager (ConfigurationManager): The configuration manager instance.
        logger (logging.Logger): The logger instance.
    Yields:
        CoreAIServiceManager: An initialized CoreAIServiceManager object.
    """
    core_ai_manager = CoreAIServiceManager(
        config_manager=module_config_manager, logger=module_logger
    )

    yield core_ai_manager


@pytest.fixture(scope="module")
def module_chat_db_manager(
    module_app_directories: AppDirectories,
) -> Generator[ChatDatabaseManager, None, None]:
    """
    Pytest fixture that provides a ChatDatabaseManager instance for testing.
    This fixture creates a temporary chat database at the specified path within the application's data directory.
    After yielding the ChatDatabaseManager for use in tests, it ensures proper cleanup by closing the manager and
    removing the test database file if it exists.
    Args:
        app_directories (AppDirectories): An object containing application directory paths.
    Yields:
        ChatDatabaseManager: An instance connected to the temporary test database.
    """
    db_path = module_app_directories.data / "test_chat.db"

    db_manager = ChatDatabaseManager(db_path=db_path)

    yield db_manager

    db_manager.close()
    time.sleep(0.1)  # Ensure the database is closed before deleting
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="module")
def module_chat_context_manager(
    module_chat_db_manager: ChatDatabaseManager, module_vault_manager: VaultManager
) -> Generator[ChatContextManager, None, None]:
    """
    Pytest fixture that provides a ChatContextManager instance for tests.

    Args:
        chat_db_manager (ChatDatabaseManager): The chat database manager dependency.
        vault_manager (VaultManager): The vault manager dependency.

    Yields:
        ChatContextManager: An instance of ChatContextManager initialized with the provided managers.
    """
    context_manager = ChatContextManager(
        db_manager=module_chat_db_manager, vault_manager=module_vault_manager
    )
    yield context_manager


@pytest.fixture(scope="module")
def module_chat_manager(
    module_chat_db_manager: ChatDatabaseManager,
    module_logger: logging.Logger,
    module_vault_manager: VaultManager,
) -> Generator[ChatManager, None, None]:
    """
    Pytest fixture that provides a ChatManager instance for tests.

    Args:
        chat_db_manager (ChatDatabaseManager): The database manager for chat data.
        logger (logging.Logger): Logger instance for logging.
        vault_manager (VaultManager): Manager for secure vault operations.

    Yields:
        ChatManager: An instance of ChatManager configured with the provided dependencies.
    """
    chat_manager = ChatManager(
        db_manager=module_chat_db_manager,
        logger=module_logger,
        vault_manager=module_vault_manager,
    )
    yield chat_manager


@pytest.fixture(scope="module")
def module_background_task_manager() -> Generator[BackgroundTaskManager, None, None]:
    """
    Pytest fixture that provides a BackgroundTaskManager instance for tests.

    Yields:
        BackgroundTaskManager: An instance of BackgroundTaskManager.
    """
    background_task_manager = BackgroundTaskManager()
    yield background_task_manager


@pytest.fixture(scope="module")
def module_models_manager(
    module_app_directories: AppDirectories,
    module_logger: logging.Logger,
    module_background_task_manager: BackgroundTaskManager,
) -> Generator[ModelsManager, None, None]:
    """
    Pytest fixture that provides a `ModelsManager` instance initialized with the given
    `app_directories` and `logger`. Yields the manager for use in tests.
    Args:
        app_directories (AppDirectories): The application directories configuration.
        logger (logging.Logger): Logger instance for logging within the manager.
    Yields:
        ModelsManager: An instance of ModelsManager for managing models during tests.
    """
    models_manager = ModelsManager(
        directories=module_app_directories,
        logger=module_logger,
        background_task_manager=module_background_task_manager,
    )

    yield models_manager


@pytest.fixture(scope="module")
def module_services(
    module_app_directories: AppDirectories,
    module_logger: logging.Logger,
    module_chat_db_manager: ChatDatabaseManager,
    module_chat_context_manager: ChatContextManager,
    module_chat_manager: ChatManager,
    module_config_manager: ConfigurationManager,
    module_app_config: AppConfig,
    module_vault_manager: VaultManager,
    module_models_manager: ModelsManager,
    module_core_ai_manager: CoreAIServiceManager,
    module_background_task_manager: BackgroundTaskManager,
) -> Generator[Services, None, None]:
    """
    Creates and yields a `Services` instance initialized with the provided application components.
    Args:
        app_directories (AppDirectories): Manages application directory paths.
        logger (logging.Logger): Logger instance for application logging.
        chat_db_manager (ChatDatabaseManager): Handles chat database operations.
        chat_context_manager (ChatContextManager): Manages chat context state.
        chat_manager (ChatManager): Orchestrates chat interactions.
        config_manager (ConfigurationManager): Manages application configuration.
        app_config (AppConfig): Application configuration settings.
        vault_manager (VaultManager): Handles secure storage operations.
        models_manager (ModelsManager): Manages AI models.
    Yields:
        Services: An initialized Services object containing all provided managers and configurations.
    """
    services = Services(
        directories=module_app_directories,
        logger=module_logger,
        db_manager=module_chat_db_manager,
        chat_context=module_chat_context_manager,
        chat_manager=module_chat_manager,
        config_manager=module_config_manager,
        app_config=module_app_config,
        vault_manager=module_vault_manager,
        models_manager=module_models_manager,
        core_ai_service_manager=module_core_ai_manager,
        background_task_manager=module_background_task_manager,
    )

    yield services


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def module_integration_client(
    event_loop: asyncio.AbstractEventLoop,
) -> Generator[TestClient, None, None]:

    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)

    orchestrator = None
    try:
        orchestrator = event_loop.run_until_complete(
            setup_async_orchestrator(temp_dir_path)
        )

        TEST_TOKEN = "Satchel-Darwinism-Croak7-Conjure-Counting"

        app.dependency_overrides[verify_token] = lambda: None

        with mock.patch.object(
            AtaraxAIOrchestratorFactory,
            "create_orchestrator",
            return_value=orchestrator,
        ):
            with TestClient(app, base_url="http://test") as client:
                app.state.secret_token = TEST_TOKEN

                client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
                yield client

    finally:
        if orchestrator:
            event_loop.run_until_complete(orchestrator.shutdown())

        shutil.rmtree(temp_dir)


# @pytest.fixture(scope="module")
# def integration_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
#     """
#     The master fixture for integration tests. It creates a temporary directory,
#     initializes a real orchestrator, patches the factory, and yields a TestClient.
#     It ensures graceful shutdown and cleanup in the correct order.
#     """
#     # 1. Create the temporary directory manually.
#     temp_dir = tempfile.mkdtemp()
#     temp_dir_path = Path(temp_dir)

#     # Create a new event loop for this test module.
#     loop = asyncio.get_event_loop_policy().new_event_loop()
#     asyncio.set_event_loop(loop)

#     try:
#         # 2. Run the async setup to get a fully initialized orchestrator.
#         orchestrator = loop.run_until_complete(setup_async_orchestrator(temp_dir_path))

#         # 3. Patch the factory to return our test-specific orchestrator.
#         #    The lambda must be async to match the original signature.
#         monkeypatch.setattr(
#             AtaraxAIOrchestratorFactory,
#             "create_orchestrator",
#             lambda: asyncio.sleep(0, result=orchestrator),
#         )

#         # 4. Create the TestClient, which will now use our patched factory.
#         with TestClient(app, base_url="http://test") as client:
#             yield client

#     finally:
#         print("\n--- Tearing down integration test environment ---")
#         if "orchestrator" in locals() and orchestrator:
#             loop.run_until_complete(orchestrator.shutdown())

#         shutil.rmtree(temp_dir)

#         loop.close()
