import asyncio
import logging
import time
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api import app
from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestratorFactory,
)
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.models_manager.models_manager import ModelsManager
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

# @pytest.fixture(scope="function")
# def client(monkeypatch):
#     """
#     Pytest fixture that provides a test client for the FastAPI application with a mocked orchestrator.

#     This fixture uses `monkeypatch` to replace the `AtaraxAIOrchestratorFactory.create_orchestrator` method
#     with a lambda that returns a `MagicMock` instance. It then creates a `TestClient` for the FastAPI app,
#     attaches the mocked orchestrator to the client, and yields the client for use in tests. After the test,
#     it clears any dependency overrides set on the app.

#     Args:
#         monkeypatch: Pytest's monkeypatch fixture for safely patching objects during tests.

#     Yields:
#         TestClient: A FastAPI TestClient instance with a mocked orchestrator attached.
#     """
#     mock_orchestrator = MagicMock()

#     monkeypatch.setattr(
#         AtaraxAIOrchestratorFactory, "create_orchestrator", lambda: mock_orchestrator
#     )

#     with TestClient(app, base_url="http://test") as test_client:
#         test_client.orchestrator = mock_orchestrator  # type: ignore
#         yield test_client

#     app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    mock_orchestrator = AsyncMock()

    async def mock_create_orchestrator():
        return mock_orchestrator

    TEST_TOKEN = "Satchel-Darwinism-Croak7-Conjure-Counting"

    monkeypatch.setattr(
        AtaraxAIOrchestratorFactory, "create_orchestrator", mock_create_orchestrator
    )

    app.dependency_overrides[verify_token] = lambda: None

    with TestClient(app, base_url="http://test") as test_client:

        app.state.secret_token = TEST_TOKEN

        test_client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def app_config() -> Generator[AppConfig, None, None]:
    """
    Yields an instance of AppConfig for use in tests.

    Returns:
        Generator[AppConfig, None, None]: A generator that yields an AppConfig instance.
    """
    yield AppConfig()


@pytest.fixture(scope="function")
def settings() -> Generator[AtaraxAISettings, None, None]:
    """
    Pytest fixture that provides an instance of AtaraxAISettings for use in tests.

    Yields:
        AtaraxAISettings: An instance of the settings class for Atarax-AI.
    """
    yield AtaraxAISettings()


@pytest.fixture(scope="function")
def app_directories(tmp_path_factory: Path) -> Generator[AppDirectories, None, None]:
    """
    Creates temporary application directories for configuration, data, cache, and logs within a given base path.
    Args:
        tmp_path (Path): The base temporary path where directories will be created.
    Yields:
        AppDirectories: An object containing paths to the created config, data, cache, and logs directories.
    """
    module_tmp_dir = tmp_path_factory.mktemp("integration_module")  # type: ignore
    config = module_tmp_dir / "config"
    data = module_tmp_dir / "data"
    cache = module_tmp_dir / "cache"
    logs = module_tmp_dir / "logs"

    config.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    print(f"Created temporary directories at {module_tmp_dir}")

    app_dirs = AppDirectories(config=config, data=data, cache=cache, logs=logs)
    yield app_dirs


@pytest.fixture(scope="function")
def logger(app_directories: AppDirectories) -> Generator[logging.Logger, None, None]:
    """
    Creates and yields a configured logger for the application.

    Args:
        app_directories (AppDirectories): An object containing application directory paths, including the log directory.

    Yields:
        logging.Logger: A logger instance configured to write logs to the specified log directory.
    """
    logger = AtaraxAILogger(log_dir=app_directories.logs).get_logger()
    yield logger


@pytest.fixture(scope="function")
def vault_manager(
    app_directories: AppDirectories,
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
    salt_path = app_directories.cache / "vault.salt"
    check_path = app_directories.cache / "vault.check"

    vault_manager = VaultManager(salt_path=str(salt_path), check_path=str(check_path))

    yield vault_manager

    for path in [salt_path, check_path]:
        if path.exists():
            path.unlink()


@pytest.fixture(scope="function")
def setup_manager(
    app_directories: AppDirectories,
    app_config: AppConfig,
    logger: logging.Logger,
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
    setup_manager = SetupManager(app_directories, app_config, logger=logger)

    yield setup_manager


@pytest.fixture(scope="function")
def config_manager(
    app_directories: AppDirectories,
    logger: logging.Logger,
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
        config_dir=app_directories.config, logger=logger
    )

    yield config_manager


@pytest.fixture(scope="function")
def core_ai_manager(
    config_manager: ConfigurationManager,
    logger: logging.Logger,
) -> Generator[CoreAIServiceManager, None, None]:
    """
    Pytest fixture that provides an instance of CoreAIServiceManager for use in tests.
    Args:
        config_manager (ConfigurationManager): The configuration manager instance.
        logger (logging.Logger): The logger instance.
    Yields:
        CoreAIServiceManager: An initialized CoreAIServiceManager object.
    """
    core_ai_manager = CoreAIServiceManager(config_manager=config_manager, logger=logger)

    yield core_ai_manager


@pytest.fixture(scope="function")
def chat_db_manager(
    app_directories: AppDirectories,
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
    db_path = app_directories.data / "test_chat.db"

    db_manager = ChatDatabaseManager(db_path=db_path)

    yield db_manager

    db_manager.close()
    time.sleep(0.1)  # Ensure the database is closed before deleting
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="function")
def chat_context_manager(
    chat_db_manager: ChatDatabaseManager, vault_manager: VaultManager
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
        db_manager=chat_db_manager, vault_manager=vault_manager
    )
    yield context_manager


@pytest.fixture(scope="function")
def chat_manager(
    chat_db_manager: ChatDatabaseManager,
    logger: logging.Logger,
    vault_manager: VaultManager,
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
        db_manager=chat_db_manager, logger=logger, vault_manager=vault_manager
    )
    yield chat_manager


@pytest.fixture(scope="function")
def background_task_manager() -> Generator[BackgroundTaskManager, None, None]:
    """
    Pytest fixture that provides a BackgroundTaskManager instance for tests.

    Yields:
        BackgroundTaskManager: An instance of BackgroundTaskManager.
    """
    background_task_manager = BackgroundTaskManager()
    yield background_task_manager


@pytest.fixture(scope="function")
def models_manager(
    app_directories: AppDirectories,
    logger: logging.Logger,
    background_task_manager: BackgroundTaskManager,
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
        directories=app_directories,
        logger=logger,
        background_task_manager=background_task_manager,
    )

    yield models_manager


@pytest.fixture(scope="function")
def services(
    app_directories: AppDirectories,
    logger: logging.Logger,
    chat_db_manager: ChatDatabaseManager,
    chat_context_manager: ChatContextManager,
    chat_manager: ChatManager,
    config_manager: ConfigurationManager,
    app_config: AppConfig,
    vault_manager: VaultManager,
    models_manager: ModelsManager,
    core_ai_manager: CoreAIServiceManager,
    background_task_manager: BackgroundTaskManager,
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
        directories=app_directories,
        logger=logger,
        db_manager=chat_db_manager,
        chat_context=chat_context_manager,
        chat_manager=chat_manager,
        config_manager=config_manager,
        app_config=app_config,
        vault_manager=vault_manager,
        models_manager=models_manager,
        core_ai_service_manager=core_ai_manager,
        background_task_manager=background_task_manager,
    )

    yield services


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def integration_client(
    event_loop: asyncio.AbstractEventLoop, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """
    The master fixture for integration tests. It patches the orchestrator factory,
    then creates a TestClient, which triggers the lifespan manager.
    """
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        orchestrator = event_loop.run_until_complete(
            setup_async_orchestrator(temp_dir_path)
        )

        TEST_TOKEN = "Satchel-Darwinism-Croak7-Conjure-Counting"

        monkeypatch.setattr(
            AtaraxAIOrchestratorFactory,
            "create_orchestrator",
            lambda: asyncio.sleep(0, result=orchestrator),
        )

        app.dependency_overrides[verify_token] = lambda: None

        with TestClient(app, base_url="http://test") as client:

            app.state.secret_token = TEST_TOKEN

            client.headers.update({"Authorization": f"Bearer {TEST_TOKEN}"})
            yield client

        event_loop.run_until_complete(orchestrator.shutdown())
