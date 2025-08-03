from pydantic import SecretStr
import pytest
import datetime
import time
import logging
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
from typing import Generator
from fastapi import status


import requests

from api import app
from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
    AtaraxAIOrchestratorFactory,
)
from ataraxai.praxis.modules.models_manager.models_manager import ModelsManager
from ataraxai.praxis.utils.app_config import AppConfig
from ataraxai.praxis.utils.app_directories import AppDirectories
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.ataraxai_settings import AtaraxAISettings
from ataraxai.praxis.utils.chat_manager import ChatManager
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
)
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager
from ataraxai.praxis.utils.services import Services
from ataraxai.praxis.utils.setup_manager import SetupManager
from ataraxai.praxis.utils.vault_manager import VaultManager
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import VaultPasswordRequest



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
def module_app_directories(tmp_path_factory: Path) -> Generator[AppDirectories, None, None]:
    """
    Creates temporary application directories for configuration, data, cache, and logs within a given base path.
    Args:
        tmp_path (Path): The base temporary path where directories will be created.
    Yields:
        AppDirectories: An object containing paths to the created config, data, cache, and logs directories.
    """
    module_tmp_dir = tmp_path_factory.mktemp("integration_module") # type: ignore
    config = module_tmp_dir / "config"
    data = module_tmp_dir / "data"
    cache = module_tmp_dir / "cache"
    logs = module_tmp_dir / "logs"

    config.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    app_dirs = AppDirectories(config=config, data=data, cache=cache, logs=logs)
    yield app_dirs


@pytest.fixture(scope="module")
def module_logger(module_app_directories: AppDirectories) -> Generator[logging.Logger, None, None]:
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
    setup_manager = SetupManager(module_app_directories, module_app_config, logger=module_logger)

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
    core_ai_manager = CoreAIServiceManager(config_manager=module_config_manager, logger=module_logger)

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
        db_manager=module_chat_db_manager, logger=module_logger, vault_manager=module_vault_manager
    )
    yield chat_manager


@pytest.fixture(scope="module")
def module_models_manager(
    module_app_directories: AppDirectories,
    module_logger: logging.Logger,
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
    models_manager = ModelsManager(directories=module_app_directories, logger=module_logger)

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
    )

    yield services


@pytest.fixture(scope="module")
def module_orchestrator(
    module_app_config: AppConfig,
    module_settings: AtaraxAISettings,
    module_logger: logging.Logger,
    module_app_directories: AppDirectories,
    module_vault_manager: VaultManager,
    module_setup_manager: SetupManager,
    module_config_manager: ConfigurationManager,
    module_core_ai_manager: CoreAIServiceManager,
    module_services: Services,
) -> Generator[AtaraxAIOrchestrator, None, None]:
    """
    Pytest fixture that provides an instance of AtaraxAIOrchestrator configured with the given dependencies.

    Args:
        app_config (AppConfig): The application configuration object.
        settings (AtaraxAISettings): The settings for AtaraxAI.
        logger (logging.Logger): Logger instance for logging.
        app_directories (AppDirectories): Object managing application directories.
        vault_manager (VaultManager): Manager for secure vault operations.
        setup_manager (SetupManager): Manager for application setup procedures.
        config_manager (ConfigurationManager): Manager for configuration operations.
        core_ai_manager (CoreAIServiceManager): Manager for core AI services.
        services (Services): Collection of service instances.

    Yields:
        AtaraxAIOrchestrator: An orchestrator instance initialized with the provided dependencies.
    """
    orchestrator = AtaraxAIOrchestrator(
        app_config=module_app_config,
        settings=module_settings,
        logger=module_logger,
        directories=module_app_directories,
        vault_manager=module_vault_manager,
        setup_manager=module_setup_manager,
        config_manager=module_config_manager,
        core_ai_manager=module_core_ai_manager,
        services=module_services,
    )

    yield orchestrator


@pytest.fixture(scope="module")
def module_integration_client(module_orchestrator: AtaraxAIOrchestrator):
    """
    Provides a pytest fixture that yields a TestClient instance configured with a custom AtaraxAIOrchestrator.
    Overrides the dependency injection for AtaraxAIOrchestratorFactory.create_orchestrator to use the provided orchestrator,
    allowing integration tests to run with a specific orchestrator instance. After the test client is yielded, the dependency
    overrides are cleared to restore the application's original state.
    Args:
        orchestrator (AtaraxAIOrchestrator): The orchestrator instance to inject into the application for testing.
    Yields:
        TestClient: A FastAPI TestClient configured with the overridden orchestrator dependency.
    """
    app.dependency_overrides[AtaraxAIOrchestratorFactory.create_orchestrator] = (
        lambda: module_orchestrator
    )

    with TestClient(app, base_url="http://test") as test_client:
        test_client.app.state.orchestrator = module_orchestrator # type: ignore
        yield test_client

    app.dependency_overrides.clear()