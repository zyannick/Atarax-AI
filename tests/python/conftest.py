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


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "unit" in str(item.path):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.path):
            item.add_marker(pytest.mark.integration)

def identity_decorator(func):
    """
    This is an identity function that acts as a decorator.
    It returns the original function completely untouched, preserving its
    critical signature for FastAPI's dependency injection.
    """
    return func


@pytest.fixture(autouse=True)
def disable_all_instrumentation(monkeypatch):
    """
    This fixture automatically runs for every test. It finds the real
    instrumentation decorator and replaces it with our safe identity_decorator.
    """
    monkeypatch.setattr(
        "ataraxai.praxis.katalepsis.katalepsis_monitor.instrument_api",
        lambda *args, **kwargs: identity_decorator,
    )


@pytest.fixture
def client(monkeypatch):
    """
    Pytest fixture that provides a test client for the FastAPI application with a mocked orchestrator.

    This fixture uses `monkeypatch` to replace the `AtaraxAIOrchestratorFactory.create_orchestrator` method
    with a lambda that returns a `MagicMock` instance. It then creates a `TestClient` for the FastAPI app,
    attaches the mocked orchestrator to the client, and yields the client for use in tests. After the test,
    it clears any dependency overrides set on the app.

    Args:
        monkeypatch: Pytest's monkeypatch fixture for safely patching objects during tests.

    Yields:
        TestClient: A FastAPI TestClient instance with a mocked orchestrator attached.
    """
    mock_orchestrator = MagicMock()

    monkeypatch.setattr(
        AtaraxAIOrchestratorFactory, "create_orchestrator", lambda: mock_orchestrator
    )

    with TestClient(app, base_url="http://test") as test_client:
        test_client.orchestrator = mock_orchestrator
        yield test_client

    app.dependency_overrides.clear()



@pytest.fixture
def app_config() -> Generator[AppConfig, None, None]:
    """
    Yields an instance of AppConfig for use in tests.

    Returns:
        Generator[AppConfig, None, None]: A generator that yields an AppConfig instance.
    """
    yield AppConfig()


@pytest.fixture
def settings() -> Generator[AtaraxAISettings, None, None]:
    """
    Pytest fixture that provides an instance of AtaraxAISettings for use in tests.

    Yields:
        AtaraxAISettings: An instance of the settings class for Atarax-AI.
    """
    yield AtaraxAISettings()


@pytest.fixture
def app_directories(tmp_path: Path) -> Generator[AppDirectories, None, None]:
    """
    Creates temporary application directories for configuration, data, cache, and logs within a given base path.
    Args:
        tmp_path (Path): The base temporary path where directories will be created.
    Yields:
        AppDirectories: An object containing paths to the created config, data, cache, and logs directories.
    """

    config = tmp_path / "config"
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    logs = tmp_path / "logs"

    config.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    app_dirs = AppDirectories(config=config, data=data, cache=cache, logs=logs)
    yield app_dirs


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def models_manager(
    app_directories: AppDirectories,
    logger: logging.Logger,
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
    models_manager = ModelsManager(directories=app_directories, logger=logger)

    yield models_manager


@pytest.fixture
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
    )

    yield services


@pytest.fixture
def orchestrator(
    app_config: AppConfig,
    settings: AtaraxAISettings,
    logger: logging.Logger,
    app_directories: AppDirectories,
    vault_manager: VaultManager,
    setup_manager: SetupManager,
    config_manager: ConfigurationManager,
    core_ai_manager: CoreAIServiceManager,
    services: Services,
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
        app_config=app_config,
        settings=settings,
        logger=logger,
        directories=app_directories,
        vault_manager=vault_manager,
        setup_manager=setup_manager,
        config_manager=config_manager,
        core_ai_manager=core_ai_manager,
        services=services,
    )

    yield orchestrator


@pytest.fixture(scope="session")
def gguf_model_path(tmp_path_factory) -> Path:
    """
    Downloads a small GGUF model once per test session for integration tests.
    This is efficient and ensures a real model is available for testing
    the Core AI Service and related components.
    """
    model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"

    session_tmp_dir = tmp_path_factory.mktemp("gguf_models")
    model_path = session_tmp_dir / "tinyllama.gguf"

    if not model_path.exists():
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    return model_path


@pytest.fixture
def small_model_info(gguf_model_path: Path) -> Generator[LlamaCPPModelInfo, None, None]:
    """
    Yields a single LlamaCPPModelInfo object containing metadata for a small GGUF model.

    Args:
        gguf_model_path (Path): The filesystem path to the GGUF model file.

    Yields:
        LlamaCPPModelInfo: An object with information about the specified GGUF model,
        including organization, repository ID, filename, local path, download time,
        file size, quantization details, creation time, downloads, and likes.
    """
    model_info = LlamaCPPModelInfo(
        organization="TheBloke",
        repo_id="TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        local_path=str(gguf_model_path),
        downloaded_at=datetime.datetime.now(),
        file_size=gguf_model_path.stat().st_size,
        quantization_bit="Q2_K",
        quantization_scheme="gguf",
        quantization_modifier="Q2_K",
        created_at=datetime.datetime.now(),
        downloads=0,
        likes=0,
    )
    yield model_info


@pytest.fixture
def llama_model_params(
    small_model_info: LlamaCPPModelInfo,
) -> Generator[LlamaModelParams, None, None]:
    """
    Yields a LlamaModelParams object with parameters for a small GGUF model.

    Args:
        small_model_info (LlamaCPPModelInfo): The model information for the small GGUF model.

    Yields:
        LlamaModelParams: An object containing the model parameters.
    """
    llama_model_params = LlamaModelParams(
        config_version=1.0,
        model_info=small_model_info,
        n_ctx=2048,
        n_gpu_layers=0,
        main_gpu=0,
        tensor_split=False,
        vocab_only=False,
        use_map=False,
        use_mlock=False,
    )
    yield llama_model_params


@pytest.fixture
def mock_orchestrator():
    """
    Pytest fixture that provides a mock instance of the AtaraxAIOrchestrator class.

    Returns:
        MagicMock: A mock object adhering to the AtaraxAIOrchestrator specification.
    """
    return MagicMock(spec=AtaraxAIOrchestrator)


@pytest.fixture
def test_data_dir(app_directories: AppDirectories) -> Path:
    """
    Pytest fixture that provides the path to the application's data directory.

    Args:
        app_directories (AppDirectories): An instance containing application directory paths.

    Returns:
        Path: The path to the application's data directory.
    """
    return app_directories.data


@pytest.fixture
def test_config_dir(app_directories: AppDirectories) -> Path:
    """
    Pytest fixture that provides the path to the application's configuration directory.

    Args:
        app_directories (AppDirectories): An instance containing application directory paths.

    Returns:
        Path: The path to the application's configuration directory.
    """
    return app_directories.config


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    """
    Pytest fixture that creates a temporary directory for session-wide tests.
    This directory is used to store session-specific test data and is cleaned up after the session ends
    Args:
        tmp_path_factory: Pytest's factory for creating temporary paths.
    Returns:
        Path: The path to the session's temporary directory.
    """
    return tmp_path_factory.mktemp("session_test_data")


@pytest.fixture(scope="module")
def module_app_directories(session_tmp_path: Path) -> AppDirectories:
    """
    Pytest fixture that creates application directories for a module-level test session.
    This fixture sets up directories for configuration, data, cache, and logs within a specified session
    temporary path, ensuring that each directory is created if it does not already exist.
    Args:
        session_tmp_path (Path): The base path for the session's temporary directory.
    Returns:
        AppDirectories: An object containing paths to the created config, data, cache, and logs directories.
    """
    config = session_tmp_path / "config"
    data = session_tmp_path / "data"
    cache = session_tmp_path / "cache"
    logs = session_tmp_path / "logs"

    for directory in [config, data, cache, logs]:
        directory.mkdir(parents=True, exist_ok=True)

    return AppDirectories(config=config, data=data, cache=cache, logs=logs)


@pytest.fixture
def integration_client(orchestrator: AtaraxAIOrchestrator):
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
        lambda: orchestrator
    )

    with TestClient(app, base_url="http://test") as test_client:
        test_client.app.state.orchestrator = orchestrator # type: ignore
        yield test_client

    app.dependency_overrides.clear()

