import pytest
import functools
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
from typing import Generator

# Import your actual app and factory
from api import app
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.vault_manager import VaultManager
from ataraxai.praxis.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager


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
    mock_orchestrator = MagicMock()

    monkeypatch.setattr(
        AtaraxAIOrchestratorFactory, "create_orchestrator", lambda: mock_orchestrator
    )

    with TestClient(app, base_url="http://test") as test_client:
        test_client.orchestrator = mock_orchestrator
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def chat_db_manager(tmp_path: Path) -> Generator[ChatDatabaseManager, None, None]:
    """
    Pytest fixture that provides a temporary ChatDatabaseManager instance for testing.

    Creates a ChatDatabaseManager using a temporary SQLite database file located in a pytest-provided temporary directory.
    Yields the manager to the test, and ensures the database connection is closed after the test completes.

    Args:
        tmp_path (Path): Temporary directory provided by pytest for storing test files.

    Yields:
        ChatDatabaseManager: An instance connected to the temporary test database.
    """
    db_path = tmp_path / "test_chat.db"

    db_manager = ChatDatabaseManager(db_path=db_path)

    yield db_manager

    db_manager.close()


@pytest.fixture
def vault_manager(tmp_path: Path) -> Generator[VaultManager, None, None]:
    """
    Fixture that provides a VaultManager instance for testing.

    Args:
        tmp_path (Path): Temporary directory provided by pytest for test isolation.

    Yields:
        VaultManager: An instance of VaultManager initialized with temporary salt and check paths.
    """
    salt_path = tmp_path
    check_path = tmp_path

    vault_manager = VaultManager(salt_path=str(salt_path), check_path=str(check_path))

    yield vault_manager


@pytest.fixture
def chat_context_manager(
    chat_db_manager: ChatDatabaseManager, vault_manager: VaultManager
) -> Generator[ChatContextManager, None, None]:
    """
    Creates and yields a ChatContextManager instance using the provided chat database and vault managers.

    Args:
        chat_db_manager (ChatDatabaseManager): The manager responsible for chat database operations.
        vault_manager (VaultManager): The manager responsible for vault operations.

    Yields:
        ChatContextManager: An instance of ChatContextManager initialized with the given managers.
    """
    context_manager = ChatContextManager(
        db_manager=chat_db_manager, vault_manager=vault_manager
    )
    yield context_manager
