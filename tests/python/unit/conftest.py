import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api import app
from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestratorFactory,
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
