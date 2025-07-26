import pytest
import functools
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Import your actual app and factory
from api import app
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.app_state import AppState

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
        lambda *args, **kwargs: identity_decorator
    )
    


@pytest.fixture
def client(monkeypatch):
    mock_orchestrator = MagicMock()

    monkeypatch.setattr(
        AtaraxAIOrchestratorFactory,
        "create_orchestrator",
        lambda: mock_orchestrator
    )

    with TestClient(app, base_url="http://test") as test_client:
        test_client.orchestrator = mock_orchestrator
        yield test_client

    app.dependency_overrides.clear()
    
    

