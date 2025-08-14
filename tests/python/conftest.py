import pytest
from unittest.mock import MagicMock
from pathlib import Path
from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
)
from ataraxai.praxis.utils.app_directories import AppDirectories

# Currently I'm duplicating the fixture for function scoped and module scoped
# TODO: Refactor to avoid duplication
# I am keeping things simple for now to avoid confusion. We'll see how it goes.
from fixtures.function_scoped import *
from fixtures.module_scoped import *


import os

@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    os.environ["ENVIRONMENT"] = "development"

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


@pytest.fixture(scope="module")
def mock_orchestrator():
    """
    Pytest fixture that provides a mock instance of the AtaraxAIOrchestrator class.

    Returns:
        MagicMock: A mock object adhering to the AtaraxAIOrchestrator specification.
    """
    return MagicMock(spec=AtaraxAIOrchestrator)


@pytest.fixture(scope="module")
def test_data_dir(app_directories: AppDirectories) -> Path:
    """
    Pytest fixture that provides the path to the application's data directory.

    Args:
        app_directories (AppDirectories): An instance containing application directory paths.

    Returns:
        Path: The path to the application's data directory.
    """
    return app_directories.data


@pytest.fixture(scope="module")
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
