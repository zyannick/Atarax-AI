import asyncio
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api import app
from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestratorFactory,
)
from ataraxai.routes.dependency_api import verify_token
from tests.python.fixtures.async_orch import setup_async_orchestrator


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
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def integration_client(
    event_loop: asyncio.AbstractEventLoop, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
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
