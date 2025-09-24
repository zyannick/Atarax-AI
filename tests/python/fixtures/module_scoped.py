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

