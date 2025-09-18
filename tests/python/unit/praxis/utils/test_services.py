from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ataraxai.praxis.utils.exceptions import ValidationError
from ataraxai.praxis.utils.services import Services


@pytest.fixture
def mock_services_dependencies():
    return {
        "directories": MagicMock(),
        "logger": MagicMock(),
        "db_manager": MagicMock(),
        "chat_context": MagicMock(),
        "chat_manager": MagicMock(),
        "config_manager": MagicMock(),
        "app_config": MagicMock(),
        "vault_manager": MagicMock(),
        "models_manager": MagicMock(),
        "core_ai_service_manager": MagicMock(),
        "background_task_manager": MagicMock(),
    }


@pytest.fixture
def services(mock_services_dependencies: dict):
    mock_services_dependencies["app_config"].database_filename = "test.db"
    mock_services_dependencies["app_config"].prompts_directory = "prompts"
    mock_services_dependencies["directories"].data = Path("/tmp/data")
    mock_services_dependencies["config_manager"].rag_config_manager = MagicMock()
    mock_services_dependencies[
        "config_manager"
    ].rag_config_manager.get_config.return_value.model_dump.return_value = {}
    return Services(**mock_services_dependencies)


def test_init_database_sets_db_manager_and_logs(services: Services):
    services.directories.data = Path("/tmp/data")
    services.app_config.database_filename = "test.db"
    services.vault_manager = MagicMock()
    services.logger = MagicMock()
    services._init_database()
    assert isinstance(services.db_manager, type(services.db_manager))
    services.logger.info.assert_called_with("Database initialized successfully")


def test_init_prompt_engine_creates_managers_and_logs(
    services: Services, tmp_path: Path
):
    services.app_config.prompts_directory = str(tmp_path / "prompts")
    services.rag_manager = MagicMock()
    services.core_ai_service_manager = MagicMock()
    services.chat_context = MagicMock()
    services.logger = MagicMock()

    mock_rag_config = MagicMock()
    mock_rag_config.config.rag_embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
    services.config_manager.rag_config_manager = mock_rag_config

    with patch(
        "ataraxai.praxis.utils.services.PromptManager"
    ) as mock_prompt_cls, patch(
        "ataraxai.praxis.utils.services.ContextManager"
    ) as mock_context_cls, patch(
        "ataraxai.praxis.utils.services.ChainTaskManager"
    ) as mock_task_cls, patch(
        "ataraxai.praxis.utils.services.ChainRunner"
    ) as mock_chain_runner_cls:

        services._init_prompt_engine()

        mock_prompt_cls.assert_called_once()
        mock_context_cls.assert_called_once()
        mock_task_cls.assert_called_once()
        mock_chain_runner_cls.assert_called_once()

        services.logger.info.assert_called_with(
            "Prompt engine initialized successfully"
        )


@pytest.mark.asyncio
async def test_initialize_success(services: Services):
    services._init_database = MagicMock()
    services._init_rag_manager = MagicMock()
    services._init_prompt_engine = MagicMock()
    services._finalize_setup = AsyncMock()
    services.logger = MagicMock()
    await services.initialize()
    services._init_database.assert_called_once()
    services._init_rag_manager.assert_called_once()
    services._init_prompt_engine.assert_called_once()
    services._finalize_setup.assert_awaited_once()
    services.logger.info.assert_called_with("Services initialized successfully")


@pytest.mark.asyncio
async def test_initialize_failure_logs_and_raises(services: Services):
    services._init_database = MagicMock(side_effect=Exception("fail"))
    services.logger = MagicMock()
    with pytest.raises(Exception):
        await services.initialize()
    services.logger.error.assert_called()


def test_set_core_ai_manager_sets_and_logs(services: Services):
    core_ai_manager = MagicMock()
    services.logger = MagicMock()
    services.set_core_ai_manager(core_ai_manager)
    assert services.core_ai_manager == core_ai_manager
    services.logger.info.assert_called_with(
        "Core AI manager set for chat manager and chain runner"
    )


@pytest.mark.asyncio
async def test_add_watched_directory_valid(services: Services):
    services.config_manager.add_watched_directory = MagicMock()
    services.rag_manager = MagicMock()
    services.rag_manager.start = AsyncMock()
    services.logger = MagicMock()
    with mock.patch(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_directory"
    ) as validate_dir:
        await services.add_watched_directory("/tmp/testdir")
        validate_dir.assert_called_with("/tmp/testdir", "Directory path")
        services.config_manager.add_watched_directory.assert_called_with("/tmp/testdir")
        services.rag_manager.start.assert_awaited_once()
        services.logger.info.assert_called_with("Added watch directory: /tmp/testdir")


@pytest.mark.asyncio
async def test_add_watched_directory_invalid(services: Services):
    with mock.patch(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_directory",
        side_effect=ValueError("bad dir"),
    ):
        with pytest.raises(ValueError):
            await services.add_watched_directory("bad_dir")


@pytest.mark.asyncio
async def test_run_task_chain_success(services: Services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain = AsyncMock(return_value="result")
    services.logger = MagicMock()
    with mock.patch(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_string"
    ) as validate_str:
        result = await services.run_task_chain([{"step": 1}], "query")
        validate_str.assert_called_with("query", "Initial user query")
        services.logger.info.assert_any_call("Executing chain for query: 'query'")
        services.logger.info.assert_any_call("Chain execution completed successfully")
        assert result == "result"


@pytest.mark.asyncio
async def test_run_task_chain_empty_definition_raises(services: Services):
    with pytest.raises(ValidationError):
        await services.run_task_chain([], "query")


@pytest.mark.asyncio
async def test_run_task_chain_exception_logs_and_raises(services: Services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain = AsyncMock(side_effect=Exception("fail"))
    services.logger = MagicMock()
    with mock.patch(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_string"
    ):
        with pytest.raises(Exception):
            await services.run_task_chain([{"step": 1}], "query")
    services.logger.error.assert_called()


@pytest.mark.asyncio
async def test_shutdown_success(services: Services):
    services.logger = MagicMock()
    services.rag_manager = MagicMock()
    services.rag_manager.stop = AsyncMock()
    services.db_manager = MagicMock()
    services.db_manager.close = MagicMock()
    services.core_ai_service_manager = MagicMock()
    services.core_ai_service_manager.shutdown = MagicMock()
    await services.shutdown()
    services.logger.info.assert_any_call("Shutting down AtaraxAI...")
    services.logger.info.assert_any_call("AtaraxAI shutdown completed successfully")


@pytest.mark.asyncio
async def test_shutdown_exception_logs_and_raises(services: Services):
    services.logger = MagicMock()
    services.rag_manager = MagicMock()
    services.rag_manager.stop = AsyncMock(side_effect=Exception("fail"))
    with pytest.raises(Exception):
        await services.shutdown()
    services.logger.error.assert_called()


@pytest.mark.asyncio
async def test_finalize_setup_calls_rag_manager_start(services: Services):
    services.rag_manager = MagicMock()
    services.rag_manager.start = AsyncMock()
    await services._finalize_setup()
    services.rag_manager.start.assert_awaited_once()
