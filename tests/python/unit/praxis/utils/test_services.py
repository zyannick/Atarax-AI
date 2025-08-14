import pytest
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch
from ataraxai.praxis.utils.services import Services
from ataraxai.praxis.utils.exceptions import ValidationError
from pathlib import Path


@pytest.fixture
def mock_dependencies():
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
    }


@pytest.fixture
def services(mock_dependencies):
    mock_dependencies["app_config"].database_filename = "test.db"
    mock_dependencies["app_config"].prompts_directory = "prompts"
    mock_dependencies["directories"].data = mock.Mock()
    return Services(**mock_dependencies)


def test_init_database_sets_up_managers(services, mock_dependencies):
    mock_dependencies["directories"].data = Path("/tmp")
    mock_dependencies["app_config"].database_filename = "test.db"
    mock_dependencies["vault_manager"] = MagicMock()

    services._init_database()
    assert hasattr(services, "db_manager")
    assert hasattr(services, "chat_context")
    assert hasattr(services, "chat_manager")
    services.logger.info.assert_called_with("Database initialized successfully")


def test_init_rag_manager_sets_rag_manager(services, mock_dependencies):
    mock_dependencies["config_manager"].rag_config_manager = MagicMock()
    mock_dependencies["config_manager"].rag_config_manager.config.rag_embedder_model = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    mock_dependencies["directories"].data = Path("/tmp")

    services._init_rag_manager()

    assert hasattr(services, "rag_manager")
    services.logger.info.assert_called_with("RAG manager initialized successfully")


@pytest.mark.asyncio
async def test_initialize_success(monkeypatch, services):
    monkeypatch.setattr(services, "_init_database", lambda: None)
    monkeypatch.setattr(services, "_init_rag_manager", lambda: None)
    monkeypatch.setattr(services, "_init_prompt_engine", lambda: None)
    monkeypatch.setattr(services, "_finalize_setup", AsyncMock())

    await services.initialize()
    services.logger.info.assert_called_with("Services initialized successfully")


@pytest.mark.asyncio
async def test_initialize_failure(monkeypatch, services):
    monkeypatch.setattr(
        services, "_init_database", lambda: (_ for _ in ()).throw(Exception("fail"))
    )
    with pytest.raises(Exception):
        await services.initialize()
    services.logger.error.assert_called()


def test_set_core_ai_manager_sets_manager_and_logs(services):
    core_ai_manager = MagicMock()
    services.set_core_ai_manager(core_ai_manager)
    assert services.core_ai_manager == core_ai_manager
    services.logger.info.assert_called_with(
        "Core AI manager set for chat manager and chain runner"
    )


@pytest.mark.asyncio
async def test_add_watched_directory_valid(monkeypatch, services):
    services.config_manager.add_watched_directory = MagicMock()
    services.rag_manager = MagicMock()
    services.rag_manager.start = AsyncMock()
    monkeypatch.setattr(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_directory",
        lambda d, n: None,
    )

    await services.add_watched_directory("/tmp/test")
    services.config_manager.add_watched_directory.assert_called_with("/tmp/test")
    services.rag_manager.start.assert_awaited()
    services.logger.info.assert_called_with("Added watch directory: /tmp/test")


@pytest.mark.asyncio
async def test_add_watched_directory_invalid(monkeypatch, services):
    monkeypatch.setattr(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_directory",
        lambda d, n: (_ for _ in ()).throw(ValueError("bad dir")),
    )
    with pytest.raises(ValueError):
        await services.add_watched_directory("/bad/dir")


@pytest.mark.asyncio
async def test_run_task_chain_success(monkeypatch, services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain = AsyncMock(return_value="result")
    monkeypatch.setattr(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_string",
        lambda s, n: None,
    )

    result = await services.run_task_chain([{"step": 1}], "query")
    assert result == "result"
    services.logger.info.assert_any_call("Executing chain for query: 'query'")
    services.logger.info.assert_any_call("Chain execution completed successfully")


@pytest.mark.asyncio
async def test_run_task_chain_empty_definition(monkeypatch, services):
    monkeypatch.setattr(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_string",
        lambda s, n: None,
    )
    with pytest.raises(ValidationError):
        await services.run_task_chain([], "query")


@pytest.mark.asyncio
async def test_run_task_chain_exception(monkeypatch, services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain = AsyncMock(side_effect=Exception("fail"))
    monkeypatch.setattr(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_string",
        lambda s, n: None,
    )
    with pytest.raises(Exception):
        await services.run_task_chain([{"step": 1}], "query")
    services.logger.error.assert_called()


@pytest.mark.asyncio
async def test_shutdown_success(monkeypatch, services):
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
async def test_shutdown_exception(monkeypatch, services):
    services.rag_manager = MagicMock()
    services.rag_manager.stop = AsyncMock(side_effect=Exception("fail"))
    with pytest.raises(Exception):
        await services.shutdown()
    services.logger.error.assert_called()


@pytest.mark.asyncio
async def test_finalize_setup_calls_rag_manager_start(services):
    services.rag_manager = MagicMock()
    services.rag_manager.start = AsyncMock()
    await services._finalize_setup()
    services.rag_manager.start.assert_awaited()
