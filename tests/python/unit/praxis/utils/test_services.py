import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
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

def test_set_core_ai_manager_sets_attribute_and_logs(services):
    core_ai_manager = MagicMock()
    services.logger = MagicMock()
    services.set_core_ai_manager(core_ai_manager)
    assert services.core_ai_manager == core_ai_manager
    services.logger.info.assert_called_with("Core AI manager set for chat manager and chain runner")

def test_add_watched_directory_validates_and_adds(services):
    services.config_manager.add_watched_directory = MagicMock()
    services.config_manager.get_watched_directories = MagicMock(return_value=["dir1", "dir2"])
    services.rag_manager = MagicMock()
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_directory") as mock_validate:
        services.add_watched_directory("dir3")
        mock_validate.assert_called_once_with("dir3", "Directory path")
        services.config_manager.add_watched_directory.assert_called_once_with("dir3")
        services.rag_manager.start_file_monitoring.assert_called_once_with(["dir1", "dir2"])
        services.logger.info.assert_called_with("Added watch directory: dir3")

def test_add_watched_directory_invalid_raises(services):
    services.config_manager.add_watched_directory = MagicMock()
    services.rag_manager = MagicMock()
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_directory", side_effect=ValueError("bad dir")):
        with pytest.raises(ValueError):
            services.add_watched_directory("bad_dir")

def test_init_database_sets_up_managers(services):
    services.directories.data = Path("/tmp")
    services.app_config.database_filename = "test.db"
    services.vault_manager = MagicMock()
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.ChatDatabaseManager") as mock_db_mgr, \
         patch("ataraxai.praxis.utils.services.ChatContextManager") as mock_ctx_mgr, \
         patch("ataraxai.praxis.utils.services.ChatManager") as mock_chat_mgr:
        services._init_database()
        mock_db_mgr.assert_called_once()
        mock_ctx_mgr.assert_called_once()
        mock_chat_mgr.assert_called_once()
        services.logger.info.assert_called_with("Database initialized successfully")

def test_init_rag_manager_sets_rag_manager(services):
    services.config_manager.rag_config_manager = MagicMock()
    services.directories.data = MagicMock()
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.AtaraxAIRAGManager") as mock_rag_mgr:
        services._init_rag_manager()
        mock_rag_mgr.assert_called_once()
        services.logger.info.assert_called_with("RAG manager initialized successfully")

def test_init_prompt_engine_creates_managers(tmp_path, services):
    services.app_config.prompts_directory = str(tmp_path / "prompts")
    services.config_manager.rag_config_manager = MagicMock()
    services.config_manager.rag_config_manager.get_config.return_value.model_dump.return_value = {}
    services.rag_manager = MagicMock()
    services.chat_context = MagicMock()
    services.core_ai_service_manager = MagicMock()
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.PromptManager") as mock_prompt_mgr, \
         patch("ataraxai.praxis.utils.services.ContextManager") as mock_ctx_mgr, \
         patch("ataraxai.praxis.utils.services.TaskManager") as mock_task_mgr, \
         patch("ataraxai.praxis.utils.services.ChainRunner") as mock_chain_runner:
        services._init_prompt_engine()
        mock_prompt_mgr.assert_called_once()
        mock_ctx_mgr.assert_called_once()
        mock_task_mgr.assert_called_once()
        mock_chain_runner.assert_called_once()
        services.logger.info.assert_called_with("Prompt engine initialized successfully")

@pytest.mark.asyncio
async def test_run_task_chain_valid(services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain = mock.AsyncMock(return_value="result")
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_string"):
        result = await services.run_task_chain([{"task": "t"}], "query")
        assert result == "result"
        services.logger.info.assert_any_call("Executing chain for query: 'query'")
        services.logger.info.assert_any_call("Chain execution completed successfully")

@pytest.mark.asyncio
async def test_run_task_chain_empty_chain_raises(services):
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_string"):
        with pytest.raises(ValidationError):
            await services.run_task_chain([], "query")

@pytest.mark.asyncio
async def test_run_task_chain_exception_logs_and_raises(services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain = mock.AsyncMock(side_effect=Exception("fail"))
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_string"):
        with pytest.raises(Exception):
            await services.run_task_chain([{"task": "t"}], "query")
        services.logger.error.assert_called()

def test_shutdown_calls_all_and_logs(services):
    services.rag_manager = MagicMock()
    services.db_manager = MagicMock()
    services.core_ai_manager = MagicMock()
    services.logger = MagicMock()
    services.shutdown()
    services.rag_manager.stop_file_monitoring.assert_called_once()
    services.db_manager.close.assert_called_once()
    services.core_ai_manager.shutdown.assert_called_once()
    services.logger.info.assert_any_call("Shutting down AtaraxAI...")
    services.logger.info.assert_any_call("AtaraxAI shutdown completed successfully")

def test_shutdown_handles_exceptions_and_logs(services):
    services.rag_manager = MagicMock()
    services.db_manager = MagicMock(side_effect=Exception("fail"))
    services.core_ai_manager = MagicMock()
    services.logger = MagicMock()
    with pytest.raises(Exception):
        services.shutdown()
    services.logger.error.assert_called()

def test_finalize_setup_rebuilds_index_if_manifest_invalid(services):
    services.config_manager.get_watched_directories = MagicMock(return_value=["dir1"])
    services.rag_manager = MagicMock()
    services.rag_manager.manifest.is_valid.return_value = False
    services.rag_manager.rag_store = MagicMock()
    services._finalize_setup()
    services.rag_manager.rebuild_index_for_watches.assert_called_once_with(["dir1"])
    services.rag_manager.start_file_monitoring.assert_called_once_with(["dir1"])

def test_finalize_setup_performs_initial_scan_if_manifest_valid(services):
    services.config_manager.get_watched_directories = MagicMock(return_value=["dir1"])
    services.rag_manager = MagicMock()
    services.rag_manager.manifest.is_valid.return_value = True
    services.rag_manager.rag_store = MagicMock()
    services._finalize_setup()
    services.rag_manager.perform_initial_scan.assert_called_once_with(["dir1"])
    services.rag_manager.start_file_monitoring.assert_called_once_with(["dir1"])
