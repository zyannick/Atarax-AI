import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
from ataraxai.praxis.utils.services import Services
from ataraxai.praxis.utils.exceptions import ValidationError

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
    }

@pytest.fixture
def services(mock_services_dependencies):
    with patch("ataraxai.praxis.utils.services.ChatDatabaseManager") as MockDBManager, \
         patch("ataraxai.praxis.utils.services.ChatContextManager") as MockChatContext, \
         patch("ataraxai.praxis.utils.services.ChatManager") as MockChatManager, \
         patch("ataraxai.praxis.utils.services.AtaraxAIRAGManager") as MockRAGManager, \
         patch("ataraxai.praxis.utils.services.PromptManager") as MockPromptManager, \
         patch("ataraxai.praxis.utils.services.ContextManager") as MockContextManager, \
         patch("ataraxai.praxis.utils.services.TaskManager") as MockTaskManager, \
         patch("ataraxai.praxis.utils.services.ChainRunner") as MockChainRunner:
        return Services(**mock_services_dependencies)

def test_set_core_ai_manager_sets_attribute_and_logs(services):
    core_ai_manager = MagicMock()
    services.logger = MagicMock()
    services.set_core_ai_manager(core_ai_manager)
    assert services.core_ai_manager == core_ai_manager
    services.logger.info.assert_called_with("Core AI manager set for chat manager and chain runner")

def test_add_watched_directory_validates_and_adds(services):
    services.config_manager.add_watched_directory = MagicMock()
    services.config_manager.get_watched_directories = MagicMock(return_value=["/tmp"])
    services.rag_manager = MagicMock()
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_directory") as mock_validate:
        services.add_watched_directory("/tmp")
        mock_validate.assert_called_once_with("/tmp", "Directory path")
        services.config_manager.add_watched_directory.assert_called_once_with("/tmp")
        services.rag_manager.start_file_monitoring.assert_called_once_with(["/tmp"])
        services.logger.info.assert_called_with("Added watch directory: /tmp")

def test_add_watched_directory_invalid_raises(services):
    services.config_manager = MagicMock()
    services.rag_manager = MagicMock()
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_directory", side_effect=ValueError("bad dir")):
        with pytest.raises(ValueError):
            services.add_watched_directory("bad_dir")

def test_initialize_success(services):
    services._init_database = MagicMock()
    services._init_rag_manager = MagicMock()
    services._init_prompt_engine = MagicMock()
    services._finalize_setup = MagicMock()
    services.logger = MagicMock()
    services.initialize()
    services._init_database.assert_called_once()
    services._init_rag_manager.assert_called_once()
    services._init_prompt_engine.assert_called_once()
    services._finalize_setup.assert_called_once()
    services.logger.info.assert_called_with("Services initialized successfully")

def test_initialize_failure_logs_and_raises(services):
    services._init_database = MagicMock(side_effect=Exception("fail"))
    services.logger = MagicMock()
    with pytest.raises(Exception):
        services.initialize()
    services.logger.error.assert_called()

def test_run_task_chain_valid(services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain = MagicMock(return_value="result")
    services.logger = MagicMock()
    chain_def = [{"task_id": "t1"}]
    result = services.run_task_chain(chain_def, "query")
    services.logger.info.assert_any_call("Executing chain for query: 'query'")
    services.logger.info.assert_any_call("Chain execution completed successfully")
    assert result == "result"

def test_run_task_chain_empty_chain_raises(services):
    with pytest.raises(ValidationError):
        services.run_task_chain([], "query")

def test_run_task_chain_invalid_query_raises(services):
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_string", side_effect=ValidationError("bad")):
        with pytest.raises(ValidationError):
            services.run_task_chain([{"task_id": "t1"}], "")

def test_run_task_chain_exception_logs_and_raises(services):
    services.chain_runner = MagicMock()
    services.chain_runner.run_chain.side_effect = Exception("fail")
    services.logger = MagicMock()
    with patch("ataraxai.praxis.utils.services.InputValidator.validate_string"):
        with pytest.raises(Exception):
            services.run_task_chain([{"task_id": "t1"}], "query")
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


def test_finalize_setup_rebuilds_index_if_manifest_invalid(services):
    services.config_manager.get_watched_directories = MagicMock(return_value=["/tmp"])
    services.rag_manager = MagicMock()
    services.rag_manager.manifest.is_valid.return_value = False
    services._finalize_setup()
    services.rag_manager.rebuild_index_for_watches.assert_called_once_with(["/tmp"])
    services.rag_manager.start_file_monitoring.assert_called_once_with(["/tmp"])

def test_finalize_setup_performs_initial_scan_if_manifest_valid(services):
    services.config_manager.get_watched_directories = MagicMock(return_value=["/tmp"])
    services.rag_manager = MagicMock()
    services.rag_manager.manifest.is_valid.return_value = True
    services._finalize_setup()
    services.rag_manager.perform_initial_scan.assert_called_once_with(["/tmp"])
    services.rag_manager.start_file_monitoring.assert_called_once_with(["/tmp"])


def test_init_database_sets_up_managers_and_logs(services, tmp_path):
    services.directories.data = tmp_path / "data"
    services.app_config.database_filename = "db.sqlite"
    with patch("ataraxai.praxis.utils.services.ChatDatabaseManager") as MockDBManager, \
            patch("ataraxai.praxis.utils.services.ChatContextManager") as MockChatContext, \
            patch("ataraxai.praxis.utils.services.ChatManager") as MockChatManager:
        services.logger = MagicMock()
        services._init_database()
        MockDBManager.assert_called_once_with(db_path= tmp_path / "data/db.sqlite")
        MockChatContext.assert_called_once()
        MockChatManager.assert_called_once()
        services.logger.info.assert_called_with("Database initialized successfully")

def test_init_rag_manager_creates_rag_manager_and_logs(services, tmp_path):
    services.config_manager.rag_config_manager = MagicMock()
    services.directories.data = tmp_path / "data"
    with patch("ataraxai.praxis.utils.services.AtaraxAIRAGManager") as MockRAGManager:
        services.logger = MagicMock()
        services._init_rag_manager()
        MockRAGManager.assert_called_once_with(
            rag_config_manager=services.config_manager.rag_config_manager,
            app_data_root_path=tmp_path / "data",
            core_ai_service=None,
        )
        services.logger.info.assert_called_with("RAG manager initialized successfully")

def test_init_prompt_engine_creates_managers_and_logs(services, tmp_path):
    prompts_dir = tmp_path / "prompts"
    services.app_config.prompts_directory = str(prompts_dir)
    services.config_manager.rag_config_manager.get_config.return_value.model_dump.return_value = {"foo": "bar"}
    services.rag_manager = MagicMock()
    with patch("ataraxai.praxis.utils.services.PromptManager") as MockPromptManager, \
            patch("ataraxai.praxis.utils.services.ContextManager") as MockContextManager, \
            patch("ataraxai.praxis.utils.services.TaskManager") as MockTaskManager, \
            patch("ataraxai.praxis.utils.services.ChainRunner") as MockChainRunner:
        services.core_ai_service_manager = MagicMock()
        services.chat_context = MagicMock()
        services.logger = MagicMock()
        services._init_prompt_engine()
        MockPromptManager.assert_called_once()
        MockContextManager.assert_called_once()
        MockTaskManager.assert_called_once()
        MockChainRunner.assert_called_once()
        services.logger.info.assert_called_with("Prompt engine initialized successfully")
