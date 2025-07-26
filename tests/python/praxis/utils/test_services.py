import pytest
from unittest import mock
from ataraxai.praxis.utils.services import Services
from ataraxai.praxis.utils.exceptions import ValidationError, ServiceInitializationError


@pytest.fixture
def mock_dependencies():
    return {
        "directories": mock.Mock(),
        "logger": mock.Mock(),
        "db_manager": mock.Mock(),
        "chat_context": mock.Mock(),
        "chat_manager": mock.Mock(),
        "config_manager": mock.Mock(),
        "app_config": mock.Mock(),
        "vault_manager": mock.Mock(),
        "models_manager": mock.Mock(),
    }


@pytest.fixture
def services(mock_dependencies):
    mock_dependencies["app_config"].database_filename = "test.db"
    mock_dependencies["app_config"].prompts_directory = "prompts"
    return Services(**mock_dependencies)


def test_initialize_success(services):
    services._init_database = mock.Mock()
    services._init_rag_manager = mock.Mock()
    services._init_prompt_engine = mock.Mock()
    services._finalize_setup = mock.Mock()
    services.logger.info = mock.Mock()

    services.initialize()

    services._init_database.assert_called_once()
    services._init_rag_manager.assert_called_once()
    services._init_prompt_engine.assert_called_once()
    services._finalize_setup.assert_called_once()
    services.logger.info.assert_any_call("Services initialized successfully")


def test_initialize_failure(services):
    services._init_database = mock.Mock(side_effect=Exception("fail"))
    services.logger.error = mock.Mock()
    with pytest.raises(Exception):
        services.initialize()
    services.logger.error.assert_called()


def test_set_core_ai_manager_sets_and_logs(services):
    core_ai_manager = mock.Mock()
    services.logger.info = mock.Mock()
    services.set_core_ai_manager(core_ai_manager)
    assert services.core_ai_manager == core_ai_manager
    services.logger.info.assert_called_with(
        "Core AI manager set for chat manager and chain runner"
    )


def test_add_watched_directory_valid(services):
    services.config_manager.add_watched_directory = mock.Mock()
    services.config_manager.get_watched_directories = mock.Mock(return_value=["dir1"])
    services.rag_manager = mock.Mock()
    services.logger.info = mock.Mock()
    with mock.patch(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_directory"
    ) as vd:
        services.add_watched_directory("dir1")
        vd.assert_called_once_with("dir1", "Directory path")
    services.config_manager.add_watched_directory.assert_called_with("dir1")
    services.rag_manager.start_file_monitoring.assert_called_with(["dir1"])
    services.logger.info.assert_called_with("Added watch directory: dir1")


def test_add_watched_directory_invalid(services):
    with mock.patch(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_directory",
        side_effect=ValueError,
    ):
        with pytest.raises(ValueError):
            services.add_watched_directory("bad_dir")


def test_run_task_chain_valid(services):
    services.core_ai_manager = mock.Mock()
    services.core_ai_manager.get_service = mock.Mock(return_value="core_service")
    services.chain_runner = mock.Mock()
    services.chain_runner.core_ai_service = None
    services.chain_runner.run_chain = mock.Mock(return_value="result")
    services.logger.info = mock.Mock()
    chain_def = [{"task_id": "t1"}]
    result = services.run_task_chain(chain_def, "query")
    assert result == "result"
    assert services.chain_runner.core_ai_service == "core_service"
    services.logger.info.assert_any_call("Chain execution completed successfully")


def test_run_task_chain_empty_chain(services):
    with pytest.raises(ValidationError):
        services.run_task_chain([], "query")


def test_run_task_chain_invalid_query(services):
    with mock.patch(
        "ataraxai.praxis.utils.input_validator.InputValidator.validate_string",
        side_effect=ValidationError,
    ):
        with pytest.raises(ValidationError):
            services.run_task_chain([{"task_id": "t1"}], "")


def test_run_task_chain_service_init_error(services):
    services.core_ai_manager = mock.Mock()
    services.core_ai_manager.get_service = mock.Mock(
        side_effect=ServiceInitializationError("fail")
    )
    services.logger.error = mock.Mock()
    with pytest.raises(ServiceInitializationError):
        services.run_task_chain([{"task_id": "t1"}], "query")
    services.logger.error.assert_called()


def test_run_task_chain_chain_runner_exception(services):
    services.core_ai_manager = mock.Mock()
    services.core_ai_manager.get_service = mock.Mock(return_value="core_service")
    services.chain_runner = mock.Mock()
    services.chain_runner.core_ai_service = None
    services.chain_runner.run_chain = mock.Mock(side_effect=Exception("fail"))
    services.logger.info = mock.Mock()
    services.logger.error = mock.Mock()
    with pytest.raises(Exception):
        services.run_task_chain([{"task_id": "t1"}], "query")
    services.logger.error.assert_called()


def test_shutdown_success(services):
    services.rag_manager = mock.Mock()
    services.db_manager = mock.Mock()
    services.core_ai_manager = mock.Mock()
    services.logger.info = mock.Mock()
    services.shutdown()
    services.rag_manager.stop_file_monitoring.assert_called_once()
    services.db_manager.close.assert_called_once()
    services.core_ai_manager.shutdown.assert_called_once()
    services.logger.info.assert_any_call("AtaraxAI shutdown completed successfully")


def test_shutdown_failure(services):
    services.rag_manager = mock.Mock()
    services.db_manager = mock.Mock()
    services.core_ai_manager = mock.Mock()
    services.rag_manager.stop_file_monitoring.side_effect = Exception("fail")
    services.logger.info = mock.Mock()
    services.logger.error = mock.Mock()
    with pytest.raises(Exception):
        services.shutdown()
    services.logger.error.assert_called()


def test_finalize_setup_manifest_invalid(services):
    services.config_manager.get_watched_directories = mock.Mock(return_value=["dir1"])
    services.rag_manager = mock.Mock()
    services.rag_manager.manifest.is_valid.return_value = False
    services._finalize_setup()
    services.rag_manager.rebuild_index_for_watches.assert_called_with(["dir1"])
    services.rag_manager.start_file_monitoring.assert_called_with(["dir1"])



def test_finalize_setup_manifest_valid(services):
    services.config_manager.get_watched_directories = mock.Mock(return_value=["dir1"])
    services.rag_manager = mock.Mock()
    services.rag_manager.manifest.is_valid.return_value = True
    services._finalize_setup()
    services.rag_manager.perform_initial_scan.assert_called_with(["dir1"])
    services.rag_manager.start_file_monitoring.assert_called_with(["dir1"])





