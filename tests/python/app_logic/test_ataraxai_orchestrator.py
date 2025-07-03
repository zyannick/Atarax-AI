import pytest
from unittest import mock
from pathlib import Path

import ataraxai.app_logic.ataraxai_orchestrator as orchestrator_mod


@pytest.fixture
def mock_dirs(tmp_path, monkeypatch):
    # Patch platformdirs to use tmp_path for all dirs
    monkeypatch.setattr(
        orchestrator_mod, "user_config_dir", lambda **_: str(tmp_path / "config")
    )
    monkeypatch.setattr(
        orchestrator_mod, "user_data_dir", lambda **_: str(tmp_path / "data")
    )
    monkeypatch.setattr(
        orchestrator_mod, "user_cache_dir", lambda **_: str(tmp_path / "cache")
    )
    monkeypatch.setattr(
        orchestrator_mod, "user_log_dir", lambda **_: str(tmp_path / "log")
    )
    return tmp_path


@pytest.fixture
def mock_dependencies(monkeypatch):
    # Patch all dependencies used in AtaraxAIOrchestrator
    monkeypatch.setattr(orchestrator_mod, "ArataxAILogger", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "PreferencesManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "LlamaConfigManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "WhisperConfigManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "RAGConfigManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "ChatDatabaseManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "ChatContextManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "AtaraxAIRAGManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "PromptManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "ContextManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "TaskManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "ChainRunner", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "core_ai_py", mock.Mock())
    monkeypatch.setattr(orchestrator_mod, "__version__", "1.0.0")
    # Patch init_params and initialize_core_ai_service to return mocks
    monkeypatch.setattr(
        orchestrator_mod,
        "init_params",
        mock.Mock(return_value=(mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())),
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "initialize_core_ai_service",
        mock.Mock(return_value=mock.Mock()),
    )


def test_orchestrator_initialization(mock_dirs, mock_dependencies):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    assert hasattr(orch, "logger")
    assert hasattr(orch, "app_config_dir")
    assert hasattr(orch, "app_data_dir")
    assert hasattr(orch, "cpp_service")
    assert hasattr(orch, "db_manager")
    assert hasattr(orch, "chat_context")
    assert hasattr(orch, "rag_manager")
    assert hasattr(orch, "prompt_manager")
    assert hasattr(orch, "context_manager")
    assert hasattr(orch, "task_manager")
    assert hasattr(orch, "chain_runner")


def test_run_task_chain_calls_chain_runner(mock_dirs, mock_dependencies):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    orch.chain_runner.run_chain = mock.Mock(return_value="result")
    chain_def = [{"step": 1}]
    query = "What is AI?"
    result = orch.run_task_chain(chain_def, query)
    orch.chain_runner.run_chain.assert_called_once_with(
        chain_definition=chain_def, initial_user_query=query
    )
    assert result == "result"


def test_shutdown_calls_dependencies(mock_dirs, mock_dependencies):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    orch.rag_manager.stop_file_monitoring = mock.Mock()
    orch.db_manager.close = mock.Mock()
    orch.cpp_service.shutdown = mock.Mock()
    orch.shutdown()
    orch.rag_manager.stop_file_monitoring.assert_called_once()
    orch.db_manager.close.assert_called_once()
    orch.cpp_service.shutdown.assert_called_once()


def test_first_launch_marker_file_created(mock_dirs, mock_dependencies):
    # Remove marker file if exists
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    marker_file = orch.app_setup_marker_file
    assert marker_file.exists()


def test_start_rag_monitoring_calls_rag_manager(mock_dirs, mock_dependencies):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    orch.prefs_manager.get.return_value = ["dir1", "dir2"]
    orch.rag_manager.start_file_monitoring = mock.Mock()
    orch._start_rag_monitoring()
    orch.rag_manager.start_file_monitoring.assert_called_once_with(["dir1", "dir2"])


def test_perform_first_launch_setup_creates_marker_file(mock_dirs, mock_dependencies):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    marker_file = orch.app_setup_marker_file
    # Remove marker file if it exists, then call setup
    if marker_file.exists():
        marker_file.unlink()
    orch._perform_first_launch_setup()
    assert marker_file.exists()


def test_init_user_dirs_creates_directories(tmp_path, monkeypatch, mock_dependencies):
    monkeypatch.setattr(
        orchestrator_mod, "user_config_dir", lambda **_: str(tmp_path / "config")
    )
    monkeypatch.setattr(
        orchestrator_mod, "user_data_dir", lambda **_: str(tmp_path / "data")
    )
    monkeypatch.setattr(
        orchestrator_mod, "user_cache_dir", lambda **_: str(tmp_path / "cache")
    )
    monkeypatch.setattr(
        orchestrator_mod, "user_log_dir", lambda **_: str(tmp_path / "log")
    )
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    assert orch.app_config_dir.exists()
    assert orch.app_data_dir.exists()
    assert orch.app_cache_dir.exists()
    assert orch.app_log_dir.exists()


def test_init_configs_sets_managers(mock_dirs, mock_dependencies):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    assert hasattr(orch, "prefs_manager")
    assert hasattr(orch, "llama_config_manager")
    assert hasattr(orch, "whisper_config_manager")
    assert hasattr(orch, "rag_config_manager")


def test_perform_first_launch_setup_creates_marker_file_when_not_exists(
    mock_dirs, mock_dependencies
):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    marker_file = orch.app_setup_marker_file
    if marker_file.exists():
        marker_file.unlink()
    assert not marker_file.exists()
    orch._perform_first_launch_setup()
    assert marker_file.exists()


def test_shutdown_does_not_fail_if_cpp_service_missing(mock_dirs, mock_dependencies):
    orch = orchestrator_mod.AtaraxAIOrchestrator()
    orch.rag_manager.stop_file_monitoring = mock.Mock()
    orch.db_manager.close = mock.Mock()
    orch.cpp_service = None
    # Should not raise
    orch.shutdown()
    orch.rag_manager.stop_file_monitoring.assert_called_once()
    orch.db_manager.close.assert_called_once()


def test_init_params_calls_core_ai_py_methods(monkeypatch):
    llama_params = mock.Mock()
    llama_params.model_dump.return_value = {"a": 1}
    gen_params = mock.Mock()
    gen_params.model_dump.return_value = {"b": 2}
    whisper_params = mock.Mock()
    whisper_params.model_dump.return_value = {"c": 3}
    whisper_trans_params = mock.Mock()
    whisper_trans_params.model_dump.return_value = {"d": 4}

    core_ai_py_mock = mock.Mock()
    monkeypatch.setattr(orchestrator_mod, "core_ai_py", core_ai_py_mock)
    orchestrator_mod.init_params(
        llama_params, gen_params, whisper_params, whisper_trans_params
    )
    assert core_ai_py_mock.LlamaModelParams.from_dict.called
    assert core_ai_py_mock.GenerationParams.from_dict.called
    assert core_ai_py_mock.WhisperModelParams.from_dict.called
    assert core_ai_py_mock.WhisperTranscriptionParams.from_dict.called


def test_initialize_core_ai_service_calls_methods():
    llama_params = mock.Mock()
    whisper_params = mock.Mock()
    core_ai_service_mock = mock.Mock()
    with mock.patch.object(
        orchestrator_mod.core_ai_py, "CoreAIService", return_value=core_ai_service_mock
    ):
        result = orchestrator_mod.initialize_core_ai_service(
            llama_params, whisper_params
        )
        core_ai_service_mock.initialize_llama_model.assert_called_once_with(
            llama_params
        )
        core_ai_service_mock.initialize_whisper_model.assert_called_once_with(
            whisper_params
        )
        assert result == core_ai_service_mock
