import pytest
from unittest import mock
from pathlib import Path


import ataraxai.app_logic.ataraxai_orchestrator as orchestrator_mod
from ataraxai.app_logic.ataraxai_orchestrator import AtaraxAIOrchestrator


@pytest.fixture
def mocked_orchestrator(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator_mod, "ArataxAILogger", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "PreferencesManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "LlamaConfigManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "WhisperConfigManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "RAGConfigManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "ChatDatabaseManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "ContextManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "AtaraxAIRAGManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "PromptManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "TaskManager", mock.Mock)
    monkeypatch.setattr(orchestrator_mod, "ChainRunner", mock.Mock)

    core_ai_py_mock = mock.Mock()
    monkeypatch.setattr(orchestrator_mod, "core_ai_py", core_ai_py_mock)

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

    dummy_model_file = tmp_path / "dummy_model.bin"
    dummy_model_file.touch()

    def mock_init_configs(self):
        self.prefs_manager = mock.Mock()

        self.llama_config_manager = mock.Mock()
        mock_llama_params = mock.Mock()
        mock_llama_params.model_path = str(dummy_model_file)
        self.llama_config_manager.get_llm_params.return_value = mock_llama_params
        self.llama_config_manager.get_generation_params.return_value = mock.Mock()

        self.whisper_config_manager = mock.Mock()
        mock_whisper_params = mock.Mock()
        mock_whisper_params.model = str(dummy_model_file)
        self.whisper_config_manager.get_whisper_params.return_value = (
            mock_whisper_params
        )
        self.whisper_config_manager.get_transcription_params.return_value = mock.Mock()

        self.rag_config_manager = mock.Mock()

    monkeypatch.setattr(AtaraxAIOrchestrator, "_init_configs", mock_init_configs)

    orchestrator = AtaraxAIOrchestrator()
    yield orchestrator


def test_orchestrator_initialization(mocked_orchestrator: AtaraxAIOrchestrator):
    assert hasattr(mocked_orchestrator, "logger")
    assert hasattr(mocked_orchestrator, "rag_manager")
    assert hasattr(mocked_orchestrator, "chain_runner")
    assert hasattr(mocked_orchestrator, "db_manager")


def test_run_task_chain_calls_chain_runner(mocked_orchestrator: AtaraxAIOrchestrator):
    mocked_orchestrator.chain_runner.run_chain = mock.Mock(return_value="result")
    chain_def = [{"step": 1}]
    query = "What is AI?"
    result = mocked_orchestrator.run_task_chain(chain_def, query)
    mocked_orchestrator.chain_runner.run_chain.assert_called_once_with(
        chain_definition=chain_def, initial_user_query=query
    )
    assert result == "result"


def test_shutdown_calls_dependencies(mocked_orchestrator: AtaraxAIOrchestrator):
    mocked_orchestrator.rag_manager.stop_file_monitoring = mock.Mock()
    mocked_orchestrator.db_manager.close = mock.Mock()
    mocked_orchestrator.cpp_service.shutdown = mock.Mock()
    mocked_orchestrator.shutdown()
    mocked_orchestrator.rag_manager.stop_file_monitoring.assert_called_once()
    mocked_orchestrator.db_manager.close.assert_called_once()
    mocked_orchestrator.cpp_service.shutdown.assert_called_once()


def test_start_rag_monitoring_calls_rag_manager(
    mocked_orchestrator: AtaraxAIOrchestrator,
):
    mocked_orchestrator.rag_manager.start_file_monitoring = mock.Mock()
    mocked_orchestrator.prefs_manager.get.return_value = ["dir1", "dir2"]
    mocked_orchestrator.rag_manager.start_file_monitoring = mock.Mock()
    mocked_orchestrator._start_rag_monitoring()
    mocked_orchestrator.rag_manager.start_file_monitoring.assert_called_once_with(
        ["dir1", "dir2"]
    )


def test_perform_first_launch_setup_creates_marker_file(
    mocked_orchestrator: AtaraxAIOrchestrator,
):
    marker_file = mocked_orchestrator.app_setup_marker_file

    if marker_file.exists():
        marker_file.unlink()
    assert not marker_file.exists()

    mocked_orchestrator._perform_first_launch_setup()

    assert marker_file.exists()


def test_init_user_dirs_creates_directories(mocked_orchestrator: AtaraxAIOrchestrator):
    assert mocked_orchestrator.app_config_dir.exists()
    assert mocked_orchestrator.app_data_dir.exists()


def test_init_configs_sets_managers(mocked_orchestrator: AtaraxAIOrchestrator):
    orch = mocked_orchestrator
    assert hasattr(orch, "prefs_manager")
    assert hasattr(orch, "llama_config_manager")
    assert hasattr(orch, "whisper_config_manager")
    assert hasattr(orch, "rag_config_manager")


def test_perform_first_launch_setup_creates_marker_file_when_not_exists(
    mocked_orchestrator: AtaraxAIOrchestrator,
):
    marker_file = mocked_orchestrator.app_setup_marker_file
    if marker_file.exists():
        marker_file.unlink()
    assert not marker_file.exists()
    mocked_orchestrator._perform_first_launch_setup()
    assert marker_file.exists()


def test_shutdown_does_not_fail_if_cpp_service_missing(
    mocked_orchestrator: AtaraxAIOrchestrator,
):
    mocked_orchestrator.rag_manager.stop_file_monitoring = mock.Mock()
    mocked_orchestrator.db_manager.close = mock.Mock()
    mocked_orchestrator.cpp_service = None
    mocked_orchestrator.shutdown()
    mocked_orchestrator.rag_manager.stop_file_monitoring.assert_called_once()
    mocked_orchestrator.db_manager.close.assert_called_once()


def test_init_params_calls_core_ai_py_methods(monkeypatch):
    mock_params = mock.Mock()
    mock_params.model_dump.return_value = {}
    core_ai_py_mock = mock.Mock()
    monkeypatch.setattr(orchestrator_mod, "core_ai_py", core_ai_py_mock)

    orchestrator_mod.init_params(mock_params, mock_params, mock_params, mock_params)

    assert core_ai_py_mock.LlamaModelParams.from_dict.called
    assert core_ai_py_mock.WhisperModelParams.from_dict.called


def test_initialize_core_ai_service_calls_methods(monkeypatch):

    mock_llama_params = mock.Mock()
    mock_whisper_params = mock.Mock()

    mock_service_instance = mock.Mock()

    monkeypatch.setattr(
        orchestrator_mod.core_ai_py, "CoreAIService", lambda **_: mock_service_instance
    )

    result = orchestrator_mod.initialize_core_ai_service(
        mock_llama_params, mock_whisper_params
    )

    mock_service_instance.initialize_llama_model.assert_called_once_with(
        mock_llama_params
    )
    mock_service_instance.initialize_whisper_model.assert_called_once_with(
        mock_whisper_params
    )

    assert result is mock_service_instance
