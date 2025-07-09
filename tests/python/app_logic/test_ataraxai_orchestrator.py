from typing import Any, Dict
import pytest
from unittest import mock
from pathlib import Path
import uuid

import ataraxai.app_logic.ataraxai_orchestrator as orchestrator_mod
from ataraxai.app_logic.ataraxai_orchestrator import AtaraxAIOrchestrator
import importlib
import sys
from _pytest.monkeypatch import MonkeyPatch
from pathlib import Path


def test_appdirectories_create_default_creates_dirs(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
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
        orchestrator_mod, "user_log_dir", lambda **_: str(tmp_path / "logs")
    )

    dirs = orchestrator_mod.AppDirectories.create_default()
    assert dirs.config.exists()
    assert dirs.data.exists()
    assert dirs.cache.exists()
    assert dirs.logs.exists()
    assert dirs.config.is_dir()
    assert dirs.data.is_dir()
    assert dirs.cache.is_dir()
    assert dirs.logs.is_dir()


def test_appdirectories_create_directories(tmp_path: Path):
    dirs = orchestrator_mod.AppDirectories(
        config=tmp_path / "c",
        data=tmp_path / "d",
        cache=tmp_path / "ca",
        logs=tmp_path / "l",
    )
    dirs.create_directories()
    assert (tmp_path / "c").exists()
    assert (tmp_path / "d").exists()
    assert (tmp_path / "ca").exists()
    assert (tmp_path / "l").exists()


def test_appconfig_get_setup_marker_filename():
    config = orchestrator_mod.AppConfig()
    version = "1.2.3"
    filename = config.get_setup_marker_filename(version)
    assert version in filename
    assert filename.startswith(".")


def test_inputvalidator_validate_uuid_and_string():
    valid_uuid = uuid.uuid4()
    orchestrator_mod.InputValidator.validate_uuid(valid_uuid, "test")
    orchestrator_mod.InputValidator.validate_string("abc", "test")
    with pytest.raises(orchestrator_mod.ValidationError):
        orchestrator_mod.InputValidator.validate_uuid(None, "test")
    with pytest.raises(orchestrator_mod.ValidationError):
        orchestrator_mod.InputValidator.validate_string("", "test")


def test_inputvalidator_validate_path_and_directory(tmp_path: Path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    orchestrator_mod.InputValidator.validate_path(
        str(file_path), "file", must_exist=True
    )
    orchestrator_mod.InputValidator.validate_directory(str(dir_path), "dir")
    with pytest.raises(orchestrator_mod.ValidationError):
        orchestrator_mod.InputValidator.validate_path(None, "file")
    with pytest.raises(orchestrator_mod.ValidationError):
        orchestrator_mod.InputValidator.validate_path(
            str(tmp_path / "missing"), "file", must_exist=True
        )
    with pytest.raises(orchestrator_mod.ValidationError):
        orchestrator_mod.InputValidator.validate_directory(
            str(tmp_path / "missing"), "dir"
        )


def test_configuration_manager_init_success(monkeypatch: MonkeyPatch):
    mock_logger = mock.Mock()
    mock_preferences = mock.Mock()
    mock_llama = mock.Mock()
    mock_whisper = mock.Mock()
    mock_rag = mock.Mock()

    monkeypatch.setattr(
        orchestrator_mod, "PreferencesManager", lambda config_path: mock_preferences
    )
    monkeypatch.setattr(
        orchestrator_mod, "LlamaConfigManager", lambda config_path: mock_llama
    )
    monkeypatch.setattr(
        orchestrator_mod, "WhisperConfigManager", lambda config_path: mock_whisper
    )
    monkeypatch.setattr(
        orchestrator_mod, "RAGConfigManager", lambda config_path: mock_rag
    )

    config_dir = Path("/tmp/fake_config")
    mgr = orchestrator_mod.ConfigurationManager(config_dir, mock_logger)
    assert mgr.preferences is mock_preferences
    assert mgr.llama_config is mock_llama
    assert mgr.whisper_config is mock_whisper
    assert mgr.rag_config is mock_rag
    mock_logger.info.assert_called_with(
        "Configuration managers initialized successfully"
    )


def test_configuration_manager_init_failure(monkeypatch: MonkeyPatch):
    mock_logger = mock.Mock()

    def fail_init(config_path):
        raise Exception("fail")

    monkeypatch.setattr(orchestrator_mod, "PreferencesManager", fail_init)
    config_dir = Path("/tmp/fake_config")
    with pytest.raises(orchestrator_mod.ServiceInitializationError):
        orchestrator_mod.ConfigurationManager(config_dir, mock_logger)
    mock_logger.error.assert_called()


def test_configuration_manager_get_watched_directories(monkeypatch: MonkeyPatch):
    mock_logger = mock.Mock()
    mock_rag_config = mock.Mock()
    mock_rag_config.get_config.return_value = mock.Mock(
        rag_watched_directories=["/a", "/b"]
    )
    monkeypatch.setattr(
        orchestrator_mod, "PreferencesManager", lambda config_path: mock.Mock()
    )
    monkeypatch.setattr(
        orchestrator_mod, "LlamaConfigManager", lambda config_path: mock.Mock()
    )
    monkeypatch.setattr(
        orchestrator_mod, "WhisperConfigManager", lambda config_path: mock.Mock()
    )
    monkeypatch.setattr(
        orchestrator_mod, "RAGConfigManager", lambda config_path: mock_rag_config
    )
    mgr = orchestrator_mod.ConfigurationManager(Path("/tmp/fake"), mock_logger)
    assert mgr.get_watched_directories() == ["/a", "/b"]


def test_configuration_manager_add_watched_directory(monkeypatch: MonkeyPatch):
    mock_logger = mock.Mock()
    watched_dirs = ["/a"]
    config_obj = mock.Mock(rag_watched_directories=watched_dirs)
    mock_rag_config = mock.Mock()
    mock_rag_config.get_config.return_value = config_obj
    mock_rag_config.set = mock.Mock()
    monkeypatch.setattr(
        orchestrator_mod, "PreferencesManager", lambda config_path: mock.Mock()
    )
    monkeypatch.setattr(
        orchestrator_mod, "LlamaConfigManager", lambda config_path: mock.Mock()
    )
    monkeypatch.setattr(
        orchestrator_mod, "WhisperConfigManager", lambda config_path: mock.Mock()
    )
    monkeypatch.setattr(
        orchestrator_mod, "RAGConfigManager", lambda config_path: mock_rag_config
    )
    mgr = orchestrator_mod.ConfigurationManager(Path("/tmp/fake"), mock_logger)
    mgr.add_watched_directory("/b")
    mock_rag_config.set.assert_called_with("rag_watched_directories", ["/a", "/b"])
    mock_rag_config.set.reset_mock()
    mgr.add_watched_directory("/a")
    mock_rag_config.set.assert_not_called()


def make_core_ai_manager(
    monkeypatch: MonkeyPatch,
    llama_params: Dict[str, Any] = {},
    whisper_params: Dict[str, Any] = {},
    llama_exists: bool = True,
    whisper_exists: bool = True,
):
    mock_logger = mock.Mock()
    mock_config_manager = mock.Mock()
    mock_llama_config = mock.Mock()
    mock_whisper_config = mock.Mock()
    mock_llama_params = llama_params or mock.Mock(model_path="/llama/model")
    mock_whisper_params = whisper_params or mock.Mock(model="/whisper/model")
    mock_llama_config.get_llama_cpp_params.return_value = mock_llama_params
    mock_llama_config.get_generation_params.return_value = mock.Mock(
        model_dump=lambda: {}
    )
    mock_whisper_config.get_whisper_params.return_value = mock_whisper_params
    mock_whisper_config.get_transcription_params.return_value = mock.Mock(
        model_dump=lambda: {}
    )
    mock_config_manager.llama_config = mock_llama_config
    mock_config_manager.whisper_config = mock_whisper_config

    monkeypatch.setattr(
        "ataraxai.app_logic.ataraxai_orchestrator.core_ai_py", mock.Mock()
    )
    monkeypatch.setattr(
        "ataraxai.app_logic.ataraxai_orchestrator.Path.exists",
        lambda self: {
            Path("/llama/model"): llama_exists,
            Path("/whisper/model"): whisper_exists,
        }.get(self, True),
    )
    return orchestrator_mod.CoreAIServiceManager(mock_config_manager, mock_logger)


def test_core_ai_service_manager_initialize_success(monkeypatch : MonkeyPatch):
    manager = make_core_ai_manager(monkeypatch)
    manager._create_core_ai_service = mock.Mock(return_value="service")
    manager.initialize()
    assert manager.status == orchestrator_mod.ServiceStatus.INITIALIZED
    assert manager.service == "service"


def test_core_ai_service_manager_initialize_failure(monkeypatch : MonkeyPatch):
    manager = make_core_ai_manager(monkeypatch)
    manager._validate_model_paths = mock.Mock(
        side_effect=orchestrator_mod.ValidationError("fail")
    )
    with pytest.raises(orchestrator_mod.ServiceInitializationError):
        manager.initialize()
    assert manager.status == orchestrator_mod.ServiceStatus.FAILED


def test_core_ai_service_manager_get_service(monkeypatch : MonkeyPatch ):
    manager = make_core_ai_manager(monkeypatch)
    manager._create_core_ai_service = mock.Mock(return_value="service")
    manager.status = orchestrator_mod.ServiceStatus.NOT_INITIALIZED
    assert manager.get_service() == "service"
    manager.status = orchestrator_mod.ServiceStatus.FAILED
    with pytest.raises(orchestrator_mod.ServiceInitializationError):
        manager.get_service()


def test_core_ai_service_manager_is_configured(monkeypatch : MonkeyPatch):
    manager = make_core_ai_manager(monkeypatch)
    assert manager.is_configured() is True
    manager._validate_model_paths = mock.Mock(
        side_effect=orchestrator_mod.ValidationError("fail")
    )
    assert manager.is_configured() is False


def test_core_ai_service_manager_get_configuration_status(monkeypatch : MonkeyPatch):
    llama_params = mock.Mock(model_path="/llama/model")
    whisper_params = mock.Mock(model="/whisper/model")
    manager = make_core_ai_manager(monkeypatch, llama_params, whisper_params)
    status = manager.get_configuration_status()
    assert status["llama_configured"] is True
    assert status["whisper_configured"] is True
    assert status["llama_model_path"] == "/llama/model"
    assert status["whisper_model_path"] == "/whisper/model"
    assert status["llama_path_exists"] is True
    assert status["whisper_path_exists"] is True
    assert status["initialization_status"] == manager.status.value


def test_core_ai_service_manager_validate_model_paths_success(monkeypatch : MonkeyPatch):
    manager = make_core_ai_manager(monkeypatch)
    manager._validate_model_paths()


def test_core_ai_service_manager_validate_model_paths_failures(monkeypatch : MonkeyPatch):
    llama_params = mock.Mock(model_path=None)
    manager = make_core_ai_manager(monkeypatch, llama_params=llama_params)
    with pytest.raises(orchestrator_mod.ValidationError):
        manager._validate_model_paths()
    llama_params = mock.Mock(model_path="/llama/model")
    manager = make_core_ai_manager(
        monkeypatch, llama_params=llama_params, llama_exists=False
    )
    with pytest.raises(orchestrator_mod.ValidationError):
        manager._validate_model_paths()
    whisper_params = mock.Mock(model=None)
    manager = make_core_ai_manager(monkeypatch, whisper_params=whisper_params)
    with pytest.raises(orchestrator_mod.ValidationError):
        manager._validate_model_paths()
    whisper_params = mock.Mock(model="/whisper/model")
    manager = make_core_ai_manager(
        monkeypatch, whisper_params=whisper_params, whisper_exists=False
    )
    with pytest.raises(orchestrator_mod.ValidationError):
        manager._validate_model_paths()


def test_core_ai_service_manager_shutdown(monkeypatch : MonkeyPatch):
    manager = make_core_ai_manager(monkeypatch)
    manager.service = mock.Mock()
    manager.status = orchestrator_mod.ServiceStatus.INITIALIZED
    manager.shutdown()
    assert manager.service is None
    assert manager.status == orchestrator_mod.ServiceStatus.NOT_INITIALIZED


def test_core_ai_service_manager_is_initialized_property(monkeypatch : MonkeyPatch):
    manager = make_core_ai_manager(monkeypatch)
    manager.status = orchestrator_mod.ServiceStatus.INITIALIZED
    assert manager.is_initialized is True
    manager.status = orchestrator_mod.ServiceStatus.NOT_INITIALIZED
    assert manager.is_initialized is False


def make_chat_manager(monkeypatch : MonkeyPatch):
    mock_db_manager = mock.Mock()
    mock_logger = mock.Mock()
    return (
        orchestrator_mod.ChatManager(mock_db_manager, mock_logger),
        mock_db_manager,
        mock_logger,
    )


def test_chatmanager_create_project_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.create_project.return_value = mock.Mock()
    with mock.patch.object(
        orchestrator_mod.ProjectResponse, "model_validate", return_value="resp"
    ) as mv:
        result = chat_manager.create_project("proj", "desc")
        assert result == "resp"
        db_manager.create_project.assert_called_with(name="proj", description="desc")
        logger.info.assert_called()


def test_chatmanager_create_project_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.create_project("", "desc")


def test_chatmanager_create_project_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.create_project.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.create_project("proj", "desc")
    logger.error.assert_called()


def test_chatmanager_get_project_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.get_project.return_value = mock.Mock()
    with mock.patch.object(
        orchestrator_mod.ProjectResponse, "model_validate", return_value="resp"
    ) as mv:
        pid = uuid.uuid4()
        result = chat_manager.get_project(pid)
        assert result == "resp"
        db_manager.get_project.assert_called_with(pid)


def test_chatmanager_get_project_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.get_project(None)


def test_chatmanager_get_project_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.get_project.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.get_project(uuid.uuid4())
    logger.error.assert_called()


def test_chatmanager_list_projects_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.list_projects.return_value = [mock.Mock(), mock.Mock()]
    with mock.patch.object(
        orchestrator_mod.ProjectResponse, "model_validate", side_effect=lambda x: x
    ):
        result = chat_manager.list_projects()
        assert result == db_manager.list_projects.return_value


def test_chatmanager_list_projects_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.list_projects.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.list_projects()
    logger.error.assert_called()


def test_chatmanager_delete_project_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_project.return_value = True
    pid = uuid.uuid4()
    result = chat_manager.delete_project(pid)
    assert result is True
    logger.info.assert_called()


def test_chatmanager_delete_project_not_found(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_project.return_value = False
    pid = uuid.uuid4()
    result = chat_manager.delete_project(pid)
    assert result is False


def test_chatmanager_delete_project_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.delete_project(None)


def test_chatmanager_delete_project_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_project.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.delete_project(uuid.uuid4())
    logger.error.assert_called()


def test_chatmanager_create_session_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.create_session.return_value = mock.Mock()
    with mock.patch.object(
        orchestrator_mod.ChatSessionResponse, "model_validate", return_value="resp"
    ):
        pid = uuid.uuid4()
        result = chat_manager.create_session(pid, "title")
        assert result == "resp"
        db_manager.create_session.assert_called_with(project_id=pid, title="title")
        logger.info.assert_called()


def test_chatmanager_create_session_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.create_session(None, "title")
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.create_session(uuid.uuid4(), "")


def test_chatmanager_create_session_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.create_session.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.create_session(uuid.uuid4(), "title")
    logger.error.assert_called()


def test_chatmanager_list_sessions_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.get_sessions_for_project.return_value = [mock.Mock()]
    with mock.patch.object(
        orchestrator_mod.ChatSessionResponse, "model_validate", side_effect=lambda x: x
    ):
        pid = uuid.uuid4()
        result = chat_manager.list_sessions(pid)
        assert result == db_manager.get_sessions_for_project.return_value


def test_chatmanager_list_sessions_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.list_sessions(None)


def test_chatmanager_list_sessions_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.get_sessions_for_project.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.list_sessions(uuid.uuid4())
    logger.error.assert_called()


def test_chatmanager_delete_session_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_session.return_value = True
    sid = uuid.uuid4()
    result = chat_manager.delete_session(sid)
    assert result is True
    logger.info.assert_called()


def test_chatmanager_delete_session_not_found(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_session.return_value = False
    sid = uuid.uuid4()
    result = chat_manager.delete_session(sid)
    assert result is False


def test_chatmanager_delete_session_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.delete_session(None)


def test_chatmanager_delete_session_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_session.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.delete_session(uuid.uuid4())
    logger.error.assert_called()


def test_chatmanager_add_message_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.add_message.return_value = mock.Mock()
    with mock.patch.object(
        orchestrator_mod.MessageResponse, "model_validate", return_value="resp"
    ):
        sid = uuid.uuid4()
        result = chat_manager.add_message(sid, "user", "hello")
        assert result == "resp"
        db_manager.add_message.assert_called_with(
            session_id=sid, role="user", content="hello"
        )


def test_chatmanager_add_message_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.add_message(None, "user", "hello")
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.add_message(uuid.uuid4(), "user", "")


def test_chatmanager_add_message_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.add_message.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.add_message(uuid.uuid4(), "user", "hello")
    logger.error.assert_called()


def test_chatmanager_get_messages_for_session_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.get_messages_for_session.return_value = [mock.Mock()]
    with mock.patch.object(
        orchestrator_mod.MessageResponse, "model_validate", side_effect=lambda x: x
    ):
        sid = uuid.uuid4()
        result = chat_manager.get_messages_for_session(sid)
        assert result == db_manager.get_messages_for_session.return_value


def test_chatmanager_get_messages_for_session_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.get_messages_for_session(None)


def test_chatmanager_get_messages_for_session_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.get_messages_for_session.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.get_messages_for_session(uuid.uuid4())
    logger.error.assert_called()


def test_chatmanager_delete_message_success(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_message.return_value = True
    mid = uuid.uuid4()
    result = chat_manager.delete_message(mid)
    assert result is True
    logger.info.assert_called()


def test_chatmanager_delete_message_not_found(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_message.return_value = False
    mid = uuid.uuid4()
    result = chat_manager.delete_message(mid)
    assert result is False


def test_chatmanager_delete_message_validation(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    with pytest.raises(orchestrator_mod.ValidationError):
        chat_manager.delete_message(None)


def test_chatmanager_delete_message_failure(monkeypatch : MonkeyPatch):
    chat_manager, db_manager, logger = make_chat_manager(monkeypatch)
    db_manager.delete_message.side_effect = Exception("fail")
    with pytest.raises(Exception):
        chat_manager.delete_message(uuid.uuid4())
    logger.error.assert_called()


@pytest.fixture
def setup_manager_components(tmp_path: Path):
    """Creates all the components needed to test the SetupManager."""
    version = "1.0.0"
    config = orchestrator_mod.AppConfig()
    dirs = orchestrator_mod.AppDirectories(
        config=tmp_path / "config",
        data=tmp_path / "data",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
    )
    dirs.create_directories()

    logger = mock.Mock()

    setup_manager = orchestrator_mod.SetupManager(dirs, config, logger)

    return setup_manager, logger


def test_setupmanager_is_first_launch_true_and_false(setup_manager_components):
    setup_manager, _ = setup_manager_components
    assert setup_manager.is_first_launch() is True
    setup_manager.perform_first_launch_setup()
    assert setup_manager.is_first_launch() is False


def test_setupmanager_perform_first_launch_setup_creates_marker(
    setup_manager_components,
):
    setup_manager, _ = setup_manager_components
    if setup_manager._marker_file.exists():
        setup_manager._marker_file.unlink()
    setup_manager.perform_first_launch_setup()
    assert setup_manager._marker_file.exists()


def test_setupmanager_perform_first_launch_setup_skips_if_already_done(
    setup_manager_components,
):
    setup_manager, logger = setup_manager_components
    setup_manager._marker_file.touch()
    assert setup_manager._marker_file.exists()
    setup_manager.perform_first_launch_setup()
    logger.info.assert_called_with("Skipping first launch setup - already completed")


def make_orchestrator(monkeypatch : MonkeyPatch, tmp_path : Path):

    mock_logger = mock.Mock()
    mock_app_config = mock.Mock()
    mock_app_config.database_filename = "db.sqlite3"
    mock_app_config.prompts_directory = str(tmp_path / "prompts")
    mock_app_config.get_setup_marker_filename = lambda version: ".setup_marker"
    mock_directories = mock.Mock()
    mock_directories.config = tmp_path / "config"
    mock_directories.data = tmp_path / "data"
    mock_directories.cache = tmp_path / "cache"
    mock_directories.logs = tmp_path / "logs"
    mock_directories.config.mkdir(parents=True, exist_ok=True)
    mock_directories.data.mkdir(parents=True, exist_ok=True)
    mock_directories.cache.mkdir(parents=True, exist_ok=True)
    mock_directories.logs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(orchestrator_mod, "ArataxAILogger", lambda: mock_logger)
    monkeypatch.setattr(orchestrator_mod, "AppConfig", lambda: mock_app_config)
    monkeypatch.setattr(
        orchestrator_mod,
        "AppDirectories",
        mock.Mock(create_default=lambda: mock_directories),
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "SetupManager",
        mock.Mock(
            return_value=mock.Mock(
                is_first_launch=lambda: False, perform_first_launch_setup=mock.Mock()
            )
        ),
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "ConfigurationManager",
        mock.Mock(
            return_value=mock.Mock(
                get_watched_directories=lambda: [],
                rag_config=mock.Mock(
                    get_config=lambda: mock.Mock(model_dump=lambda: {})
                ),
                llama_config=mock.Mock(
                    get_llama_cpp_params=lambda: mock.Mock(),
                    get_generation_params=lambda: mock.Mock(model_dump=lambda: {}),
                ),
                whisper_config=mock.Mock(
                    get_whisper_params=lambda: mock.Mock(),
                    get_transcription_params=lambda: mock.Mock(model_dump=lambda: {}),
                ),
            )
        ),
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "CoreAIServiceManager",
        mock.Mock(
            return_value=mock.Mock(
                get_configuration_status=lambda: {
                    "llama_configured": True,
                    "whisper_configured": True,
                },
                get_service=lambda: mock.Mock(),
                is_initialized=True,
                shutdown=mock.Mock(),
            )
        ),
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "ChatDatabaseManager",
        mock.Mock(
            return_value=mock.Mock(
                close=mock.Mock(), get_project_summary=lambda pid: {"summary": "ok"}
            )
        ),
    )
    monkeypatch.setattr(orchestrator_mod, "ChatContextManager", mock.Mock())
    monkeypatch.setattr(
        orchestrator_mod,
        "ChatManager",
        mock.Mock(
            return_value=mock.Mock(
                create_project=lambda n, d: "project",
                get_project=lambda pid: "project",
                list_projects=lambda: ["project1", "project2"],
                delete_project=lambda pid: True,
                create_session=lambda pid, t: "session",
                list_sessions=lambda pid: ["session1"],
                delete_session=lambda sid: True,
                add_message=lambda sid, r, c: "message",
                get_messages_for_session=lambda sid: ["msg1"],
                delete_message=lambda mid: True,
            )
        ),
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "AtaraxAIRAGManager",
        mock.Mock(
            return_value=mock.Mock(
                manifest=mock.Mock(is_valid=lambda rag_store: True),
                rag_store=mock.Mock(),
                rebuild_index=mock.Mock(),
                perform_initial_scan=mock.Mock(),
                start_file_monitoring=mock.Mock(),
                stop_file_monitoring=mock.Mock(),
                core_ai_service=None,
            )
        ),
    )
    monkeypatch.setattr(orchestrator_mod, "PromptManager", mock.Mock())
    monkeypatch.setattr(orchestrator_mod, "ContextManager", mock.Mock())
    monkeypatch.setattr(orchestrator_mod, "TaskManager", mock.Mock())
    monkeypatch.setattr(
        orchestrator_mod,
        "ChainRunner",
        mock.Mock(
            return_value=mock.Mock(
                run_chain=lambda chain_definition, initial_user_query: "result",
                core_ai_service=None,
            )
        ),
    )
    monkeypatch.setattr(orchestrator_mod, "__version__", "1.0.0")
    return AtaraxAIOrchestrator(app_config=mock_app_config)


def test_orchestrator_init(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    assert orch.app_config is not None
    assert orch.logger is not None
    assert orch.directories is not None
    assert orch.setup_manager is not None
    assert orch.config_manager is not None
    assert orch.core_ai_manager is not None


def test_orchestrator_run_task_chain(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    orch.chain_runner = mock.Mock()
    orch.chain_runner.core_ai_service = None
    orch.chain_runner.run_chain = mock.Mock(return_value="chain_result")
    orch.rag_manager = mock.Mock()
    orch.rag_manager.core_ai_service = None
    orch.core_ai_manager.get_service = mock.Mock(return_value="core_ai_service")
    chain_def = [{"task_id": "test_chain"}]
    result = orch.run_task_chain(chain_def, "query")
    assert result == "chain_result"
    orch.chain_runner.run_chain.assert_called_with(
        chain_definition=chain_def, initial_user_query="query"
    )


def test_orchestrator_run_task_chain_validation(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    with pytest.raises(orchestrator_mod.ValidationError):
        orch.run_task_chain([], "query")
    with pytest.raises(orchestrator_mod.ValidationError):
        orch.run_task_chain([{"task_id": "t"}], "")


def test_orchestrator_add_watch_directory(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    orch.config_manager.add_watched_directory = mock.Mock()
    orch.config_manager.get_watched_directories = mock.Mock(return_value=["/tmp"])
    orch.rag_manager.start_file_monitoring = mock.Mock()
    orch.add_watch_directory("/tmp")
    orch.config_manager.add_watched_directory.assert_called_with("/tmp")
    orch.rag_manager.start_file_monitoring.assert_called_with(["/tmp"])


def test_orchestrator_project_methods(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    assert orch.create_project("n", "d") == "project"
    assert orch.get_project("pid") == "project" # type: ignore
    assert orch.list_projects() == ["project1", "project2"]
    assert orch.delete_project("pid") is True # type: ignore


def test_orchestrator_session_methods(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    assert orch.create_session("pid", "title") == "session" # type: ignore
    assert orch.list_sessions("pid") == ["session1"] # type: ignore
    assert orch.delete_session("sid") is True # type: ignore


def test_orchestrator_message_methods(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    assert orch.add_message("sid", "role", "content") == "message" # type: ignore
    assert orch.get_messages_for_session("sid") == ["msg1"] # type: ignore
    assert orch.delete_message("mid") is True # type: ignore


def test_orchestrator_get_project_summary(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    orch.db_manager = mock.Mock(get_project_summary=lambda pid: {"summary": "ok"})
    assert orch.get_project_summary("pid") == {"summary": "ok"} # type: ignore


def test_orchestrator_shutdown(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    orch.rag_manager = mock.Mock(stop_file_monitoring=mock.Mock())
    orch.db_manager = mock.Mock(close=mock.Mock())
    orch.core_ai_manager = mock.Mock(shutdown=mock.Mock())
    orch.logger = mock.Mock()
    orch.shutdown()
    orch.rag_manager.stop_file_monitoring.assert_called()
    orch.db_manager.close.assert_called()
    orch.core_ai_manager.shutdown.assert_called()
    orch.logger.info.assert_any_call("AtaraxAI shutdown completed successfully")


def test_orchestrator_context_manager(monkeypatch : MonkeyPatch, tmp_path : Path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    with orch as o:
        assert o is orch


