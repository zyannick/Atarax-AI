import uuid
import pytest
from unittest import mock
from ataraxai.praxis.utils.chat_manager import ChatManager

@pytest.fixture
def mock_db_manager():
    return mock.Mock()

@pytest.fixture
def mock_logger():
    return mock.Mock()

@pytest.fixture
def mock_vault_manager():
    vm = mock.Mock()
    vm.encrypt.side_effect = lambda b: b"encrypted_" + b
    vm.decrypt.side_effect = lambda b: b.replace(b"encrypted_", b"")
    return vm

@pytest.fixture
def chat_manager(mock_db_manager, mock_logger, mock_vault_manager):
    return ChatManager(
        db_manager=mock_db_manager,
        logger=mock_logger,
        vault_manager=mock_vault_manager,
    )

def test_create_project_success(chat_manager, mock_db_manager):
    project_data = {"id": uuid.uuid4(), "name": "Test", "description": "desc"}
    with mock.patch("ataraxai.praxis.modules.chat.chat_models.ProjectResponse.model_validate", return_value="validated_project"):
        mock_db_manager.create_project.return_value = project_data
        result = chat_manager.create_project("Test", "desc")
        assert result == "validated_project"
        chat_manager.logger.info.assert_called_once()

def test_create_project_validation_error(chat_manager):
    with pytest.raises(Exception):
        chat_manager.create_project("", "desc")

def test_get_project_success(chat_manager, mock_db_manager):
    pid = uuid.uuid4()
    project_data = {"id": pid, "name": "Test", "description": "desc"}
    with mock.patch("ataraxai.praxis.modules.chat.chat_models.ProjectResponse.model_validate", return_value="validated_project"):
        mock_db_manager.get_project.return_value = project_data
        result = chat_manager.get_project(pid)
        assert result == "validated_project"

def test_list_projects_success(chat_manager, mock_db_manager):
    projects = [{"id": uuid.uuid4(), "name": "A", "description": "d"}]
    with mock.patch("ataraxai.praxis.modules.chat.chat_models.ProjectResponse.model_validate", side_effect=lambda x: x):
        mock_db_manager.list_projects.return_value = projects
        result = chat_manager.list_projects()
        assert result == projects

def test_delete_project_success(chat_manager, mock_db_manager):
    pid = uuid.uuid4()
    mock_db_manager.delete_project.return_value = True
    result = chat_manager.delete_project(pid)
    assert result is True
    chat_manager.logger.info.assert_called_once()

def test_create_session_success(chat_manager, mock_db_manager):
    pid = uuid.uuid4()
    session_data = {"id": uuid.uuid4(), "project_id": pid, "title": "Session"}
    with mock.patch("ataraxai.praxis.modules.chat.chat_models.ChatSessionResponse.model_validate", return_value="validated_session"):
        mock_db_manager.create_session.return_value = session_data
        result = chat_manager.create_session(pid, "Session")
        assert result == "validated_session"

def test_get_session_success(chat_manager, mock_db_manager):
    sid = uuid.uuid4()
    session_data = {"id": sid, "title": "Session"}
    with mock.patch("ataraxai.praxis.modules.chat.chat_models.ChatSessionResponse.model_validate", return_value="validated_session"):
        mock_db_manager.get_session.return_value = session_data
        result = chat_manager.get_session(sid)
        assert result == "validated_session"

def test_list_sessions_success(chat_manager, mock_db_manager):
    pid = uuid.uuid4()
    sessions = [{"id": uuid.uuid4(), "title": "S"}]
    with mock.patch("ataraxai.praxis.modules.chat.chat_models.ChatSessionResponse.model_validate", side_effect=lambda x: x):
        mock_db_manager.get_sessions_for_project.return_value = sessions
        result = chat_manager.list_sessions(pid)
        assert result == sessions

def test_delete_session_success(chat_manager, mock_db_manager):
    sid = uuid.uuid4()
    mock_db_manager.delete_session.return_value = True
    result = chat_manager.delete_session(sid)
    assert result is True
    chat_manager.logger.info.assert_called_once()

def test_add_message_success(chat_manager, mock_db_manager, mock_vault_manager):
    sid = uuid.uuid4()
    db_message = mock.Mock()
    db_message.id = uuid.uuid4()
    db_message.session = mock.Mock(id=sid)
    db_message.role = "user"
    db_message.timestamp = mock.Mock()
    db_message.timestamp.to_timestamp.return_value = 1234567890
    mock_db_manager.add_message.return_value = db_message
    result = chat_manager.add_message(sid, "user", "hello")
    assert result.content == "hello"
    assert result.role == "user"
    assert result.session_id == sid

def test_get_message_success(chat_manager, mock_db_manager, mock_vault_manager):
    mid = uuid.uuid4()
    sid = uuid.uuid4()
    db_message = mock.Mock()
    db_message.id = mid
    db_message.session = mock.Mock(id=sid)
    db_message.role = "user"
    db_message.content = b"encrypted_hello"
    db_message.timestamp = mock.Mock()
    db_message.timestamp.to_timestamp.return_value = 1234567890
    mock_db_manager.get_message.return_value = db_message
    result = chat_manager.get_message(mid)
    assert result.content == "hello"
    assert result.role == "user"
    assert result.session_id == sid

def test_get_messages_for_session_success(chat_manager, mock_db_manager, mock_vault_manager):
    sid = uuid.uuid4()
    db_message = mock.Mock()
    db_message.id = uuid.uuid4()
    db_message.session = mock.Mock(id=sid)
    db_message.role = "user"
    db_message.content = b"encrypted_hello"
    db_message.timestamp = mock.Mock()
    db_message.timestamp.to_timestamp.return_value = 1234567890
    mock_db_manager.get_messages_for_session.return_value = [db_message]
    result = chat_manager.get_messages_for_session(sid)
    assert len(result) == 1
    assert result[0].content == "hello"

def test_delete_message_success(chat_manager, mock_db_manager):
    mid = uuid.uuid4()
    mock_db_manager.delete_message.return_value = True
    result = chat_manager.delete_message(mid)
    assert result is True
    chat_manager.logger.info.assert_called_once()
   