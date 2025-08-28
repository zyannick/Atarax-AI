from datetime import datetime
import uuid
import pytest
from unittest import mock
from ataraxai.praxis.utils.chat_manager import ChatManager


@pytest.fixture
def mock_db_manager():
    return mock.AsyncMock()

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

@pytest.mark.asyncio
async def test_create_project_success(chat_manager, mock_db_manager):
    project_data = {"id": uuid.uuid4(), "name": "Test", "description": "desc"}
    mock_db_manager.create_project.return_value = project_data
    with mock.patch("ataraxai.praxis.utils.chat_manager.ProjectResponse") as MockPR:
        MockPR.model_validate.return_value = "validated_project"
        result = await chat_manager.create_project("Test", "desc")
        assert result == "validated_project"
        mock_db_manager.create_project.assert_called_once_with(name="Test", description="desc")

@pytest.mark.asyncio
async def test_create_project_validation_error(chat_manager):
    with pytest.raises(Exception):
        await chat_manager.create_project("", "desc")

@pytest.mark.asyncio
async def test_get_project_success(chat_manager, mock_db_manager):
    project_id = uuid.uuid4()
    project_data = {"id": project_id, "name": "Test", "description": "desc"}
    mock_db_manager.get_project.return_value = project_data
    with mock.patch("ataraxai.praxis.utils.chat_manager.ProjectResponse") as MockPR:
        MockPR.model_validate.return_value = "validated_project"
        result = await chat_manager.get_project(project_id)
        assert result == "validated_project"
        mock_db_manager.get_project.assert_called_once_with(project_id)

@pytest.mark.asyncio
async def test_list_projects_success(chat_manager, mock_db_manager):
    projects = [{"id": uuid.uuid4(), "name": "A", "description": "d"}]
    mock_db_manager.list_projects.return_value = projects
    with mock.patch("ataraxai.praxis.utils.chat_manager.ProjectResponse") as MockPR:
        MockPR.model_validate.side_effect = lambda x: x
        result = await chat_manager.list_projects()
        assert result == projects

@pytest.mark.asyncio
async def test_delete_project_success(chat_manager, mock_db_manager):
    project_id = uuid.uuid4()
    mock_db_manager.delete_project.return_value = True
    result = await chat_manager.delete_project(project_id)
    assert result is True
    mock_db_manager.delete_project.assert_called_once_with(project_id)

@pytest.mark.asyncio
async def test_create_session_success(chat_manager, mock_db_manager):
    project_id = uuid.uuid4()
    session_data = {"id": uuid.uuid4(), "project_id": project_id, "title": "Session"}
    mock_db_manager.create_session.return_value = session_data
    with mock.patch("ataraxai.praxis.utils.chat_manager.ChatSessionResponse") as MockCSR:
        MockCSR.model_validate.return_value = "validated_session"
        result = await chat_manager.create_session(project_id, "Session")
        assert result == "validated_session"

@pytest.mark.asyncio
async def test_get_session_success(chat_manager, mock_db_manager, mock_vault_manager):
    session_id = uuid.uuid4()
    message = mock.Mock()
    message.content = b"encrypted_hello"
    message.get_id.return_value = uuid.uuid4()
    message.get_role.return_value = "user"
    message.get_date_time.return_value = datetime.now()
    db_session = mock.Mock()
    db_session.get_id.return_value = session_id
    db_session.get_project_id.return_value = uuid.uuid4()
    db_session.get_title.return_value = "title"
    db_session.get_created_at.return_value = datetime.now()
    db_session.get_updated_at.return_value = datetime.now()
    db_session.messages = [message]
    mock_db_manager.get_session.return_value = db_session
    result = await chat_manager.get_session(session_id)
    assert result.id == session_id
    assert result.messages[0].content == "hello"

@pytest.mark.asyncio
async def test_list_sessions_success(chat_manager, mock_db_manager):
    project_id = uuid.uuid4()
    sessions = [{"id": uuid.uuid4(), "project_id": project_id, "title": "S"}]
    mock_db_manager.get_sessions_for_project.return_value = sessions
    with mock.patch("ataraxai.praxis.utils.chat_manager.ChatSessionResponse") as MockCSR:
        MockCSR.model_validate.side_effect = lambda x: x
        result = await chat_manager.list_sessions(project_id)
        assert result == sessions

@pytest.mark.asyncio
async def test_delete_session_success(chat_manager, mock_db_manager):
    session_id = uuid.uuid4()
    mock_db_manager.delete_session.return_value = True
    result = await chat_manager.delete_session(session_id)
    assert result is True

@pytest.mark.asyncio
async def test_add_message_success(chat_manager, mock_db_manager, mock_vault_manager):
    session_id = uuid.uuid4()
    db_message = mock.Mock()
    db_message.id = uuid.uuid4()
    db_message.session.id = session_id
    db_message.role = "user"
    db_message.get_date_time.return_value = "2024-06-01T12:00:00"
    mock_db_manager.add_message.return_value = db_message
    result = await chat_manager.add_message(session_id, "user", "hello")
    assert result.content == "hello"
    assert result.role == "user"

@pytest.mark.asyncio
async def test_get_message_success(chat_manager, mock_db_manager, mock_vault_manager):
    message_id = uuid.uuid4()
    db_message = mock.Mock()
    db_message.id = message_id
    db_message.session.id = uuid.uuid4()
    db_message.role = "assistant"
    db_message.content = b"encrypted_reply"
    db_message.get_date_time.return_value = "2024-06-01T12:00:00"
    mock_db_manager.get_message.return_value = db_message
    result = await chat_manager.get_message(message_id)
    assert result.content == "reply"
    assert result.role == "assistant"

@pytest.mark.asyncio
async def test_get_messages_for_session_success(chat_manager, mock_db_manager, mock_vault_manager):
    session_id = uuid.uuid4()
    msg = mock.Mock()
    msg.content = b"encrypted_foo"
    msg.get_id.return_value = uuid.uuid4()
    msg.get_session_id.return_value = session_id
    msg.get_role.return_value = "user"
    msg.get_date_time.return_value = "2024-06-01T12:00:00"
    mock_db_manager.get_messages_for_session.return_value = [msg]
    result = await chat_manager.get_messages_for_session(session_id)
    assert result[0].content == "foo"

@pytest.mark.asyncio
async def test_delete_message_success(chat_manager, mock_db_manager):
    message_id = uuid.uuid4()
    mock_db_manager.delete_message.return_value = True
    result = await chat_manager.delete_message(message_id)
    assert result is True
