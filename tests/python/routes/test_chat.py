import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest import mock
import uuid
from ataraxai.routes.chat import router_chat
from ataraxai.routes.status import Status
from fastapi import FastAPI
from ataraxai.routes.chat import get_unlocked_orchestrator


app = FastAPI()
app.include_router(router_chat)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_orchestrator():
    orch = mock.Mock()
    orch.chat = mock.Mock()
    return orch

def override_get_unlocked_orchestrator():
    return mock_orchestrator()

app.dependency_overrides = {}

def set_orch_override(orch):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch

def test_create_new_project_success(client, mock_orchestrator):
    mock_project = mock.Mock()
    mock_project.id = uuid.uuid4()
    mock_project.name = "Test Project"
    mock_project.description = "A test project"
    mock_orchestrator.chat.create_project.return_value = mock_project
    set_orch_override(mock_orchestrator)

    payload = {
        "name": "Test Project",
        "description": "A test project"
    }
    response = client.post("/api/v1/chat/projects", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["project_id"] == str(mock_project.id)
    assert data["name"] == "Test Project"
    assert data["description"] == "A test project"

def test_create_new_project_failure(client, mock_orchestrator):
    mock_orchestrator.chat.create_project.side_effect = Exception("fail")
    set_orch_override(mock_orchestrator)
    payload = {
        "name": "Bad Project",
        "description": "Should fail"
    }
    response = client.post("/api/v1/chat/projects", json=payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

def test_delete_project_success(client, mock_orchestrator):
    project_id = uuid.uuid4()
    mock_project = mock.Mock()
    mock_project.id = project_id
    mock_orchestrator.chat.get_project.return_value = mock_project
    set_orch_override(mock_orchestrator)
    response = client.delete(f"/api/v1/chat/projects/{project_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS

def test_delete_project_not_found(client, mock_orchestrator):
    project_id = uuid.uuid4()
    mock_orchestrator.chat.get_project.return_value = None
    set_orch_override(mock_orchestrator)
    response = client.delete(f"/api/v1/chat/projects/{project_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_get_project_success(client, mock_orchestrator):
    project_id = uuid.uuid4()
    mock_project = mock.Mock()
    mock_project.id = project_id
    mock_project.name = "Proj"
    mock_project.description = "Desc"
    mock_orchestrator.chat.get_project.return_value = mock_project
    set_orch_override(mock_orchestrator)
    response = client.get(f"/api/v1/chat/projects/{project_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["project_id"] == str(project_id)

def test_get_project_not_found(client, mock_orchestrator):
    project_id = uuid.uuid4()
    mock_orchestrator.chat.get_project.return_value = None
    set_orch_override(mock_orchestrator)
    response = client.get(f"/api/v1/chat/projects/{project_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_list_projects(client, mock_orchestrator):
    mock_project = mock.Mock()
    mock_project.id = uuid.uuid4()
    mock_project.name = "Proj"
    mock_project.description = "Desc"
    mock_orchestrator.chat.list_projects.return_value = [mock_project]
    set_orch_override(mock_orchestrator)
    response = client.get("/api/v1/chat/projects")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)
    assert response.json()[0]["name"] == "Proj"

def test_list_sessions(client, mock_orchestrator):
    project_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = uuid.uuid4()
    mock_session.title = "Session"
    mock_session.project_id = project_id
    mock_orchestrator.chat.list_sessions.return_value = [mock_session]
    set_orch_override(mock_orchestrator)
    response = client.get(f"/api/v1/chat/projects/{project_id}/sessions")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)
    assert response.json()[0]["title"] == "Session"

def test_create_session_success(client, mock_orchestrator):
    project_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = uuid.uuid4()
    mock_session.title = "Session"
    mock_session.project_id = project_id
    mock_orchestrator.chat.create_session.return_value = mock_session
    set_orch_override(mock_orchestrator)
    payload = {
        "project_id": str(project_id),
        "title": "Session"
    }
    response = client.post("/api/v1/chat/session", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["title"] == "Session"

def test_create_session_failure(client, mock_orchestrator):
    mock_orchestrator.chat.create_session.side_effect = Exception("fail")
    set_orch_override(mock_orchestrator)
    payload = {
        "project_id": str(uuid.uuid4()),
        "title": "Session"
    }
    response = client.post("/api/v1/chat/session", json=payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

def test_delete_session_success(client, mock_orchestrator):
    session_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    mock_orchestrator.chat.get_session.return_value = mock_session
    set_orch_override(mock_orchestrator)
    response = client.delete(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS

def test_delete_session_not_found(client, mock_orchestrator):
    session_id = uuid.uuid4()
    mock_orchestrator.chat.get_session.return_value = None
    set_orch_override(mock_orchestrator)
    response = client.delete(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_get_session_success(client, mock_orchestrator):
    session_id = uuid.uuid4()
    project_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    mock_session.title = "Session"
    mock_session.project_id = project_id
    mock_orchestrator.chat.get_session.return_value = mock_session
    set_orch_override(mock_orchestrator)
    response = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["session_id"] == str(session_id)

def test_get_session_not_found(client, mock_orchestrator):
    session_id = uuid.uuid4()
    mock_orchestrator.chat.get_session.return_value = None
    set_orch_override(mock_orchestrator)
    response = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_send_message_success(client, mock_orchestrator):
    session_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    mock_orchestrator.chat.get_session.return_value = mock_session
    mock_response = mock.Mock()
    mock_response.content = "Assistant reply"
    mock_orchestrator.chat.add_message.return_value = mock_response
    set_orch_override(mock_orchestrator)
    payload = {
        "user_query": "Hello"
    }
    response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["assistant_response"] == "Assistant reply"

def test_send_message_session_not_found(client, mock_orchestrator):
    session_id = uuid.uuid4()
    mock_orchestrator.chat.get_session.return_value = None
    set_orch_override(mock_orchestrator)
    payload = {
        "user_query": "Hello"
    }
    response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json=payload)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_send_message_failure(client, mock_orchestrator):
    session_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    mock_orchestrator.chat.get_session.return_value = mock_session
    mock_orchestrator.chat.add_message.side_effect = Exception("fail")
    set_orch_override(mock_orchestrator)
    payload = {
        "user_query": "Hello"
    }
    response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json=payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR