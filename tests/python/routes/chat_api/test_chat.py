import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest import mock
import uuid
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.chat_api.chat import router_chat
from ataraxai.routes.status import Status
from fastapi import FastAPI
from api import app
from ataraxai.routes.chat_api.chat import get_unlocked_orchestrator


@pytest.fixture
def client(monkeypatch):
    """
    This is the master fixture. It mocks the orchestrator at the factory level,
    then runs the app's lifespan events to ensure app.state is populated correctly.
    """

    mock_orchestrator = mock.MagicMock()
    monkeypatch.setattr(
        AtaraxAIOrchestratorFactory, "create_orchestrator", lambda: mock_orchestrator
    )

    with TestClient(app, base_url="http://test") as test_client:
        test_client.mock_orchestrator = mock_orchestrator
        test_client.mock_orchestrator.state = AppState.UNLOCKED
        yield test_client


def test_create_new_project_success(client):
    mock_project = mock.Mock()
    mock_project.id = uuid.uuid4()
    mock_project.name = "Test Project"
    mock_project.description = "A test project"
    client.mock_orchestrator.chat.create_project.return_value = mock_project

    payload = {"name": "Test Project", "description": "A test project"}
    response = client.post("/api/v1/chat/projects", json=payload)

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"API returned an error: {response.text}"
    data = response.json()
    assert data["project_id"] == str(mock_project.id)
    assert data["name"] == "Test Project"
    assert data["description"] == "A test project"


def test_create_new_project_failure(client):
    client.mock_orchestrator.chat.create_project.side_effect = Exception("DB Error")

    payload = {"name": "Bad Project", "description": "Should fail"}
    response = client.post("/api/v1/chat/projects", json=payload)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_delete_project_success(client):
    project_id = uuid.uuid4()
    client.mock_orchestrator.chat.get_project.return_value = mock.Mock()

    response = client.delete(f"/api/v1/chat/projects/{project_id}")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS


def test_delete_project_not_found(client):
    project_id = uuid.uuid4()
    client.mock_orchestrator.chat.get_project.return_value = None

    response = client.delete(f"/api/v1/chat/projects/{project_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_project_success(client):
    project_id = uuid.uuid4()
    mock_project = mock.Mock()
    mock_project.id = project_id
    mock_project.name = "Proj"
    mock_project.description = "Desc"
    client.mock_orchestrator.chat.get_project.return_value = mock_project
    response = client.get(f"/api/v1/chat/projects/{project_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["project_id"] == str(project_id)


def test_get_project_not_found(client):
    project_id = uuid.uuid4()
    client.mock_orchestrator.chat.get_project.return_value = None
    response = client.get(f"/api/v1/chat/projects/{project_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_list_projects(client):
    mock_project = mock.Mock()
    mock_project.id = uuid.uuid4()
    mock_project.name = "Proj"
    mock_project.description = "Desc"
    client.mock_orchestrator.chat.list_projects.return_value = [mock_project]

    response = client.get("/api/v1/chat/projects")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)
    assert response.json()[0]["name"] == "Proj"


def test_list_sessions(client):
    project_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = uuid.uuid4()
    mock_session.title = "Session"
    mock_session.project_id = project_id
    client.mock_orchestrator.chat.list_sessions.return_value = [mock_session]

    response = client.get(f"/api/v1/chat/projects/{project_id}/sessions")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)
    assert response.json()[0]["title"] == "Session"


def test_create_session_success(client):
    project_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = uuid.uuid4()
    mock_session.title = "Session"
    mock_session.project_id = project_id
    client.mock_orchestrator.chat.create_session.return_value = mock_session

    payload = {"project_id": str(project_id), "title": "Session"}
    response = client.post("/api/v1/chat/sessions", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["title"] == "Session"


def test_create_session_failure(client):
    client.mock_orchestrator.chat.create_session.side_effect = Exception("fail")

    payload = {"project_id": str(uuid.uuid4()), "title": "Session"}
    response = client.post("/api/v1/chat/sessions", json=payload)
    assert (
        response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    ), f"API returned an error: {response.text}"


def test_delete_session_success(client):
    session_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    client.mock_orchestrator.chat.get_session.return_value = mock_session

    response = client.delete(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS


def test_delete_session_not_found(client):
    session_id = uuid.uuid4()
    client.mock_orchestrator.chat.get_session.return_value = None

    response = client.delete(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_session_success(client):
    session_id = uuid.uuid4()
    project_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    mock_session.title = "Session"
    mock_session.project_id = project_id
    client.mock_orchestrator.chat.get_session.return_value = mock_session

    response = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["session_id"] == str(session_id)


def test_get_session_not_found(client):
    session_id = uuid.uuid4()
    client.mock_orchestrator.chat.get_session.return_value = None

    response = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_send_message_success(client):
    session_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    client.mock_orchestrator.chat.get_session.return_value = mock_session
    mock_response = mock.Mock()
    mock_response.session_id = session_id
    mock_response.role = "user"
    mock_response.date_time = "2024-06-01T12:00:00"
    mock_response.id = uuid.uuid4()
    mock_response.content = "Random content"
    client.mock_orchestrator.chat.add_message.return_value = mock_response

    payload = {"user_query": "Hello"}
    response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json=payload)
    assert response.status_code == status.HTTP_200_OK, f"API returned an error: {response.text}"
    data = response.json()
    assert data["session_id"] == str(session_id)
    assert data["role"] == "user"
    assert data["content"] == "Random content"
    assert data["date_time"] == "2024-06-01T12:00:00"


def test_send_message_session_not_found(client):
    session_id = uuid.uuid4()
    client.mock_orchestrator.chat.get_session.return_value = None

    payload = {"user_query": "Hello"}
    response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json=payload)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_send_message_failure(client):
    session_id = uuid.uuid4()
    mock_session = mock.Mock()
    mock_session.id = session_id
    client.mock_orchestrator.chat.get_session.return_value = mock_session
    client.mock_orchestrator.chat.add_message.side_effect = Exception("fail")

    payload = {"user_query": "Hello"}
    response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json=payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
