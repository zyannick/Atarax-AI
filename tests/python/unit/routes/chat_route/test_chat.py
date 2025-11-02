import datetime
import uuid
from unittest import mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# from fastapi import FastAPI
from api import app
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.app_state import AppState

# from ataraxai.routes.chat_route.chat import router_chat
from ataraxai.routes.chat_route.chat_api_models import CreateSessionRequestAPI
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.routes.status import StatusResponse
from ataraxai.routes.status import TaskStatus as Status


@pytest.fixture
def mock_orchestrator(monkeypatch: pytest.MonkeyPatch):
    mock_orch = mock.AsyncMock()
    mock_orch.get_state = mock.AsyncMock(return_value=AppState.UNLOCKED)
    monkeypatch.setattr(
        "ataraxai.routes.dependency_api.get_unlocked_orchestrator", lambda: mock_orch
    )
    return mock_orch


@pytest.fixture
def mock_req_manager(monkeypatch: pytest.MonkeyPatch):
    mock_req = mock.AsyncMock()
    monkeypatch.setattr(
        "ataraxai.routes.dependency_api.get_request_manager", lambda: mock_req
    )
    return mock_req


@pytest.fixture
def mock_task_manager(monkeypatch: pytest.MonkeyPatch):
    mock_task = mock.Mock()
    monkeypatch.setattr(
        "ataraxai.routes.dependency_api.get_gatewaye_task_manager", lambda: mock_task
    )
    return mock_task


@pytest.mark.asyncio
async def test_create_new_project(
    client: TestClient,
    mock_orchestrator: mock.AsyncMock,
    mock_req_manager: mock.AsyncMock,
    mock_task_manager: mock.Mock,
):
    mock_chat_manager = mock.AsyncMock()
    mock_orchestrator.get_chat_manager.return_value = mock_chat_manager

    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    mock_project = mock.Mock()
    mock_project.id = uuid.uuid4()
    mock_project.name = "Test Project"
    mock_project.description = "Desc"
    mock_project.created_at = datetime.datetime.now()
    mock_project.updated_at = datetime.datetime.now()

    async def mock_future():
        return mock_project

    mock_req_manager.submit_request.return_value = mock_future()

    response = client.post(
        "/api/v1/chat/projects", json={"name": "Test Project", "description": "Desc"}
    )
    assert response.status_code == status.HTTP_200_OK, f"Unexpected status code: {response.text}"
    data = response.json()
    assert data["name"] == "Test Project"
    assert data["description"] == "Desc"
    assert "project_id" in data

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_delete_project_not_found(
    client: TestClient,
    mock_orchestrator: mock.AsyncMock,
    mock_req_manager: mock.AsyncMock,
    mock_task_manager: mock.Mock,
):
    mock_chat_manager = mock.AsyncMock()
    mock_orchestrator.get_chat_manager.return_value = mock_chat_manager
    mock_chat_manager.get_project.return_value = None

    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    project_id = str(uuid.uuid4())
    response = client.delete(f"/api/v1/chat/projects/{project_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Project not found"

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_get_project_deletion_status_not_found(
    client: TestClient, mock_orchestrator: mock.AsyncMock, mock_task_manager: mock.Mock
):

    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    mock_task_manager.get_task_status.return_value = None
    task_id = "fake-task-id"
    response = client.get(f"/api/v1/chat/projects/delete/{task_id}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.ERROR.value
    assert data["task_id"] == task_id

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_list_projects(
    client: TestClient,
    mock_orchestrator: mock.AsyncMock,
    mock_req_manager: mock.AsyncMock,
    mock_task_manager: mock.Mock,
):
    mock_chat_manager = mock.AsyncMock()
    mock_orchestrator.get_chat_manager.return_value = mock_chat_manager

    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    mock_project = mock.Mock()
    mock_project.id = uuid.uuid4()
    mock_project.name = "Proj"
    mock_project.description = "Desc"
    mock_project.created_at = datetime.datetime.now()
    mock_project.updated_at = datetime.datetime.now()

    async def mock_future():
        return [mock_project]

    mock_req_manager.submit_request.return_value = mock_future()

    response = client.get("/api/v1/chat/projects")
    assert response.status_code == status.HTTP_200_OK, f"Unexpected status code: {response.text}"
    data = response.json()
    projects = data["projects"]
    assert len(projects) == 1
    assert projects[0]["name"] == "Proj"

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_create_session_project_not_found(
    client: TestClient,
    mock_orchestrator: mock.AsyncMock,
    mock_req_manager: mock.AsyncMock,
    mock_task_manager: mock.Mock,
):
    mock_chat_manager = mock.AsyncMock()
    mock_orchestrator.get_chat_manager.return_value = mock_chat_manager
    mock_chat_manager.get_project.side_effect = Exception("Not found")

    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    session_data = CreateSessionRequestAPI(project_id=uuid.uuid4(), title="Session")

    response = client.post(
        "/api/v1/chat/sessions", json=session_data.model_dump(mode="json")
    )
    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Unexpected status code: {response.text}"
    assert response.json()["detail"] == "Project not found"

    app.dependency_overrides = {}


def test_get_session_not_found(
    client: TestClient,
    mock_orchestrator: mock.AsyncMock,
    mock_req_manager: mock.AsyncMock,
    mock_task_manager: mock.Mock,
):
    mock_chat_manager = mock.AsyncMock()
    mock_orchestrator.get_chat_manager.return_value = mock_chat_manager
    mock_chat_manager.get_session.return_value = None

    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    session_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Unexpected status code: {response.text}"
    assert response.json()["detail"] == "Session not found"

    app.dependency_overrides = {}


def test_get_messages_session_not_found(
    client: TestClient,
    mock_orchestrator: mock.AsyncMock,
    mock_req_manager: mock.AsyncMock,
    mock_task_manager: mock.Mock,
):
    mock_chat_manager = mock.AsyncMock()
    mock_orchestrator.get_chat_manager.return_value = mock_chat_manager
    mock_chat_manager.get_session.return_value = None

    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    session_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/chat/sessions/{session_id}/messages")
    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Unexpected status code: {response.text}"
    assert response.json()["detail"] == "Session not found"

    app.dependency_overrides = {}
