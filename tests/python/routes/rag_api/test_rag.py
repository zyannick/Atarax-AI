from fastapi.testclient import TestClient
import pytest
from unittest.mock import MagicMock
from fastapi import BackgroundTasks, status

from api import app
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from ataraxai.routes.status import Status


@pytest.fixture
def client(monkeypatch):
    """
    This is the master fixture. It mocks the orchestrator at the factory level,
    then runs the app's lifespan events to ensure app.state is populated correctly.
    """

    mock_orchestrator = MagicMock()
    monkeypatch.setattr(
        AtaraxAIOrchestratorFactory, "create_orchestrator", lambda: mock_orchestrator
    )

    with TestClient(app, base_url="http://test") as test_client:
        test_client.mock_orchestrator = mock_orchestrator
        test_client.mock_orchestrator.state = AppState.UNLOCKED
        yield test_client


def test_check_manifest_success(client):
    client.mock_orchestrator.rag.check_manifest_validity.return_value = True
    response = client.get("/api/v1/rag/check_manifest")
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Manifest is valid."
    client.mock_orchestrator.rag.check_manifest_validity.assert_called_once()


def test_check_manifest_failure(client):
    client.mock_orchestrator.rag.check_manifest_validity.return_value = False
    response = client.get("/api/v1/rag/check_manifest")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.ERROR
    assert data["message"] == "Manifest is invalid or missing."


def test_check_manifest_exception(client):
    client.mock_orchestrator.rag.check_manifest_validity.side_effect = Exception(
        "Disk read error"
    )
    response = client.get("/api/v1/rag/check_manifest")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_rebuild_index_success(client):
    client.mock_orchestrator.rag.rebuild_index.return_value = True
    response = client.post("/api/v1/rag/rebuild_index")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS


def test_rebuild_index_exception(client):
    client.mock_orchestrator.rag.rebuild_index.side_effect = Exception("Rebuild failed")
    response = client.post("/api/v1/rag/rebuild_index")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_task_endpoints_fail_when_vault_is_locked(client):

    client.mock_orchestrator.state = AppState.LOCKED
    endpoints = [
        "/api/v1/rag/scan_and_index",
        "/api/v1/rag/add_directories",
        "/api/v1/rag/remove_directories",
    ]
    payload = {"directories": ["dir"]}

    for endpoint in endpoints:
        response = client.post(endpoint, json=payload)
        assert response.status_code == status.HTTP_403_FORBIDDEN


def test_scan_and_index_schedules_task(client):
    mock_orch = MagicMock()
    mock_orch.state = AppState.UNLOCKED
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orch

    mock_background_tasks = MagicMock(spec=BackgroundTasks)
    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks

    response = client.post("/api/v1/rag/scan_and_index")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS
    app.dependency_overrides.clear()


def test_add_directory_schedules_task(client):
    mock_orch = MagicMock()
    mock_orch.state = AppState.UNLOCKED
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orch

    mock_background_tasks = MagicMock(spec=BackgroundTasks)
    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks
    payload = {"directories": ["dir1", "dir2"]}

    response = client.post("/api/v1/rag/add_directories", json=payload)

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS
    app.dependency_overrides.clear()
