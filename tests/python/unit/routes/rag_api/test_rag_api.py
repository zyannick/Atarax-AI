from fastapi.testclient import TestClient
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import BackgroundTasks, status

from api import app
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.routes.status import Status


# @pytest.fixture
# def unlocked_client(client):
#     client.app.state.orchestrator.state = AppState.UNLOCKED
#     yield client


# @pytest.fixture
# def locked_client(client):
#     client.app.state.orchestrator.state = AppState.LOCKED
#     yield client


# @pytest.fixture
# def first_launch_client(client):
#     client.app.state.orchestrator.state = AppState.FIRST_LAUNCH
#     yield client


@pytest.fixture
def orch_mock():
    orch = MagicMock()
    orch.state = AppState.UNLOCKED
    orch.rag.rebuild_index = AsyncMock(return_value=True)
    orch.rag.check_manifest_validity = AsyncMock(return_value=True)
    orch.rag.health_check = AsyncMock(return_value=True)
    orch.rag.add_watch_directories = AsyncMock(return_value=True)
    orch.rag.remove_watch_directories = AsyncMock(return_value=True)
    return orch


@pytest.fixture
def req_manager_mock():
    req_manager = MagicMock()
    req_manager.submit_request = AsyncMock(return_value=AsyncMock())
    return req_manager


@pytest.fixture
def task_manager_mock():
    task_manager = MagicMock()
    task_manager.create_task = MagicMock(return_value="task123")
    task_manager.get_task_status = MagicMock(return_value="done")
    task_manager.cancel_task = MagicMock(return_value=True)
    return task_manager


def test_rebuild_index(
    client: TestClient,
    orch_mock: MagicMock,
    req_manager_mock: MagicMock,
    task_manager_mock: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch_mock
    app.dependency_overrides[get_request_manager] = lambda: req_manager_mock
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: task_manager_mock

    response = client.post("/api/v1/rag/rebuild_index")
    assert response.status_code == 200, f"Expected 200, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["result"] == "task123"

    app.dependency_overrides = {}


def test_get_rebuild_index_result(
    client: TestClient,
    orch_mock: MagicMock,
    req_manager_mock: MagicMock,
    task_manager_mock: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch_mock
    app.dependency_overrides[get_request_manager] = lambda: req_manager_mock
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: task_manager_mock

    response = client.get("/api/v1/rag/rebuild_index/task123")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["result"] == "done"

    task_manager_mock.get_task_status.return_value = None
    response = client.get("/api/v1/rag/rebuild_index/unknown")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == Status.ERROR

    app.dependency_overrides = {}


def test_cancel_rebuild_index(
    client: TestClient,
    orch_mock: MagicMock,
    req_manager_mock: MagicMock,
    task_manager_mock: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch_mock
    app.dependency_overrides[get_request_manager] = lambda: req_manager_mock
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: task_manager_mock

    response = client.delete("/api/v1/rag/rebuild_index/task123")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == Status.SUCCESS

    task_manager_mock.cancel_task.return_value = False
    response = client.delete("/api/v1/rag/rebuild_index/unknown")
    assert response.status_code == 404

    app.dependency_overrides = {}


def test_check_manifest(
    client: TestClient,
    orch_mock: MagicMock,
    req_manager_mock: MagicMock,
    task_manager_mock: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch_mock
    app.dependency_overrides[get_request_manager] = lambda: req_manager_mock
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: task_manager_mock

    orch_mock.rag.check_manifest_validity.return_value = True

    response = client.get("/api/v1/rag/check_manifest")
    assert response.status_code == 200
    assert response.json()["status"] == Status.SUCCESS

    orch_mock.rag.check_manifest_validity.return_value = False
    response = client.get("/api/v1/rag/check_manifest")
    assert response.status_code == 200
    assert response.json()["status"] == Status.ERROR

    app.dependency_overrides = {}


def test_health_check(
    client: TestClient,
    orch_mock: MagicMock,
    req_manager_mock: MagicMock,
    task_manager_mock: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch_mock
    app.dependency_overrides[get_request_manager] = lambda: req_manager_mock
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: task_manager_mock

    orch_mock.rag.health_check.return_value = True

    response = client.get("/api/v1/rag/health_check")
    assert response.status_code == 200
    assert response.json()["status"] == Status.SUCCESS

    orch_mock.rag.health_check.return_value = False
    response = client.get("/api/v1/rag/health_check")
    assert response.status_code == 200
    assert response.json()["status"] == Status.ERROR

    app.dependency_overrides = {}


def test_add_directory(
    client: TestClient,
    orch_mock: MagicMock,
    req_manager_mock: MagicMock,
    task_manager_mock: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch_mock
    app.dependency_overrides[get_request_manager] = lambda: req_manager_mock
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: task_manager_mock

    payload = {"directories": ["/tmp/test"]}
    response = client.post("/api/v1/rag/add_directories", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["result"] == "task123"

    task_manager_mock.create_task.return_value = None
    response = client.post("/api/v1/rag/add_directories", json=payload)
    assert response.status_code == 500

    app.dependency_overrides = {}


def test_remove_directory(
    client: TestClient,
    orch_mock: MagicMock,
    req_manager_mock: MagicMock,
    task_manager_mock: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: orch_mock
    app.dependency_overrides[get_request_manager] = lambda: req_manager_mock
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: task_manager_mock

    payload = {"directories": ["/tmp/test"]}
    response = client.post("/api/v1/rag/remove_directories", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["result"] == "task123"

    task_manager_mock.create_task.return_value = None
    response = client.post("/api/v1/rag/remove_directories", json=payload)
    assert response.status_code == 500

    app.dependency_overrides = {}
