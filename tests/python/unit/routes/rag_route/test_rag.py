import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api import app
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.routes.rag_route.rag_api_models import (
    DirectoriesToAddRequest,
    DirectoriesToRemoveRequest,
)
from ataraxai.routes.status import Status


@pytest.fixture
def rag_manager_mock():
    rag_mock = MagicMock()
    rag_mock.rebuild_index = AsyncMock(return_value=True)
    rag_mock.check_manifest_validity = AsyncMock(return_value=True)
    rag_mock.health_check = AsyncMock(return_value=True)
    rag_mock.add_watch_directories = AsyncMock(return_value=True)
    rag_mock.remove_watch_directories = AsyncMock(return_value=True)
    return rag_mock


@pytest.fixture
def mock_orchestrator(rag_manager_mock: MagicMock):
    orch = MagicMock()
    orch.get_state = AsyncMock(return_value=AppState.UNLOCKED)
    orch.get_rag_manager = AsyncMock(return_value=rag_manager_mock)
    # orch.rag = rag_manager_mock
    return orch


@pytest.fixture
def mock_req_manager():
    req_manager = MagicMock()

    def create_and_resolve_future(*args, **kwargs):
        future = asyncio.Future()
        future.set_result("mocked_future_result")
        return future

    req_manager.submit_request = AsyncMock(side_effect=create_and_resolve_future)
    return req_manager


@pytest.fixture
def mock_task_manager():
    task_manager = MagicMock()
    task_manager.create_task = MagicMock(return_value="task123")
    task_manager.get_task_status = MagicMock(
        return_value={"status": "SUCCESS", "result": "done"}
    )
    task_manager.cancel_task = MagicMock(return_value=True)
    return task_manager


@pytest.fixture
def setup_dependencies(
    mock_orchestrator: MagicMock,
    mock_req_manager: MagicMock,
    mock_task_manager: MagicMock,
):
    app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator
    app.dependency_overrides[get_request_manager] = lambda: mock_req_manager
    app.dependency_overrides[get_gatewaye_task_manager] = lambda: mock_task_manager

    yield

    app.dependency_overrides = {}


class TestRagRebuildIndex:

    @pytest.mark.asyncio
    async def test_rebuild_index_success(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_orchestrator: MagicMock,
        rag_manager_mock: MagicMock,
    ):
        response = client.post("/api/v1/rag/rebuild_index")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS.value
        assert data["result"] == "task123"

    @pytest.mark.asyncio
    async def test_get_rebuild_index_result_success(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        response = client.get("/api/v1/rag/rebuild_index/task123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS.value
        assert data["result"]["result"] == "done"

        mock_task_manager.get_task_status.assert_called_once_with("task123")

    @pytest.mark.asyncio
    async def test_get_rebuild_index_result_not_found(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        mock_task_manager.get_task_status.return_value = None

        response = client.get("/api/v1/rag/rebuild_index/unknown")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.ERROR.value

    @pytest.mark.asyncio
    async def test_cancel_rebuild_index_success(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        response = client.delete("/api/v1/rag/rebuild_index/task123")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS.value

        mock_task_manager.cancel_task.assert_called_once_with("task123")

    @pytest.mark.asyncio
    async def test_cancel_rebuild_index_failure(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        mock_task_manager.cancel_task.return_value = False

        response = client.delete("/api/v1/rag/rebuild_index/unknown")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestRagManifestAndHealth:

    @pytest.mark.asyncio
    async def test_check_manifest_valid(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        rag_manager_mock: MagicMock,
    ):
        rag_manager_mock.check_manifest_validity.return_value = True

        response = client.get("/api/v1/rag/check_manifest")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS.value

    @pytest.mark.asyncio
    async def test_check_manifest_invalid(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        rag_manager_mock: MagicMock,
    ):
        rag_manager_mock.check_manifest_validity.return_value = False

        response = client.get("/api/v1/rag/check_manifest")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.ERROR.value, f"Expected ERROR but got {data}"

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        rag_manager_mock: MagicMock,
    ):
        rag_manager_mock.health_check.return_value = True

        response = client.get("/api/v1/rag/health_check")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS.value

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        rag_manager_mock: MagicMock,
    ):
        rag_manager_mock.health_check.return_value = False

        response = client.get("/api/v1/rag/health_check")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.ERROR.value


class TestRagDirectoryManagement:

    @pytest.mark.asyncio
    async def test_add_directory_success(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        directory_request = DirectoriesToAddRequest(directories=["/tmp/test"])

        response = client.post(
            "/api/v1/rag/add_directories",
            json=directory_request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS.value
        assert data["result"] == "task123"

        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_directory_task_creation_failure(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        mock_task_manager.create_task.return_value = None
        directory_request = DirectoriesToAddRequest(directories=["/tmp/test"])

        response = client.post(
            "/api/v1/rag/add_directories",
            json=directory_request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_remove_directory_success(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        directory_request = DirectoriesToRemoveRequest(directories=["/tmp/test"])

        response = client.post(
            "/api/v1/rag/remove_directories",
            json=directory_request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS.value
        assert data["result"] == "task123"

        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_directory_task_creation_failure(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        mock_task_manager: MagicMock,
    ):
        mock_task_manager.create_task.return_value = None
        directory_request = DirectoriesToRemoveRequest(directories=["/tmp/test"])

        response = client.post(
            "/api/v1/rag/remove_directories",
            json=directory_request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.parametrize(
        "directories,expected_count",
        [
            (["/tmp/test1"], 1),
            (["/tmp/test1", "/tmp/test2"], 2),
            ([], 0),
        ],
    )
    @pytest.mark.asyncio
    async def test_add_directory_multiple_directories(
        self,
        client: TestClient,
        setup_dependencies: MagicMock,
        directories,
        expected_count,
    ):
        directory_request = DirectoriesToAddRequest(directories=directories)

        response = client.post(
            "/api/v1/rag/add_directories",
            json=directory_request.model_dump(mode="json"),
        )

        if expected_count > 0:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == Status.SUCCESS.value
        else:
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
            ]
