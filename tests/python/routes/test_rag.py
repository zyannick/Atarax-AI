import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from ataraxai.routes.rag import router_rag
from ataraxai.routes.status import Status
from fastapi import FastAPI


class DummyAddRequest:
    directories = ["dir1", "dir2"]


class DummyRemoveRequest:
    directories = ["dir3", "dir4"]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router_rag)
    return TestClient(app)


@pytest.fixture
def orch_mock():
    orch = MagicMock()
    orch.rag.check_manifest_validity.return_value = True
    return orch


def test_check_manifest_success(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(rag=MagicMock(check_manifest_validity=lambda: True)),
    ):
        response = client.get("/api/v1/rag/check_manifest")
        assert response.status_code == 200
        assert response.json()["status"] == Status.SUCCESS


def test_check_manifest_failure(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(rag=MagicMock(check_manifest_validity=lambda: False)),
    ):
        response = client.get("/api/v1/rag/check_manifest")
        assert response.status_code == 200
        assert response.json()["status"] == Status.ERROR


def test_check_manifest_exception(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(
            rag=MagicMock(
                check_manifest_validity=MagicMock(side_effect=Exception("fail"))
            )
        ),
    ):
        response = client.get("/api/v1/rag/check_manifest")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_rebuild_index_success(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(rag=MagicMock(rebuild_index=MagicMock())),
    ):
        response = client.post("/api/v1/rag/rebuild_index")
        assert response.status_code == 200
        assert response.json()["status"] == Status.SUCCESS


def test_rebuild_index_exception(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(
            rag=MagicMock(rebuild_index=MagicMock(side_effect=Exception("fail")))
        ),
    ):
        response = client.post("/api/v1/rag/rebuild_index")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_scan_and_index_success(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(rag=MagicMock(perform_initial_scan=MagicMock())),
    ):
        response = client.post("/api/v1/rag/scan_and_index")
        assert response.status_code == 200
        assert response.json()["status"] == Status.SUCCESS


def test_scan_and_index_exception(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(
            rag=MagicMock(perform_initial_scan=MagicMock(side_effect=Exception("fail")))
        ),
    ):
        response = client.post("/api/v1/rag/scan_and_index")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_add_directory_success(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(rag=MagicMock(add_watch_directories=MagicMock())),
    ):
        payload = {"directories": ["dir1", "dir2"]}
        response = client.post("/api/v1/rag/add_directories", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == Status.SUCCESS
        assert "dir1" in response.json()["message"]


def test_add_directory_exception(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(
            rag=MagicMock(
                add_watch_directories=MagicMock(side_effect=Exception("fail"))
            )
        ),
    ):
        payload = {"directories": ["dir1", "dir2"]}
        response = client.post("/api/v1/rag/add_directories", json=payload)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_remove_directory_success(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(rag=MagicMock(remove_watch_directories=MagicMock())),
    ):
        payload = {"directories": ["dir3", "dir4"]}
        response = client.post("/api/v1/rag/remove_directories", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == Status.SUCCESS
        assert "dir3" in response.json()["message"]


def test_remove_directory_exception(client):
    with patch(
        "ataraxai.routes.rag.get_unlocked_orchestrator",
        return_value=MagicMock(
            rag=MagicMock(
                remove_watch_directories=MagicMock(side_effect=Exception("fail"))
            )
        ),
    ):
        payload = {"directories": ["dir3", "dir4"]}
        response = client.post("/api/v1/rag/remove_directories", json=payload)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
