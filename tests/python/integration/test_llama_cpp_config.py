import random
import time
from pydantic import SecretStr
import pytest
from fastapi import WebSocketDisconnect, status
import ulid
from typing import Optional
from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadModelRequest,
    DownloadTaskStatus,
    SearchModelsRequest,
)
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import VaultPasswordRequest
from starlette.testclient import WebSocketDenialResponse


@pytest.fixture
def unlocked_client(integration_client):
    """
    Fixture to ensure the vault is unlocked before running tests.
    """
    orchestrator = integration_client.app.state.orchestrator
    assert orchestrator.state == AppState.FIRST_LAUNCH

    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Vault initialized and unlocked."
    assert orchestrator.state == AppState.UNLOCKED
    return integration_client


def test_get_llama_cpp_config(unlocked_client):
    response = unlocked_client.get("/api/v1/llama_cpp_config/get_llama_cpp_config")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Llama CPP configuration retrieved successfully."
    assert "config" in data
    assert isinstance(data["config"], dict)
    

@pytest.fixture
def get_llama_cpp_model_info():
    search_model_request = SearchModelsRequest(
        query="llama", limit=100, filters_tags=[]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    data = response.json()


    # We select the smallest model to download for testing purposes
    # This is to ensure the download completes quickly in CI environments.
    model_to_download = sorted(data["models"], key=lambda x: x["file_size"])[0]

    download_request = DownloadModelRequest(**model_to_download)

    orchestrator = unlocked_client.app.state.orchestrator

    response = unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_202_ACCEPTED
    ), f"Expected 202 Accepted, got {response.text}"
    data = response.json()
    task_id = data["task_id"]
    assert task_id is not None, "Task ID should not be None."
    assert data["status"] in [
        DownloadTaskStatus.PENDING.value,
        DownloadTaskStatus.RUNNING.value,
        DownloadTaskStatus.COMPLETED.value,
    ], f"Expected status to be PENDING or RUNNING or COMPLETED, got {data['status']} but {DownloadTaskStatus.PENDING.value}"


    final_status = None
    try:
        # with unlocked_client.websocket_connect(f"ws://test/api/v1/models_manager/download_progress/{task_id}") as websocket:
        with unlocked_client.websocket_connect(
            f"/api/v1/models_manager/download_progress/{task_id}",
            headers={"Host": "test"},
        ) as websocket:
            timeout = time.time() + 180
            while time.time() < timeout:
                message = websocket.receive_json()
                final_status = message
                # print(f"Received message: {message}")
                if str(message.get("status")) in [DownloadTaskStatus.COMPLETED.value, DownloadTaskStatus.FAILED.value] :
                    break
    except WebSocketDisconnect as e:
        pytest.fail(
            "WebSocket connection was unexpectedly disconnected. "
            f"Disconnection details: {e}"
        )
    except WebSocketDenialResponse as e:
        pytest.fail(
            "WebSocket connection was denied by the server. " f"Denial details: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"WebSocket connection failed with an unexpected error: {type(e).__name__} - {e}"
        )
        
    
    
    
def test_update_llama_cpp_config(unlocked_client):
    new_config = {
        "model_name": "llama-3-70b",
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }
    
    response = unlocked_client.put(
        "/api/v1/llama_cpp_config/update_llama_cpp_config",
        json=new_config
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Llama CPP configuration updated successfully."
    assert data["config"]["model_name"] == new_config["model_name"]