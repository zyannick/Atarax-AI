from typing import Any, Dict
from fastapi import WebSocketDisconnect, status
from fastapi.testclient import TestClient
import pytest
from ataraxai.routes.status import Status
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadTaskStatus,
)
from starlette.testclient import WebSocketDenialResponse

import time

DOWNLOAD_TIMEOUT = 180
MAX_MODELS_TO_DOWNLOAD = 2
TEST_PASSWORD = "Saturate-Heave8-Unfasten-Squealing"
SEARCH_LIMIT = 100


def clean_downloaded_models(modified_client: TestClient):
    response = modified_client.post("/api/v1/models_manager/remove_all_models")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Model manifests removed successfully."

def monitor_download_progress(modified_client: TestClient, task_id: str, timeout: int = DOWNLOAD_TIMEOUT):
    orchestrator = modified_client.app.state.orchestrator # type: ignore
    final_status = None
    
    try:
        print(f"Attempting WebSocket connection for task_id: {task_id}")
        print(f"Orchestrator state: {orchestrator.state}") # type: ignore
        with modified_client.websocket_connect(
            f"/api/v1/models_manager/download_progress/{task_id}",
            headers={"Host": "test"},
        ) as websocket:
            timeout_time = time.time() + timeout
            while time.time() < timeout_time:
                message = websocket.receive_json()
                final_status = message
                
                if int(message.get("status")) in [
                    int(DownloadTaskStatus.COMPLETED.value),
                    int(DownloadTaskStatus.FAILED.value)
                ]:
                    break
                    
    except WebSocketDenialResponse as e:
        task_check = modified_client.get(f"/api/v1/models_manager/download_status/{task_id}")
        print(
            f"Task status check: {task_check.status_code}, "
            f"{task_check.json() if task_check.status_code == 200 else task_check.text}"
        )
        pytest.fail(
            "WebSocket connection was denied by the server. "
            f"Denial details: {e}"
        )
    except WebSocketDisconnect as e:
        pytest.fail(
            "WebSocket connection was unexpectedly disconnected. "
            f"Disconnection details: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"WebSocket connection failed with an unexpected error: "
            f"{type(e).__name__} - {e}"
        )
        
    return final_status


def validate_model_structure(model: Dict[str, Any]) -> None:
    required_fields = ["repo_id", "filename", "local_path", "file_size", "organization"]
    
    for field in required_fields:
        assert field in model, f"Missing field: {field}"
        assert model[field] is not None, f"Field {field} should not be None"
    
    assert model["file_size"] >= 0, "File size should be non-negative"