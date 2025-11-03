import time
from typing import Dict

import pytest
import pytest_asyncio
from fastapi import status
from fastapi.testclient import TestClient

from ataraxai.gateway.gateway_task_manager import TaskStatus
from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.routes.chat_route.chat_api_models import (
    CreateProjectRequestAPI,
    CreateSessionRequestAPI,
)
from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config_api_models import (
    LlamaCPPConfigAPI,
    LlamaCPPGenerationParamsAPI,
)
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    SearchModelsManifestRequest,
)
from ataraxai.routes.status import Status


@pytest_asyncio.fixture(scope="function")
async def unlocked_client_with_project_and_session(
    unlocked_client_with_filled_manifest: TestClient,
) -> TestClient:

    search_model_request = SearchModelsManifestRequest()
    response = unlocked_client_with_filled_manifest.post(
        "/api/v1/models_manager/get_model_info_manifest",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert (
        data["status"] == Status.SUCCESS.value
    ), f"Expected success status, got {data}"
    assert data["message"] == "Model information retrieved successfully."
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0, "Expected at least one model in the response."

    selected_model_dict = data["models"][0]
    selected_model = LlamaCPPModelInfo(**selected_model_dict)
    llama_cpp_config = LlamaCPPConfigAPI(
        model_info=selected_model,
        n_ctx=1024,
        n_gpu_layers=40,
        main_gpu=0,
        tensor_split=False,
        vocab_only=False,
        use_map=False,
        use_mlock=False,
    )

    response = unlocked_client_with_filled_manifest.put(
        "/api/v1/llama_cpp_config/update_llama_cpp_config",
        json=llama_cpp_config.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"

    new_generation_params: LlamaCPPGenerationParamsAPI = LlamaCPPGenerationParamsAPI(
        n_predict=256,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repeat_penalty=1.2,
        penalty_last_n=64,
        penalty_freq=0.8,
        penalty_present=1.0,
        stop_sequences=["<|endoftext|>"],
        n_batch=2048,
        n_threads=0,
    )

    response = unlocked_client_with_filled_manifest.put(
        "/api/v1/llama_cpp_config/update_llama_cpp_generation_params",
        json=new_generation_params.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Llama CPP generation parameters updated successfully."

    response = unlocked_client_with_filled_manifest.post(
        "/api/v1/core_ai_service/initialize_core_ai_service"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Core AI Service initialized successfully."

    project_data = CreateProjectRequestAPI(
        name="Test Project",
        description="This is a test project.",
    )
    response = unlocked_client_with_filled_manifest.post(
        "/api/v1/chat/projects",
        json=project_data.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert "project_id" in data, "Expected project_id in the response."
    assert isinstance(data["project_id"], str), "Expected project_id to be a string."
    assert "name" in data, "Expected name in the response."
    assert data["name"] == project_data.name, "Project name does not match."
    assert "description" in data, "Expected description in the response."
    assert (
        data["description"] == project_data.description
    ), "Project description does not match."

    session_data = CreateSessionRequestAPI(
        project_id=data["project_id"],
        title="Test Session",
    )
    response = unlocked_client_with_filled_manifest.post(
        "/api/v1/chat/sessions",
        json=session_data.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert "session_id" in data, "Expected session_id in the response."
    assert isinstance(data["session_id"], str), "Expected session_id to be a string."
    assert data["title"] == "Test Session", "Session title does not match."

    return unlocked_client_with_filled_manifest


@pytest.mark.asyncio
async def test_list_available_tasks(
    unlocked_client_with_project_and_session: TestClient,
) -> None:
    response = unlocked_client_with_project_and_session.get(
        "/api/v1/chain_runner/available_tasks"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"

    data = response.json()
    assert (
        data["status"] == Status.SUCCESS.value
    ), f"Expected success status, got {data}"
    assert data["message"] == "Available tasks retrieved successfully."
    assert isinstance(data["list_available_tasks"], list)
    assert (
        len(data["list_available_tasks"]) > 0
    ), "Expected at least one available task."

    for task in data["list_available_tasks"]:
        assert isinstance(task, Dict), "Expected each task to be a dictionary."
        assert "id" in task, "Expected task to have an 'id' field."
        assert isinstance(task["id"], str), "Expected task 'id' to be a string."
        assert "description" in task, "Expected task to have a 'description' field."
        assert isinstance(
            task["description"], str
        ), "Expected task 'description' to be a string."
        assert (
            "required_inputs" in task
        ), "Expected task to have a 'required_inputs' field."
        assert isinstance(
            task["required_inputs"], list
        ), "Expected task 'required_inputs' to be a list."
        for input_item in task["required_inputs"]:
            assert isinstance(
                input_item, str
            ), "Expected each input item to be a string."
        assert (
            "prompt_template" in task
        ), "Expected task to have a 'prompt_template' field."


def _get_project_and_session_id(
    unlocked_client_with_project_and_session: TestClient,
) -> Dict[str, str]:
    response = unlocked_client_with_project_and_session.get("/api/v1/chat/projects")
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"

    data = response.json()
    projects = data["projects"]
    assert isinstance(projects, list), "Expected a list of projects."
    project_info = projects[0]
    assert "project_id" in project_info, "Expected project_id in the response."
    assert isinstance(
        project_info["project_id"], str
    ), "Expected project_id to be a string."

    response = unlocked_client_with_project_and_session.get(
        f"/api/v1/chat/projects/{project_info['project_id']}/sessions"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"

    session_data = response.json()
    sessions = session_data["sessions"]
    session_info = sessions[0]
    assert "session_id" in session_info, "Expected session_id in the response."
    assert isinstance(
        session_info["session_id"], str
    ), "Expected session_id to be a string."

    return {
        "project_id": project_info["project_id"],
        "session_id": session_info["session_id"],
    }


def _check_chain_runner_progress(
    unlocked_client_with_project_and_session: TestClient,
    task_id: str,
) -> Dict[str, str]:
    max_wait_seconds = 30
    start_time = time.time()
    task_status = None

    while time.time() - start_time < max_wait_seconds:
        status_response = unlocked_client_with_project_and_session.get(
            f"/api/v1/chain_runner/run_chain/{task_id}"
        )
        print(f"Status response: {status_response}")
        if status_response.status_code == 200:
            task_status = status_response.json()
            task_result = task_status.get("result", {})
            current_status = task_result.get("status", TaskStatus.PENDING.value)
            print(f"Task status: {task_status}")
            print(f"Current status: {current_status}")
            if current_status in [
                TaskStatus.SUCCESS.value,
                TaskStatus.ERROR.value,
                TaskStatus.FAILED.value,
            ]:
                break
        time.sleep(1)

    assert task_status is not None, "Test timed out waiting for task to complete."
    assert (
        task_status["status"] == Status.SUCCESS.value
    ), f"Task failed with error: {task_status}"

    return task_status


@pytest.mark.asyncio
async def test_run_chain_chat(
    unlocked_client_with_project_and_session: TestClient,
) -> None:
    project_and_session = _get_project_and_session_id(
        unlocked_client_with_project_and_session
    )
    session_id = project_and_session["session_id"]

    run_chain_request = {
        "chain_definition": [
            {
                "task_id": "standard_chat",
                "inputs": {
                    "user_query": "Who is Marcus Aurelius?",
                    "session_id": session_id,
                },
            }
        ],
        "initial_user_query": "Who is Marcus Aurelius?",
    }

    response = unlocked_client_with_project_and_session.post(
        "/api/v1/chain_runner/run_chain",
        json=run_chain_request,
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    task_id = data["task_id"]

    print(f"Started chain with task ID: {task_id}")

    _check_chain_runner_progress(unlocked_client_with_project_and_session, task_id)

    response = unlocked_client_with_project_and_session.get(
        f"/api/v1/chat/sessions/{session_id}/messages"
    )
    assert response.status_code == status.HTTP_200_OK, f"Expected 200 OK, got {response.text}"

    messages = response.json()
    assert len(messages) >= 2, "Expected at least two messages in the history."

    user_message = messages[0]
    assistant_message = messages[1]

    assert user_message["role"] == "user"
    assert assistant_message["role"] == "assistant"
    assert len(assistant_message["content"]) > 0


@pytest.mark.asyncio
async def test_run_chain_invalid_task(
    unlocked_client_with_project_and_session: TestClient,
) -> None:
    project_and_session = _get_project_and_session_id(
        unlocked_client_with_project_and_session
    )
    session_id = project_and_session["session_id"]

    run_chain_request = {
        "chain_definition": [
            {
                "task_id": "non_existent_task",
                "inputs": {
                    "user_query": "What is the capital of France?",
                    "session_id": session_id,
                },
            }
        ],
        "initial_user_query": "What is the capital of France?",
    }

    # with pytest.raises(ValueError) as exc_info:
    response = unlocked_client_with_project_and_session.post(
        "/api/v1/chain_runner/run_chain",
        json=run_chain_request,
    )

    data = response.json()
    assert response.status_code == status.HTTP_200_OK
    task_id = data["task_id"]

    task_status = _check_chain_runner_progress(
        unlocked_client_with_project_and_session, task_id
    )

    print("-" * 80)
    print(f"Response data: {task_status}")
    assert task_status["status"] == Status.SUCCESS.value
    assert task_status["result"]["result"] is None
    assert task_status["result"]["status"] == TaskStatus.FAILED.value
    assert "Task with id 'non_existent_task' not found." in task_status["result"]["error"]
