from typing import Dict
import uuid
from fastapi import APIRouter
from fastapi.params import Depends
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
from fastapi import status


from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
    GenerationParams,
)
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
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator



@pytest.fixture(scope="function")
def unlocked_client_with_project_and_session(
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

    assert data["status"] == Status.SUCCESS, f"Expected success status, got {data}"
    assert data["message"] == "Model information retrieved successfully."
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0, "Expected at least one model in the response."

    selected_model_dict = data["models"][0]
    selected_model = LlamaCPPModelInfo(**selected_model_dict)
    llama_cpp_config = LlamaCPPConfigAPI(
        model_info=selected_model,
        n_ctx=1024,  # Reduced context size for testing
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
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Llama CPP generation parameters updated successfully."

    response = unlocked_client_with_filled_manifest.post(
        "/api/v1/core_ai_service/initialize_core_ai_service"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
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



def test_list_available_tasks(
    unlocked_client_with_project_and_session: TestClient,
) -> None:
    response = unlocked_client_with_project_and_session.get(
        "/api/v1/chain_runner/available_tasks"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    
    data = response.json()
    assert data["status"] == Status.SUCCESS, f"Expected success status, got {data}"
    assert data["message"] == "Available tasks retrieved successfully."
    assert isinstance(data["list_available_tasks"], list)
    assert len(data["list_available_tasks"]) > 0, "Expected at least one available task."

    for task in data["list_available_tasks"]:
        assert isinstance(task, Dict), "Expected each task to be a dictionary."
        assert "id" in task, "Expected task to have an 'id' field."
        assert isinstance(task["id"], str), "Expected task 'id' to be a string."
        assert "description" in task, "Expected task to have a 'description' field."
        assert isinstance(task["description"], str), "Expected task 'description' to be a string."
        assert "required_inputs" in task, "Expected task to have a 'required_inputs' field."
        assert isinstance(task["required_inputs"], list), "Expected task 'required_inputs' to be a list."
        for input_item in task["required_inputs"]:
            assert isinstance(input_item, str), "Expected each input item to be a string."
        assert "prompt_template" in task, "Expected task to have a 'prompt_template' field."


def _get_project_and_session_id(
    unlocked_client_with_project_and_session: TestClient,
) -> Dict[str, str]:
    response = unlocked_client_with_project_and_session.get(
        "/api/v1/chat/projects"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    
    data = response.json()
    project_info = data[0]
    assert "project_id" in project_info, "Expected project_id in the response."
    assert isinstance(project_info["project_id"], str), "Expected project_id to be a string."

    response = unlocked_client_with_project_and_session.get(
        f"/api/v1/chat/projects/{project_info['project_id']}/sessions"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    
    session_data = response.json()
    session_info = session_data[0]
    assert "session_id" in session_info, "Expected session_id in the response."
    assert isinstance(session_info["session_id"], str), "Expected session_id to be a string."

    return {
        "project_id": project_info["project_id"],
        "session_id": session_info["session_id"]
    }

def test_run_chain_chat(
    unlocked_client_with_project_and_session: TestClient,
) -> None:

    project_and_session = _get_project_and_session_id(unlocked_client_with_project_and_session)
    project_id = project_and_session["project_id"]
    session_id = project_and_session["session_id"]

    run_chain_request = {
        "chain_definition": [
            {
                "task_id": "standard_chat",
                "inputs": {
                    "user_query": "Who is Marcus Aurelius?",
                    "session_id": session_id
                },
            }
        ],
        "initial_user_query": "Who is Marcus Aurelius?"
    }

    response = unlocked_client_with_project_and_session.post(
        "/api/v1/chain_runner/run_chain",
        json=run_chain_request,
    )
    
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    
    data = response.json()
    assert data["status"] == Status.SUCCESS, f"Expected success status, got {data}"
    assert data["message"] == "Chain execution started successfully."
    assert data["result"] is None, "Expected result to be None for background task execution."


    response = unlocked_client_with_project_and_session.get(
        f"/api/v1/chat/sessions/{session_id}/messages"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    
    messages = response.json()
    assert isinstance(messages, list), "Expected messages to be a list."
    assert len(messages) > 0, "Expected at least one message in the chat history."
    
    user_message = messages[0]
    assert user_message["role"] == "user", "Expected the first message to be from the user."
    assert user_message["content"] == "Who is Marcus Aurelius?", "User query does not match."

    assistant_message = messages[1]
    assert assistant_message["role"] == "assistant", "Expected the second message to be from the assistant."
    assert isinstance(assistant_message["content"], str), "Expected assistant response to be a string."
    assert len(assistant_message["content"]) > 0, "Expected assistant response to be non-empty."
    


def test_run_chain_invalid_task(
    unlocked_client_with_project_and_session: TestClient,
) -> None:
    project_and_session = _get_project_and_session_id(unlocked_client_with_project_and_session)
    session_id = project_and_session["session_id"]

    run_chain_request = {
        "chain_definition": [
            {
                "task_id": "non_existent_task",
                "inputs": {
                    "user_query": "What is the capital of France?",
                    "session_id": session_id
                },
            }
        ],
        "initial_user_query": "What is the capital of France?"
    }

    with pytest.raises(ValueError) as exc_info:
        response = unlocked_client_with_project_and_session.post(
            "/api/v1/chain_runner/run_chain",
            json=run_chain_request,
        )

