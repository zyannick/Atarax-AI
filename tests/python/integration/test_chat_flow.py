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
def unlock_client_core_ai_service(
    unlocked_client_with_filled_manifest: TestClient,
):
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
        n_batch=1,
        n_threads=4,
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

    return unlocked_client_with_filled_manifest


def test_create_project_with_invalid_data(unlock_client_core_ai_service: TestClient):
    with pytest.raises(ValidationError, match="Project name cannot be empty."):
        project_data = CreateProjectRequestAPI(
            name="",  # Invalid name
            description="This is a test project with invalid data.",
        )


def _test_project_creation(
    unlock_client_core_ai_service: TestClient,
    project_name: str,
    project_description: str,
):
    project_data = CreateProjectRequestAPI(
        name=project_name,
        description=project_description,
    )
    response = unlock_client_core_ai_service.post(
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

    return data["project_id"]


def test_create_project(unlock_client_core_ai_service: TestClient):
    _test_project_creation(
        unlock_client_core_ai_service,
        project_name="Test Project",
        project_description="This is a test project.",
    )


def test_many_projects_creation(unlock_client_core_ai_service: TestClient):
    nb_of_projects = 10
    for i in range(nb_of_projects):
        _test_project_creation(
            unlock_client_core_ai_service,
            project_name=f"Test Project {i}",
            project_description=f"This is test project number {i}.",
        )

    response = unlock_client_core_ai_service.get("/api/v1/chat/projects")
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert isinstance(data, list), "Expected a list of projects."
    assert len(data) == nb_of_projects, f"Expected 10 projects, got {data}."
    for i, project in enumerate(data):
        project_num = nb_of_projects - i - 1
        assert (
            project["name"] == f"Test Project {project_num}"
        ), f"Project name mismatch for index {i}. --> {project['name']}"
        assert (
            project["description"] == f"This is test project number {project_num}."
        ), f"Project description mismatch for index {i}."


def test_create_project_with_empty_description(
    unlock_client_core_ai_service: TestClient,
):
    with pytest.raises(ValidationError, match="Project description cannot be empty."):
        project_data = CreateProjectRequestAPI(
            name="Test Project with Empty Description",
            description="",  # Empty description
        )


def test_create_project_with_long_name(unlock_client_core_ai_service: TestClient):
    long_name = "A" * 33
    with pytest.raises(ValueError, match="Project name exceeds maximum length."):
        project_data = CreateProjectRequestAPI(
            name=long_name,  # Long name
            description="This project has a very long name.",
        )


def test_create_project_with_long_description(
    unlock_client_core_ai_service: TestClient,
):
    long_description = "B" * 300
    with pytest.raises(ValueError, match="Project description exceeds maximum length."):
        project_data = CreateProjectRequestAPI(
            name="Test Project with Long Description",
            description=long_description,
        )

def _test_create_session(
    unlock_client_core_ai_service: TestClient,
    project_id: uuid.UUID,
    session_title: str,
):
    session_data = CreateSessionRequestAPI(
        project_id=project_id,
        title=session_title,
    )
    response = unlock_client_core_ai_service.post(
        "/api/v1/chat/sessions",
        json=session_data.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert "session_id" in data, "Expected session_id in the response."
    assert isinstance(data["session_id"], str), "Expected session_id to be a string."
    assert data["title"] == session_title, "Session title does not match."
    
    return data["session_id"], project_id

def test_create_session(unlock_client_core_ai_service: TestClient):
    project_id = _test_project_creation(
        unlock_client_core_ai_service,
        project_name="Test Project for Session",
        project_description="This project is for testing session creation.",
    )
    _test_create_session(
        unlock_client_core_ai_service,
        project_id=project_id,
        session_title="Test Session",
    )


def test_create_session_with_non_existent_project(
    unlock_client_core_ai_service: TestClient,
):
    session_data = CreateSessionRequestAPI(
        project_id=uuid.uuid4(),
        title="Test Session with Non-Existent Project",
    )
    response = unlock_client_core_ai_service.post(
        "/api/v1/chat/sessions",
        json=session_data.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"



def test_create_session_with_invalid_data(
    unlock_client_core_ai_service: TestClient,
):
    project_id = _test_project_creation(
        unlock_client_core_ai_service,
        project_name="Test Project for Invalid Session",
        project_description="This project is for testing invalid session creation.",
    )
    with pytest.raises(ValidationError, match="Session title cannot be empty."):
        session_data = CreateSessionRequestAPI(
            project_id=project_id,
            title="",
        )


def test_create_session_with_long_title(
    unlock_client_core_ai_service: TestClient,
):
    project_id = _test_project_creation(
        unlock_client_core_ai_service,
        project_name="Test Project for Long Title",
        project_description="This project is for testing long session title.",
    )
    long_title = "C" * 65
    with pytest.raises(ValueError, match="Session title exceeds maximum length."):
        session_data = CreateSessionRequestAPI(
            project_id=project_id,
            title=long_title,
        )

def test_create_many_sessions(
    unlock_client_core_ai_service: TestClient,
):
    project_id = _test_project_creation(
        unlock_client_core_ai_service,
        project_name="Test Project for Many Sessions",
        project_description="This project is for testing multiple session creation.",
    )
    nb_of_sessions = 10
    for i in range(nb_of_sessions):
        _test_create_session(
            unlock_client_core_ai_service,
            project_id=project_id,
            session_title=f"Test Session {i}",
        )

    response = unlock_client_core_ai_service.get(f"/api/v1/chat/projects/{project_id}/sessions")
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert isinstance(data, list), "Expected a list of sessions."
    assert len(data) == nb_of_sessions, f"Expected {nb_of_sessions} sessions, got {len(data)}."
    for i, session in enumerate(data):
        session_num = nb_of_sessions - i - 1
        assert (
            session["title"] == f"Test Session {session_num}"
        ), f"Session title mismatch for index {i}. --> {session['title']}"
        assert (
            session["project_id"] == str(project_id)
        ), f"Session project_id mismatch for index {i}. --> {session['project_id']}"