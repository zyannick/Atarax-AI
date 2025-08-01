from fastapi import APIRouter
from fastapi.params import Depends
from fastapi.testclient import TestClient
import pytest
from fastapi import status


from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
    GenerationParams,
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


@pytest.fixture
def unlock_client_with_llama_config_set(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    search_model_request = SearchModelsManifestRequest()
    response = module_unlocked_client_with_filled_manifest.post(
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

    response = module_unlocked_client_with_filled_manifest.put(
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

    response = module_unlocked_client_with_filled_manifest.put(
        "/api/v1/llama_cpp_config/update_llama_cpp_generation_params",
        json=new_generation_params.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Llama CPP generation parameters updated successfully."

    return module_unlocked_client_with_filled_manifest


def test_initialize_core_ai_service(unlock_client_with_llama_config_set: TestClient):

    response = unlock_client_with_llama_config_set.post(
        "/api/v1/core_ai_service/initialize_core_ai_service"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Core AI Service initialized successfully."
    assert True, "This test is currently disabled due to initialization issues."
