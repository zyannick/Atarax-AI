from fastapi import  status
from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config_api_models import (
    LlamaCPPConfigAPI,
)
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    SearchModelsManifestRequest,
)
from ataraxai.routes.status import Status

def test_llama_cpp_config(unlocked_client_with_filled_manifest):

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
    llama_cpp_config = LlamaCPPConfigAPI(model_info=selected_model)

    response = unlocked_client_with_filled_manifest.put(
        "/api/v1/llama_cpp_config/update_llama_cpp_config",
        json=llama_cpp_config.model_dump(mode="json"),  
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Llama CPP configuration updated successfully."
    assert "config" in data
    assert isinstance(data["config"], dict)
    
    response = unlocked_client_with_filled_manifest.get(
        "/api/v1/llama_cpp_config/get_llama_cpp_config"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Llama CPP configuration retrieved successfully."
    assert "config" in data
    assert isinstance(data["config"], dict)
    assert data["config"]["model_info"]["local_path"] == selected_model.local_path
