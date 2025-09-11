# from fastapi import status
# from fastapi.testclient import TestClient
# from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
# from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config_api_models import (
#     LlamaCPPConfigAPI,
#     LlamaCPPGenerationParamsAPI,
# )
# from ataraxai.routes.models_manager_route.models_manager_api_models import (
#     SearchModelsManifestRequest,
# )
# from ataraxai.routes.status import Status


# def test_llama_cpp_config(module_unlocked_client_with_filled_manifest: TestClient):

#     search_model_request = SearchModelsManifestRequest()
#     response = module_unlocked_client_with_filled_manifest.post(
#         "/api/v1/models_manager/get_model_info_manifest",
#         json=search_model_request.model_dump(mode="json"),
#     )

#     assert (
#         response.status_code == status.HTTP_200_OK
#     ), f"Expected 200 OK, got {response.text}"
#     data = response.json()

#     assert data["status"] == Status.SUCCESS, f"Expected success status, got {data}"
#     assert data["message"] == "Model information retrieved successfully."
#     assert isinstance(data["models"], list)
#     assert len(data["models"]) > 0, "Expected at least one model in the response."

#     selected_model_dict = data["models"][0]
#     selected_model = LlamaCPPModelInfo(**selected_model_dict)
#     llama_cpp_config = LlamaCPPConfigAPI(
#         model_info=selected_model,
#         n_ctx=2048,
#         n_gpu_layers=40,
#         main_gpu=0,
#         tensor_split=False,
#         vocab_only=False,
#         use_map=False,
#         use_mlock=False,
#     )

#     response = module_unlocked_client_with_filled_manifest.put(
#         "/api/v1/llama_cpp_config/update_llama_cpp_config",
#         json=llama_cpp_config.model_dump(mode="json"),
#     )

#     assert (
#         response.status_code == status.HTTP_200_OK
#     ), f"Expected 200 OK, got {response.text}"
#     data = response.json()

#     assert data["status"] == Status.SUCCESS
#     assert data["message"] == "Llama CPP configuration updated successfully."
#     assert "config" in data
#     assert isinstance(data["config"], dict)

#     response = module_unlocked_client_with_filled_manifest.get(
#         "/api/v1/llama_cpp_config/get_llama_cpp_config"
#     )
#     assert (
#         response.status_code == status.HTTP_200_OK
#     ), f"Expected 200 OK, got {response.text}"
#     data = response.json()
#     assert data["status"] == Status.SUCCESS
#     assert data["message"] == "Llama CPP configuration retrieved successfully."
#     assert "config" in data
#     assert isinstance(data["config"], dict)
#     assert data["config"]["model_info"]["local_path"] == selected_model.local_path
#     assert data["config"]["n_ctx"] == 2048
#     assert data["config"]["n_gpu_layers"] == 40
#     assert data["config"]["main_gpu"] == 0
#     assert data["config"]["tensor_split"] is False
#     assert data["config"]["vocab_only"] is False
#     assert data["config"]["use_map"] is False
#     assert data["config"]["use_mlock"] is False


# def test_llama_cpp_generation_params(module_unlocked_client_with_filled_manifest: TestClient):
#     response = module_unlocked_client_with_filled_manifest.get(
#         "/api/v1/llama_cpp_config/get_llama_cpp_generation_params"
#     )
#     assert (
#         response.status_code == status.HTTP_200_OK
#     ), f"Expected 200 OK, got {response.text}"
#     data = response.json()
#     assert data["status"] == Status.SUCCESS
#     assert data["message"] == "Llama CPP generation parameters retrieved successfully."
#     assert "params" in data
#     assert isinstance(data["params"], dict)

#     new_generation_params : LlamaCPPGenerationParamsAPI = LlamaCPPGenerationParamsAPI(
#         n_predict=256,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.9,
#         repeat_penalty=1.2,
#         penalty_last_n=64,
#         penalty_freq=0.8,
#         penalty_present= 1.0,
#         stop_sequences= ["<|endoftext|>"],
#         n_batch=1,
#         n_threads=4,
#     )

#     response = module_unlocked_client_with_filled_manifest.put(
#         "/api/v1/llama_cpp_config/update_llama_cpp_generation_params",
#         json=new_generation_params.model_dump(mode="json"),
#     )

#     assert (
#         response.status_code == status.HTTP_200_OK
#     ), f"Expected 200 OK, got {response.text}"
#     data = response.json()
#     assert data["status"] == Status.SUCCESS
#     assert data["message"] == "Llama CPP generation parameters updated successfully."
#     assert "params" in data
#     assert isinstance(data["params"], dict)
#     assert data["params"]["n_predict"] == 256
#     assert data["params"]["temperature"] == 0.7
#     assert data["params"]["top_k"] == 50
#     assert data["params"]["top_p"] == 0.9
#     assert data["params"]["repeat_penalty"] == 1.2
#     assert data["params"]["penalty_last_n"] == 64
#     assert data["params"]["penalty_freq"] == 0.8
#     assert data["params"]["penalty_present"] == 1.0
#     assert data["params"]["stop_sequences"] == ["<|endoftext|>"]
#     assert data["params"]["n_batch"] == 1
#     assert data["params"]["n_threads"] == 4