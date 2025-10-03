from datetime import datetime
import time
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.routes.benchmark_route.benchmark_api_models import (
    BenchmarkJobAPI,
    BenchmarkParamsAPI,
    QuantizedModelInfoAPI,
)
from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config_api_models import (
    LlamaCPPConfigAPI,
    LlamaCPPGenerationParamsAPI,
)
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    SearchModelsManifestRequest,
)
from ataraxai.routes.status import Status


@pytest.mark.asyncio
async def test_start_benchmark_worker(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/start"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Benchmark worker started."


@pytest.mark.asyncio
async def test_stop_benchmark_worker(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/stop"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Benchmark worker stopped."


@pytest.mark.asyncio
async def test_get_empty_benchmark_status(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    response = module_unlocked_client_with_filled_manifest.get(
        "/api/v1/benchmark/status"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert "Benchmark worker status retrieved." in data["message"]
    assert data["data"]["is_worker_running"] is False
    assert data["data"]["queued_count"] == 0
    assert data["data"]["running_count"] == 0
    assert data["data"]["completed_count"] == 0


@pytest.mark.asyncio
async def test_get_benchmark_status_running(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    start_response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/start"
    )
    assert (
        start_response.status_code == status.HTTP_200_OK
    ), f"Response: {start_response.text}"

    response = module_unlocked_client_with_filled_manifest.get(
        "/api/v1/benchmark/status"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert "Benchmark worker status retrieved." in data["message"]
    assert data["data"]["is_worker_running"] is True
    assert data["data"]["queued_count"] == 0
    assert data["data"]["running_count"] == 0
    assert data["data"]["completed_count"] == 0

    stop_response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/stop"
    )
    assert (
        stop_response.status_code == status.HTTP_200_OK
    ), f"Response: {stop_response.text}"
    time.sleep(1)
    data = stop_response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Benchmark worker stopped."


@pytest.mark.asyncio
async def test_get_inexistent_job(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    job_id = str(uuid4())
    response = module_unlocked_client_with_filled_manifest.get(
        f"/api/v1/benchmark/job/{job_id}"
    )
    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Response: {response.text}"
    data = response.json()
    assert data["detail"] == f"Job with ID {job_id} not found."


@pytest.mark.asyncio
async def test_cancel_inexistent_job(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    job_id = str(uuid4())
    response = module_unlocked_client_with_filled_manifest.post(
        f"/api/v1/benchmark/job/{job_id}/cancel"
    )
    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Response: {response.text}"
    data = response.json()
    assert (
        data["detail"]
        == f"Failed to cancel job with ID {job_id}. It may not be in the queue."
    )


@pytest.mark.asyncio
async def test_clear_empty_completed_jobs(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/jobs/clear_completed"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Cleared 0 completed jobs from the queue."




@pytest.mark.asyncio
async def test_benchmark_workflow(
    module_unlocked_client_with_filled_manifest: TestClient,
):
    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/start"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"

    response = module_unlocked_client_with_filled_manifest.get(
        "/api/v1/benchmark/status"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"
    data = response.json()
    assert data["data"]["is_worker_running"] is True
    assert data["data"]["queued_count"] == 0
    assert data["data"]["running_count"] == 0
    assert data["data"]["completed_count"] == 0

    search_model_request = SearchModelsManifestRequest()
    response = module_unlocked_client_with_filled_manifest.post(
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

    benchmark_job: BenchmarkJobAPI = BenchmarkJobAPI(
        id=str(uuid4()),
        model_info=QuantizedModelInfoAPI(
            model_id=selected_model.repo_id,
            local_path=selected_model.local_path,
            quantisation_type=selected_model.quantization,
            last_modified=str(datetime.now().isoformat()),
            size_bytes=selected_model.file_size,
        ),
        benchmark_params=BenchmarkParamsAPI(
            n_gpu_layers=20,
            repetitions=1,
            warmup=True,
            generation_params=LlamaCPPGenerationParamsAPI(
                temperature=0.7,
                top_p=0.9,
            ),
        ),
        llama_model_params=LlamaCPPConfigAPI(model_info=selected_model),
    )

    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/jobs/enqueue",
        json=benchmark_job.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Benchmark job enqueued successfully."
    job_id = data["data"]["job_id"]

    response = module_unlocked_client_with_filled_manifest.get(
        f"/api/v1/benchmark/job/{job_id}"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Benchmarker job info retrieved successfully."
    assert data["data"]["job_info"]["id"] == job_id

    max_wait_time = 60
    poll_interval = 5
    elapsed_time = 0
    job_completed = False

    while elapsed_time < max_wait_time:
        response = module_unlocked_client_with_filled_manifest.get(
        f"/api/v1/benchmark/job/{job_id}"
        )
        assert (
            response.status_code == status.HTTP_200_OK
        ), f"Response: {response.text}"
        data = response.json()
        job_info = data["data"]["job_info"]
        if job_info["status"] == "COMPLETED":
            job_completed = True
            benchmark_result = job_info.get("benchmark_result") 
            assert benchmark_result is not None, "Benchmark result should not be None."
            assert benchmark_result.get("model_id") == selected_model.repo_id
            benchmark_metrics = benchmark_result.get("metrics")
            assert benchmark_metrics is not None, "Benchmark metrics should not be None."
            assert benchmark_metrics.get("load_time_ms") > 0
            break
        time.sleep(poll_interval)
        elapsed_time += poll_interval

    assert job_completed, "Benchmark job did not complete in time."

    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/jobs/clear_completed"
    )
    assert response.status_code == status.HTTP_200_OK, f"Response: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value

    benchmark_job: BenchmarkJobAPI = BenchmarkJobAPI(
        id=str(uuid4()),
        model_info=QuantizedModelInfoAPI(
            model_id=selected_model.repo_id,
            local_path=selected_model.local_path,
            quantisation_type=selected_model.quantization,
            last_modified=str(datetime.now().isoformat()),
            size_bytes=selected_model.file_size,
        ),
        benchmark_params=BenchmarkParamsAPI(
            n_gpu_layers=20,
            repetitions=1,
            warmup=True,
            generation_params=LlamaCPPGenerationParamsAPI(
                temperature=0.7,
                top_p=0.9,
            ),
        ),
        llama_model_params=LlamaCPPConfigAPI(model_info=selected_model),
    )

    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/benchmark/jobs/enqueue",
        json=benchmark_job.model_dump(mode="json"),
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Benchmark job enqueued successfully."
    job_id = data["data"]["job_id"] 

    response = module_unlocked_client_with_filled_manifest.post(
        f"/api/v1/benchmark/job/{job_id}/cancel"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == f"Request to cancel job {job_id} sent."

    time.sleep(2)

    response = module_unlocked_client_with_filled_manifest.get(
        f"/api/v1/benchmark/job/{job_id}"
    )
    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["data"]["job_info"]["id"] == job_id
    assert data["data"]["job_info"]["status"] in ["CANCELLED", "COMPLETED"]

