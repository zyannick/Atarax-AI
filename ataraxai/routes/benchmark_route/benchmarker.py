from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
)
from ataraxai.praxis.modules.benchmark.benchmark_queue_manager import (
    BenchmarkQueueManager,
)
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.configs.config_schemas.benchmarker_config_schema import (
    BenchmarkParams,
    QuantizedModelInfo,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
)
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.benchmark_route.benchmark_api_models import BenchmarkJobAPI
from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from ataraxai.routes.status import StatusResponse
from ataraxai.routes.status import TaskStatus as Status

logger = AtaraxAILogger("ataraxai.praxis.benchmark").get_logger()

router_benchmark = APIRouter(prefix="/api/v1/benchmark", tags=["Benchmark"])


async def get_benchmark_queue_manager(
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> BenchmarkQueueManager:
    manager = await orchestrator.get_benchmark_queue_manager()
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark module is not configured or available.",
        )
    return manager


@router_benchmark.post("/start", response_model=StatusResponse)
@handle_api_errors("Start Benchmark Worker", logger=logger)
async def start_benchmarker_worker(
    bqm: Annotated[BenchmarkQueueManager, Depends(get_benchmark_queue_manager)],
) -> StatusResponse:
    await bqm.start_worker()
    return StatusResponse(status=Status.SUCCESS, message="Benchmark worker started.")


@router_benchmark.post("/stop", response_model=StatusResponse)
@handle_api_errors("Stop Benchmark Worker", logger=logger)
async def stop_benchmark_worker(
    bqm: Annotated[BenchmarkQueueManager, Depends(get_benchmark_queue_manager)],
) -> StatusResponse:
    await bqm.stop_worker()
    return StatusResponse(status=Status.SUCCESS, message="Benchmark worker stopped.")


@router_benchmark.get("/status", response_model=StatusResponse)
@handle_api_errors("Get Benchmark Status", logger=logger)
async def get_benchmark_status(
    bqm: Annotated[BenchmarkQueueManager, Depends(get_benchmark_queue_manager)],
) -> StatusResponse:
    is_running = bqm.is_worker_running()
    return StatusResponse(
        status=Status.SUCCESS,
        message="Benchmark worker status retrieved.",
        data={"is_running": is_running, **bqm.get_queue_status()},
    )


@router_benchmark.get("/job/{job_id}", response_model=StatusResponse)
@handle_api_errors("Get Benchmark Job Info", logger=logger)
async def get_benchmarker_job_info(
    job_id: str,
    bqm: Annotated[BenchmarkQueueManager, Depends(get_benchmark_queue_manager)],
) -> StatusResponse:
    job_info = bqm.get_job(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
    return StatusResponse(
        status=Status.SUCCESS,
        message="Benchmarker job info retrieved successfully.",
        data={"job_info": job_info.model_dump()},
    )


@router_benchmark.post("/job/{job_id}/cancel", response_model=StatusResponse)
@handle_api_errors("Cancel Benchmarker Job", logger=logger)
async def cancel_benchmarker_job(
    job_id: str,
    bqm: Annotated[BenchmarkQueueManager, Depends(get_benchmark_queue_manager)],
) -> StatusResponse:
    success = await bqm.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to cancel job with ID {job_id}. It may not be in the queue.",
        )
    return StatusResponse(
        status=Status.SUCCESS,
        message=f"Request to cancel job {job_id} sent.",
    )


@router_benchmark.post("/jobs/clear_completed", response_model=StatusResponse)
@handle_api_errors("Clear Completed Benchmarker Jobs", logger=logger)
async def clear_completed_benchmarker_jobs(
    bqm: Annotated[BenchmarkQueueManager, Depends(get_benchmark_queue_manager)],
) -> StatusResponse:
    cleared_count = bqm.clear_completed_jobs()
    return StatusResponse(
        status=Status.SUCCESS,
        message=f"Cleared {cleared_count} completed jobs from the queue.",
    )


@router_benchmark.post("/jobs/enqueue", response_model=StatusResponse)
@handle_api_errors("Enqueue Benchmarker Job", logger=logger)
async def enqueue_benchmarker_job(
    benchmark_job: BenchmarkJobAPI,
    bqm: Annotated[BenchmarkQueueManager, Depends(get_benchmark_queue_manager)],
) -> StatusResponse:
    job_id = bqm.enqueue_job(
        model_info=QuantizedModelInfo(**benchmark_job.model_info.model_dump()),
        benchmark_params=BenchmarkParams(**benchmark_job.benchmark_params.model_dump()),
        llama_model_params=LlamaModelParams(
            **benchmark_job.llama_model_params.model_dump()
        ),
    )
    return StatusResponse(
        status=Status.SUCCESS,
        message="Benchmark job enqueued successfully.",
        data={"job_id": job_id},
    )
