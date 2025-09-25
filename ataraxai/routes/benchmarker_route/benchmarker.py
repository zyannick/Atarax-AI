from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ataraxai.gateway.gateway_task_manager import GatewayTaskManager
from ataraxai.gateway.request_manager import RequestManager
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.praxis.modules.benchmarker.benchmarker import BenchmarkQueueManager
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.configs.config_schemas.benchmarker_config_schema import (
    BenchmarkParams,
    QuantizedModelInfo,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
)
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.benchmarker_route.benchmark_api_models import BenchmarkJobAPI
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.routes.status import StatusResponse
from ataraxai.routes.status import TaskStatus as Status

logger = AtaraxAILogger("ataraxai.praxis.benchmarker").get_logger()


router_benchmarker = APIRouter(prefix="/api/v1/benchmarker", tags=["Benchmarker"])


@router_benchmarker.get("/status", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Benchmarker Status", logger=logger)
async def get_benchmarker_status(
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    request_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StatusResponse:

    benchmark_queue_manager: BenchmarkQueueManager = (
        await orchestrator.get_benchmark_queue_manager()
    )
    if benchmark_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmarker module is not configured.",
        )
    is_running = benchmark_queue_manager.get_status()

    return StatusResponse(
        status=Status.SUCCESS,
        message=(
            "Benchmarker is running." if is_running else "Benchmarker is not running."
        ),
    )


@router_benchmarker.get("/job_status/{job_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Benchmarker Job Status", logger=logger)
async def get_benchmarker_job_status(
    job_id: str,
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    request_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StatusResponse:

    benchmark_queue_manager: BenchmarkQueueManager = (
        await orchestrator.get_benchmark_queue_manager()
    )
    if benchmark_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmarker module is not configured.",
        )

    job_status = benchmark_queue_manager.get_job_status(job_id)

    return StatusResponse(
        status=Status.SUCCESS,
        message="Benchmarker job status retrieved successfully.",
        data={"job_status": job_status},
    )


@router_benchmarker.get("/job_info/{job_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Benchmarker Job Info", logger=logger)
async def get_benchmarker_job_info(
    job_id: str,
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    request_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StatusResponse:

    benchmark_queue_manager: BenchmarkQueueManager = (
        await orchestrator.get_benchmark_queue_manager()
    )
    if benchmark_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmarker module is not configured.",
        )

    job_info = benchmark_queue_manager.get_job(job_id)

    return StatusResponse(
        status=Status.SUCCESS,
        message="Benchmarker job info retrieved successfully.",
        data={"job_info": job_info},
    )


@router_benchmarker.post("/cancel_job/{job_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("POST")
@handle_api_errors("Cancel Benchmarker Job", logger=logger)
async def cancel_benchmarker_job(
    job_id: str,
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    request_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StatusResponse:

    benchmark_queue_manager: BenchmarkQueueManager = (
        await orchestrator.get_benchmark_queue_manager()
    )
    if benchmark_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmarker module is not configured.",
        )

    success = await benchmark_queue_manager.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to cancel job with ID {job_id}. It may not exist or is already completed.",
        )

    return StatusResponse(
        status=Status.SUCCESS,
        message=f"Benchmarker job with ID {job_id} has been cancelled successfully.",
    )


@router_benchmarker.get("/queue_status", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Benchmarker Queue Status", logger=logger)
async def get_benchmarker_queue_status(
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    request_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StatusResponse:

    benchmark_queue_manager: BenchmarkQueueManager = (
        await orchestrator.get_benchmark_queue_manager()
    )
    if benchmark_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmarker module is not configured.",
        )

    queue_status = benchmark_queue_manager.get_queue_status()

    return StatusResponse(
        status=Status.SUCCESS,
        message="Benchmarker queue status retrieved successfully.",
        data={"queue_status": queue_status},
    )


@router_benchmarker.post("clear_completed_jobs", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("POST")
@handle_api_errors("Clear Completed Benchmarker Jobs", logger=logger)
async def clear_completed_benchmarker_jobs(
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    request_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StatusResponse:

    benchmark_queue_manager: BenchmarkQueueManager = (
        await orchestrator.get_benchmark_queue_manager()
    )
    if benchmark_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmarker module is not configured.",
        )

    cleared_count = benchmark_queue_manager.clear_completed_jobs()

    return StatusResponse(
        status=Status.SUCCESS,
        message=f"Cleared {cleared_count} completed benchmark jobs from the queue.",
    )


@router_benchmarker.post("/enqueue_job", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("POST")
@handle_api_errors("Enqueue Benchmarker Job", logger=logger)
async def enqueue_benchmarker_job(
    benchmark_job: BenchmarkJobAPI,
    orchestrator: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    request_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StatusResponse:

    benchmark_queue_manager: BenchmarkQueueManager = (
        await orchestrator.get_benchmark_queue_manager()
    )
    if benchmark_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmarker module is not configured.",
        )

    enqueued_job_id = benchmark_queue_manager.enqueue_job(
        model_info=QuantizedModelInfo(**benchmark_job.model_info.model_dump()),
        benchmark_params=BenchmarkParams(**benchmark_job.benchmark_params.model_dump()),
        llama_model_params=LlamaModelParams(
            **benchmark_job.llama_model_params.model_dump()
        ),
    )

    return StatusResponse(
        status=Status.SUCCESS,
        message="Benchmarker job enqueued successfully.",
        data={"job_id": enqueued_job_id},
    )
