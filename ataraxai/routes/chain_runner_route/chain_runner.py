from typing import Annotated

from fastapi import APIRouter, HTTPException
from fastapi.params import Depends

from ataraxai.gateway.gateway_task_manager import GatewayTaskManager
from ataraxai.gateway.request_manager import RequestManager, RequestPriority
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.chain_runner_route.chain_runner_api_models import (
    AvailableTasksResponse,
    RunChainRequest,
    RunChainResponse,
    StartChainResponse,
)
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.routes.status import Status

logger = AtaraxAILogger("ataraxai.praxis.chain_runner").get_logger()


router_chain_runner = APIRouter(prefix="/api/v1/chain_runner", tags=["Chain Runner"])


@router_chain_runner.get("/available_tasks", response_model=AvailableTasksResponse)
@katalepsis_monitor.instrument_api("GET")  # type: ignore
@handle_api_errors("List Available Tasks", logger=logger)
async def list_available_tasks(
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> AvailableTasksResponse:
    """
    Endpoint to list all available tasks that can be executed.

    This endpoint retrieves the list of tasks that are currently available in the system.
    Returns a response containing the status and a list of task IDs.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        AvailableTasksResponse: An object containing the status (SUCCESS or ERROR) and a list of available tasks.
    """
    chain_tasks = await orch.get_chain_task_manager()
    tasks = chain_tasks.list_available_tasks()
    return AvailableTasksResponse(
        status=Status.SUCCESS,
        message="Available tasks retrieved successfully.",
        list_available_tasks=tasks,
    )


@router_chain_runner.post("/run_chain", response_model=StartChainResponse)
async def run_chain(
    request: RunChainRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> StartChainResponse:

    task_coroutine = orch.run_task_chain(
        request.chain_definition, request.initial_user_query
    )

    future = await req_manager.submit_request(  # type: ignore
        coro=task_coroutine, priority=RequestPriority.HIGH
    )

    task_id = task_manager.create_task(future)  # type: ignore

    return StartChainResponse(
        status=Status.SUCCESS,
        message="Chain execution started successfully.",
        task_id=task_id,
    )


@router_chain_runner.get("/run_chain/{task_id}", response_model=RunChainResponse)
async def get_chain_result(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> RunChainResponse:
    result = task_manager.get_task_status(task_id)

    if result is None:
        return RunChainResponse(
            status=Status.ERROR,
            message=f"Task with ID {task_id} not found or not completed.",
            result=None,
        )

    return RunChainResponse(
        status=Status.SUCCESS,
        message=f"Task {task_id} completed successfully.",
        result=result,
    )


@router_chain_runner.delete("/run_chain/{task_id}", status_code=200)
async def cancel_chain_run(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    was_cancelled = task_manager.cancel_task(task_id)

    if not was_cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"Task with ID '{task_id}' not found or is no longer running.",
        )

    return {
        "status": "success",
        "message": f"Cancellation requested for task {task_id}.",
    }
