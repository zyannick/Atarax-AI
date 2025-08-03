from fastapi import APIRouter, BackgroundTasks
from fastapi.params import Depends
from ataraxai.routes.chain_runner_route.chain_runner_api_models import AvailableTasksResponse, RunChainRequest, RunChainResponse
from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor


logger = AtaraxAILogger("ataraxai.praxis.chain_runner").get_logger()


router_chain_runner = APIRouter(prefix="/api/v1/chain_runner", tags=["Chain Runner"])


@router_chain_runner.get("/available_tasks", response_model=AvailableTasksResponse)
@katalepsis_monitor.instrument_api("GET")  # type: ignore
@handle_api_errors("List Available Tasks", logger=logger)
async def list_available_tasks(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator) # type: ignore
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
    tasks = orch.task_manager.list_available_tasks()
    return AvailableTasksResponse(
        status=Status.SUCCESS,
        message="Available tasks retrieved successfully.",
        list_available_tasks=tasks
    )


@router_chain_runner.post("/run_chain", response_model=RunChainResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Run Chain", logger=logger)
async def run_chain(
    request: RunChainRequest,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> RunChainResponse:
    
    background_tasks.add_task(orch.run_task_chain, request.chain_definition, request.initial_user_query)

    return RunChainResponse(
       status=Status.SUCCESS,
       message="Chain execution started successfully.",
       result=None
   )
