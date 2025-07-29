import asyncio
import json
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, status
from fastapi.params import Depends
from prometheus_client import Enum
import ulid
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator, get_unlocked_orchestrator_ws
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadModelResponse,
    DownloadTaskStatus,
    ModelInfoResponse,
    SearchModelsResponse,
    SearchModelsRequest,
    DownloadModelRequest,
)
from ataraxai.praxis.modules.models_manager.models_manager import (
    LlamaCPPModelInfo,
)
from fastapi import WebSocket, WebSocketDisconnect


logger = AtaraxAILogger("ataraxai.praxis.models_manager").get_logger()


router_models_manager = APIRouter(
    prefix="/api/v1/models_manager", tags=["Models Manager"]
)


@router_models_manager.post("/search_models", response_model=SearchModelsResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Search Models", logger=logger)
async def search_models(request: SearchModelsRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)) -> SearchModelsResponse:  # type: ignore
    models = orch.models_manager.search_models(
        query=request.query,
        limit=request.limit,
        filter_tags=request.filters_tags,
    )
    if not models:
        return SearchModelsResponse(
            status=Status.SUCCESS,
            message="No models found matching the search criteria.",
            models=[],
        )
    return SearchModelsResponse(
        status=Status.SUCCESS,
        message="Models retrieved successfully.",
        models=[ModelInfoResponse(**model.model_dump()) for model in models],
    )
    
    
@router_models_manager.websocket("/download_progress/{task_id}")
async def download_progress_websocket(
    websocket: WebSocket,
    task_id: str,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator_ws) # type: ignore
):
    await websocket.accept()
    
    try:
        while True:
            status_data : Optional[Dict[Any, Any]]= orch.models_manager.get_download_status(task_id)
            
            if status_data:
                status_data = {
                    k: (v.value if isinstance(v, Enum) else v) #type: ignore
                    for k, v in status_data.items()
                }

            await websocket.send_json(status_data)

            if status_data.get("status") == DownloadTaskStatus.COMPLETED.value: #type: ignore
                break
            
            if status_data.get("status") == DownloadTaskStatus.FAILED.value: #type: ignore
                await websocket.send_json({
                    "task_id": task_id,
                    "status": DownloadTaskStatus.FAILED.value,
                    "message": "Download failed.",
                    "percentage": 0,
                })
                break
            

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print(f"Client disconnected from WebSocket for task: {task_id}")
    finally:
        await websocket.close()



@router_models_manager.get("/download_status/{task_id}")
async def get_download_status(
    task_id: str,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> DownloadModelResponse:
    try:
        progress = orch.models_manager.get_download_status(task_id)
        if not progress:
            return DownloadModelResponse(
                status=DownloadTaskStatus.FAILED,
                message="No download task found with the provided ID.",
                percentage=0,
                task_id=task_id,
            )

        return DownloadModelResponse(
            status=progress.get("status", DownloadTaskStatus.PENDING),
            message=progress.get("message", "No message available."),
            percentage=progress.get("percentage", 0),
            task_id=task_id,
        )
    except Exception as e:
        logger.error(f"Error getting download progress for task {task_id}: {str(e)}")
        return DownloadModelResponse(
            status=DownloadTaskStatus.FAILED,
            message=f"Error retrieving download status: {str(e)}",
            percentage=0,
            task_id=task_id,
        )


@router_models_manager.post(
    "/download_model",
    response_model=DownloadModelResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@handle_api_errors("Download Models", logger=logger)
async def download_model(
    background_tasks: BackgroundTasks,
    request: DownloadModelRequest,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> DownloadModelResponse:
    task_id = str(ulid.ULID())

    try:
        background_tasks.add_task(
            orch.models_manager.start_download_task,
            task_id=task_id,
            model_info=LlamaCPPModelInfo(**request.model_dump()),
        )

        return DownloadModelResponse(
            status=DownloadTaskStatus.PENDING,
            message="Download task has been created.",
            task_id=task_id,
            percentage=0,
        )
    except Exception as e:
        logger.error(f"Error creating download task: {str(e)}")
        return DownloadModelResponse(
            status=DownloadTaskStatus.FAILED,
            message=f"Failed to create download task: {str(e)}",
            task_id=task_id,
            percentage=0,
        )
