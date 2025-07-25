import asyncio
import json
import uuid
from fastapi import APIRouter, BackgroundTasks
from fastapi.params import Depends
import ulid
from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.routes.models_manager_api.models_manager_api_models import (
    DownloadModelResponse,
    DownloadTaskStatus,
    ModelInfoResponse,
    SearchModelsResponse,
    SearchModelsRequest,
    DownloadModelRequest,
)
from ataraxai.praxis.modules.models_manager.models_manager import (
    ModelInfo,
)
from fastapi import WebSocket, WebSocketDisconnect


logger = AtaraxAILogger("ataraxai.praxis.models_manager").get_logger()


router_models_manager = APIRouter(
    prefix="/api/v1/models_manager", tags=["Models Manager"]
)


@router_models_manager.get("/search_models", response_model=SearchModelsResponse)
@katalepsis_monitor.instrument_api("GET")  # type: ignore
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
async def download_progress_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            try:
                progress_response = await get_download_progress(task_id)

                if progress_response.status in [
                    DownloadTaskStatus.COMPLETED,
                    DownloadTaskStatus.FAILED,
                ]:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "progress": progress_response.percentage,
                                "status": progress_response.status,
                                "message": progress_response.message,
                            }
                        )
                    )
                    break

                await websocket.send_text(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "progress": progress_response.percentage,
                            "status": progress_response.status,
                            "message": progress_response.message,
                        }
                    )
                )

                await asyncio.sleep(1)

            except Exception as e:
                await websocket.send_text(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "progress": 0,
                            "status": DownloadTaskStatus.FAILED,
                            "message": f"Error retrieving progress: {str(e)}",
                        }
                    )
                )
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {str(e)}")
    finally:
        await websocket.close()


@router_models_manager.get("/download_progress/{task_id}")
async def get_download_progress(
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


@router_models_manager.post("/download_model", response_model=DownloadModelResponse)
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
            model_info=ModelInfo(**request.model_dump()),
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
