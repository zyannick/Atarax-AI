import asyncio
from typing import Annotated, Any, Dict, Optional

import ulid
from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.params import Depends
from prometheus_client import Enum

from ataraxai.gateway.request_manager import RequestManager, RequestPriority
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.praxis.modules.models_manager.models_manager import (
    LlamaCPPModelInfo,
)
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import (
    get_request_manager,
    get_unlocked_orchestrator,
    get_unlocked_orchestrator_ws,
)
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadModelRequest,
    DownloadModelResponse,
    DownloadTaskStatus,
    ModelInfoResponse,
    ModelInfoResponsePaginated,
    SearchModelsManifestRequest,
    SearchModelsRequest,
    SearchModelsResponsePaginated,
)
from ataraxai.routes.status import Status, StatusResponse

logger = AtaraxAILogger("ataraxai.praxis.models_manager").get_logger()


router_models_manager = APIRouter(
    prefix="/api/v1/models_manager", tags=["Models Manager"]
)


@router_models_manager.post(
    "/search_models", response_model=SearchModelsResponsePaginated
)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Search Models", logger=logger)
async def search_models(
    request: SearchModelsRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> SearchModelsResponsePaginated:

    models_manager = await orch.get_models_manager()
    models = models_manager.search_models(
        query=request.query,
        limit=request.limit,
        filter_tags=request.filters_tags,
    )
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No models found matching the search criteria.",
        )
    # the pagination will be handled by the frontend
    # for now, we return all models found
    # in the future, we can implement pagination if needed
    return SearchModelsResponsePaginated(
        status=Status.SUCCESS,
        message="Models retrieved successfully.",
        models=[ModelInfoResponse(**model.model_dump()) for model in models],
        total_count=len(models),
        page=1,
        page_size=len(models),
        total_pages=1,
        has_next=False,
        has_previous=False,
    )


async def start_download_in_thread(
    orch: AtaraxAIOrchestrator, model_info: LlamaCPPModelInfo
) -> str:
    task_id = str(ulid.ULID())

    models_manager = await orch.get_models_manager()

    await asyncio.to_thread(
        models_manager.start_download_task, task_id=task_id, model_info=model_info # type: ignore
    )

    return task_id


@router_models_manager.websocket("/download_progress/{task_id}")
async def download_progress_websocket(
    websocket: WebSocket,
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator_ws)],
):
    await websocket.accept()

    models_manager = await orch.get_models_manager()

    try:
        while True:
            status_data: Optional[Dict[Any, Any]] = ( # type: ignore
                models_manager.get_download_status(task_id) # type: ignore
            )

            if status_data:
                status_data = {
                    k: (v.value if isinstance(v, Enum) else v)  # type: ignore
                    for k, v in status_data.items()
                }

            await websocket.send_json(status_data)

            if status_data.get("status") == DownloadTaskStatus.COMPLETED.value:  # type: ignore
                break

            if status_data.get("status") == DownloadTaskStatus.FAILED.value:  # type: ignore
                await websocket.send_json(
                    {
                        "task_id": task_id,
                        "status": DownloadTaskStatus.FAILED.value,
                        "message": "Download failed.",
                        "percentage": 0,
                    }
                )
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print(f"Client disconnected from WebSocket for task: {task_id}")
    finally:
        await websocket.close()


@router_models_manager.get("/download_status/{task_id}")
async def get_download_status(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> DownloadModelResponse:
    models_manager = await orch.get_models_manager()
    progress = models_manager.get_download_status(task_id)
    print(progress)
    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No download task found with the provided ID.",
        )

    return DownloadModelResponse(
        status=progress.get("status", DownloadTaskStatus.PENDING),
        message=progress.get("message", "No message available."),
        percentage=progress.get("percentage", 0),
        task_id=task_id,
    )


@router_models_manager.post("/cancel_download/{task_id}")
async def cancel_download(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> DownloadModelResponse:
    try:
        models_manager = await orch.get_models_manager()
        models_manager.cancel_download(task_id)
        return DownloadModelResponse(
            status=DownloadTaskStatus.CANCELLED,
            message="Download task has been cancelled.",
            task_id=task_id,
            percentage=0,
        )
    except Exception as e:
        logger.error(f"Error cancelling download task {task_id}: {str(e)}")
        return DownloadModelResponse(
            status=DownloadTaskStatus.FAILED,
            message=f"Error cancelling download task: {str(e)}",
            task_id=task_id,
            percentage=0,
        )


@router_models_manager.post(
    "/download_model",
    response_model=DownloadModelResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@handle_api_errors("Download Models", logger=logger)
async def download_model(
    request: DownloadModelRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
):
    task_coroutine = start_download_in_thread(
        orch=orch,
        model_info=LlamaCPPModelInfo(**request.model_dump()),
    )

    future = await req_manager.submit_request( # type: ignore
        coro=task_coroutine, priority=RequestPriority.MEDIUM
    )

    task_id = await future

    return DownloadModelResponse(
        status=DownloadTaskStatus.PENDING,
        message="Download task has been created.",
        task_id=task_id,
        percentage=0,
    )


# @router_models_manager.post(
#     "/download_model",
#     response_model=DownloadModelResponse,
#     status_code=status.HTTP_202_ACCEPTED,
# )
# @handle_api_errors("Download Models", logger=logger)
# async def download_model(
#     background_tasks: BackgroundTasks,
#     request: DownloadModelRequest,
#     orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
# ) -> DownloadModelResponse:


#     try:
#         background_tasks.add_task(
#             orch.models_manager.start_download_task,
#             task_id=task_id,
#             model_info=LlamaCPPModelInfo(**request.model_dump()),
#         )

#         return DownloadModelResponse(
#             status=DownloadTaskStatus.PENDING,
#             message="Download task has been created.",
#             task_id=task_id,
#             percentage=0,
#         )
#     except Exception as e:
#         logger.error(f"Error creating download task: {str(e)}")
#         return DownloadModelResponse(
#             status=DownloadTaskStatus.FAILED,
#             message=f"Failed to create download task: {str(e)}",
#             task_id=task_id,
#             percentage=0,
#         )


@router_models_manager.post(
    "/get_model_info_manifest", response_model=ModelInfoResponsePaginated
)
@handle_api_errors("Get Model Info", logger=logger)
async def get_model_info(
    request: SearchModelsManifestRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> ModelInfoResponsePaginated:
    models_manager = await orch.get_models_manager()
    results = models_manager.get_list_of_models_from_manifest(
        search_infos=request.model_dump(mode="json"),
    )
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No models found matching the search criteria.",
        )
    return ModelInfoResponsePaginated(
        status=Status.SUCCESS,
        message="Model information retrieved successfully.",
        models=[ModelInfoResponse(**model) for model in results],
        total_count=len(results),
        page=1,
        page_size=len(results),
        total_pages=1,
        has_next=False,
        has_previous=False,
    )


@router_models_manager.post("/remove_all_models")
async def remove_all_models(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> StatusResponse:
    try:
        orch.models_manager.remove_all_models()
        return StatusResponse(
            status=Status.SUCCESS,
            message="Model manifests removed successfully.",
        )
    except Exception as e:
        logger.error(f"Error removing model manifests: {str(e)}")
        return StatusResponse(
            status=Status.ERROR,
            message=f"Failed to remove all model manifests: {str(e)}",
        )
