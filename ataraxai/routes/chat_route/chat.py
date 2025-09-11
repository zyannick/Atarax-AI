from fastapi import APIRouter, HTTPException, status
from fastapi.params import Depends
from typing import Annotated, List
import uuid
from ataraxai.gateway.gateway_task_manager import GatewayTaskManager
from ataraxai.gateway.request_manager import RequestManager, RequestPriority
from ataraxai.routes.chat_route.chat_api_models import (
    CreateProjectRequestAPI,
    ProjectResponseAPI,
    CreateSessionRequestAPI,
    SessionResponseAPI,
    MessageResponseAPI,
)
from ataraxai.routes.status import StatusResponse, TaskStatus as Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator

from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger


logger = AtaraxAILogger("ataraxai.praxis.chat").get_logger()


router_chat = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router_chat.post("/projects", response_model=ProjectResponseAPI)
@katalepsis_monitor.instrument_api("POST")
@handle_api_errors("Create Project", logger=logger)
async def create_new_project(
    project_data: CreateProjectRequestAPI,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    task_routine = chat_manager.create_project(
        name=project_data.name, description=project_data.description
    )
    project = await req_manager.submit_request( # type: ignore
        coro=task_routine, priority=RequestPriority.HIGH
    )
    return ProjectResponseAPI(
        project_id=project.id, name=project.name, description=project.description
    )


@router_chat.delete("/projects/{project_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("DELETE")
@handle_api_errors("Delete Project", logger=logger)
async def delete_project(
    project_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    project = await chat_manager.get_project(project_id)
    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    task_routine = chat_manager.delete_project(project_id)
    future = await req_manager.submit_request( # type: ignore
        coro=task_routine, priority=RequestPriority.HIGH
    )

    task_id = task_manager.create_task(future) # type: ignore

    return StatusResponse(
        status=Status.PENDING,
        message=f"Project {project_id} deletion started.",
        task_id=task_id,
    )


@router_chat.get(("/projects/delete/{task_id}"), response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Project Deletion Status", logger=logger)
async def get_project_deletion_status(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    result = task_manager.get_task_status(task_id)
    if result is None:
        return StatusResponse(
            status=Status.ERROR,
            message=f"Task with ID {task_id} not found.",
            task_id=task_id,
        )
    return StatusResponse(
        status=result.get("status", Status.ERROR),
        message="Project deletion status retrieved.",
        task_id=task_id,
    )


@router_chat.get("/projects/{project_id}", response_model=ProjectResponseAPI)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Project", logger=logger)
async def get_project(
    project_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    task_routine = chat_manager.get_project(project_id)
    project = await req_manager.submit_request( # type: ignore
        coro=task_routine, priority=RequestPriority.HIGH
    )
    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )
    return ProjectResponseAPI(
        project_id=project.id, name=project.name, description=project.description
    )


@router_chat.get("/projects", response_model=List[ProjectResponseAPI])
@katalepsis_monitor.instrument_api("GET")  # type: ignore
@handle_api_errors("List Projects", logger=logger)
async def list_projects(
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    task_routine = chat_manager.list_projects()
    projects = await req_manager.submit_request( # type: ignore
        coro=task_routine, priority=RequestPriority.HIGH
    )
    return [
        ProjectResponseAPI(
            project_id=project.id, name=project.name, description=project.description
        )
        for project in projects
    ]


@router_chat.get(
    "/projects/{project_id}/sessions", response_model=List[SessionResponseAPI]
)
@katalepsis_monitor.instrument_api("GET")  # type: ignore
@handle_api_errors("List Sessions", logger=logger)
async def list_sessions(
    project_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    task_routine = chat_manager.list_sessions(project_id)
    sessions = await req_manager.submit_request( # type: ignore
        coro=task_routine, priority=RequestPriority.HIGH
    )
    return [
        SessionResponseAPI(
            session_id=session.id, title=session.title, project_id=session.project_id
        )
        for session in sessions
    ]


@router_chat.post("/sessions", response_model=SessionResponseAPI)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Create Session", logger=logger)
async def create_session(
    session_data: CreateSessionRequestAPI,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    try:
        await chat_manager.get_project(session_data.project_id)
    except Exception as e:
        logger.error(f"Error retrieving project with ID {session_data.project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    task_routine = chat_manager.create_session(
        project_id=session_data.project_id, title=session_data.title
    )
    session = await req_manager.submit_request( # type: ignore
        coro=task_routine, priority=RequestPriority.HIGH
    )

    return SessionResponseAPI(
        session_id=session.id, title=session.title, project_id=session.project_id
    )


@router_chat.delete("/sessions/{session_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("DELETE")
@handle_api_errors("Delete Session", logger=logger)
async def delete_session(
    session_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    session = await chat_manager.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    task_routine = chat_manager.delete_session(session_id)
    future = await req_manager.submit_request( # type: ignore
        coro=task_routine, priority=RequestPriority.HIGH
    )

    task_id = task_manager.create_task(future) # type: ignore
    return StatusResponse(
        status=Status.PENDING,
        message=f"Session {session_id} deleted successfully.",
        task_id=task_id,
    )


@router_chat.get("/sessions/delete/{task_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Delete Session", logger=logger)
async def get_delete_session_status(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    result = task_manager.get_task_status(task_id)
    if result is None:
        return StatusResponse(
            status=Status.ERROR,
            message=f"Task with ID {task_id} not found.",
            task_id=task_id,
        )
    return StatusResponse(
        status=result.get("status", Status.ERROR),
        message="Session deletion status retrieved.",
        task_id=task_id,
    )


@router_chat.get("/sessions/{session_id}", response_model=SessionResponseAPI)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Session", logger=logger)
async def get_session(
    session_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    session = await chat_manager.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return SessionResponseAPI(
        session_id=session.id, title=session.title, project_id=session.project_id
    )


@router_chat.get(
    "/sessions/{session_id}/messages", response_model=List[MessageResponseAPI]
)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Messages", logger=logger)
async def get_messages(
    session_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
):
    chat_manager = await orch.get_chat_manager()
    session = await chat_manager.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    messages = await chat_manager.get_messages_for_session(session_id=session_id)
    return [MessageResponseAPI.model_validate(msg) for msg in messages]
