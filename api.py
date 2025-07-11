from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from fastapi import FastAPI, Form
from ataraxai.app_logic.ataraxai_orchestrator import AtaraxAIOrchestrator
from fastapi import FastAPI, HTTPException, BackgroundTasks


class ChatMessageResponse(BaseModel):
    assistant_response: str = Field(..., description="The AI's generated response.")
    session_id: uuid.UUID = Field(..., description="The ID of the chat session.")


class CreateProjectResponse(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str


class CreateSessionResponse(BaseModel):
    session_id: uuid.UUID
    title: str
    project_id: uuid.UUID


app = FastAPI(
    title="AtaraxAI API",
    description="API for the AtaraxAI Local Assistant Engine",
    version="1.0.0",
)

orchestrator = AtaraxAIOrchestrator()


@app.post("/v1/projects", response_model=CreateProjectResponse)
async def create_new_project(
    name: str = Form(..., description="The name of the new project."),
    description: str = Form(..., description="A brief description of the project."),
):
    try:
        project = orchestrator.create_project(
            name=name, description=description
        )

        return CreateProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.delete("/v1/projects/{project_id}", response_model=CreateProjectResponse)
async def delete_project(project_id: uuid.UUID):
    project = orchestrator.db_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator.delete_project(project_id)
    return CreateProjectResponse(
        project_id=project.id, name=project.name, description=project.description
    )


@app.get("/v1/projects/{project_id}", response_model=CreateProjectResponse)
async def get_project(project_id: uuid.UUID):
    project = orchestrator.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return CreateProjectResponse(
        project_id=project.id, name=project.name, description=project.description
    )


@app.get("/v1/projects", response_model=List[CreateProjectResponse])
async def list_projects():
    projects = orchestrator.list_projects()
    return [
        CreateProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
        for project in projects
    ]


@app.get(
    "/v1/projects/{project_id}/sessions", response_model=List[CreateSessionResponse]
)
async def list_sessions(project_id: uuid.UUID):
    sessions = orchestrator.list_sessions(project_id)
    return [
        CreateSessionResponse(
            session_id=session.id, title=session.title, project_id=session.project.id
        )
        for session in sessions
    ]


@app.post("/v1/sessions", response_model=CreateSessionResponse)
async def create_new_session(
    project_id: uuid.UUID = Form(
        ..., description="The ID of the project this session belongs to."
    ),
    title: str = Form(..., description="The initial title for the new chat session."),
):
    try:
        session = orchestrator.create_session(
            project_id=project_id, title=title
        )
        return CreateSessionResponse(
            session_id=session.id, title=session.title, project_id=session.project.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/sessions/{session_id}", response_model=CreateSessionResponse)
async def delete_session(session_id: uuid.UUID):
    session = orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator.delete_session(session_id)
    return CreateSessionResponse(
        session_id=session.id, title=session.title, project_id=session.project.id
    )


@app.post("/v1/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_chat_message(
    session_id: uuid.UUID,
    user_query: str = Form(..., description="The user's message to the AI."),
):
    try:
        chat_chain = [
            {
                "task_id": "standard_chat",
                "inputs": {"user_query": user_query, "session_id": session_id},
            }
        ]

        final_result = orchestrator.run_task_chain(
            chain_definition=chat_chain, initial_user_query=user_query
        )

        return ChatMessageResponse(
            assistant_response=final_result, session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.on_event("shutdown")
def shutdown_event():
    print("API is shutting down. Closing orchestrator resources.")
    orchestrator.shutdown()
