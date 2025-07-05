from fastapi import FastAPI, HTTPException,BackgroundTasks
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
        project = orchestrator.db_manager.create_project(
            name=name, description=description
        )

        return CreateProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/v1/sessions", response_model=CreateSessionResponse)
async def create_new_session(
    project_id: uuid.UUID = Form(
        ..., description="The ID of the project this session belongs to."
    ),
    title: str = Form(..., description="The initial title for the new chat session."),
):
    try:
        session = orchestrator.db_manager.create_session(
            project_id=project_id, title=title
        )
        return CreateSessionResponse(
            session_id=session.id, title=session.title, project_id=session.project.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

