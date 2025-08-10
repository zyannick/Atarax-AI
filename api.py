from fastapi import  FastAPI
from contextlib import asynccontextmanager
from fastapi.params import Depends
import os
from ataraxai.gateway.request_manager import RequestManager
from ataraxai.gateway.task_manager import TaskManager
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai import __version__
from ataraxai.routes.status import StatusResponse, Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from ataraxai.praxis.katalepsis import Katalepsis
from ataraxai.routes.rag_route.rag import router_rag
from ataraxai.routes.vault_route.vault import router_vault
from ataraxai.routes.chat_route.chat import router_chat
from ataraxai.routes.models_manager_route.models_manager import router_models_manager
from ataraxai.routes.configs_routes.user_preferences_route.user_preferences import router_user_preferences
from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config import router_llama_cpp
from ataraxai.routes.configs_routes.rag_config_route.rag_config_route import router_rag_config
from ataraxai.routes.core_ai_service.core_ai_service import router_core_ai_service_config
from ataraxai.routes.chain_runner_route.chain_runner import router_chain_runner
from ataraxai.routes.dependency_api import get_orchestrator

os.environ.setdefault("ENVIRONMENT", "development")  # Default to development if not set
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = AtaraxAILogger("ataraxai.praxis.api").get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.orchestrator = AtaraxAIOrchestratorFactory.create_orchestrator()
    app.state.katalepsis_monitor = Katalepsis()
    app.state.request_manager = RequestManager()
    app.state.task_manager = TaskManager()
    await app.state.request_manager.start()
    yield
    logger.info("API is shutting down. Closing orchestrator resources.")
    app.state.orchestrator.shutdown()
    await app.state.request_manager.stop() 


if ENVIRONMENT == "development":
    app = FastAPI(
        title="AtaraxAI API",
        description="API for the AtaraxAI Local Assistant Engine",
        version=__version__,
        lifespan=lifespan,
    )
    allowed_hosts = ["localhost", "127.0.0.1", "test"]
    allow_origins=["http://localhost:3000", "http://test"]
else:
    app = FastAPI(
        title="AtaraxAI API",
        description="API for the AtaraxAI Local Assistant Engine",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    allowed_hosts = ["localhost", "127.0.0.1"]
    allow_origins=["http://localhost:3000"]



app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/status", response_model=StatusResponse)
async def get_state(orch: AtaraxAIOrchestrator = Depends(get_orchestrator)) -> StatusResponse:  # type: ignore
    return StatusResponse(
        status=Status.SUCCESS,
        message=f"AtaraxAI is currently in state: {orch.state.name}",
    )


@app.get("/v1/health", response_model=StatusResponse)
async def get_health(orch: AtaraxAIOrchestrator = Depends(get_orchestrator)) -> StatusResponse:  # type: ignore
    return StatusResponse(
        status=Status.SUCCESS,
        message="AtaraxAI is healthy.",
    )


app.include_router(router_vault)
app.include_router(router_chat)
app.include_router(router_rag)
app.include_router(router_models_manager)
app.include_router(router_user_preferences)
app.include_router(router_llama_cpp)
app.include_router(router_rag_config)
app.include_router(router_core_ai_service_config)
app.include_router(router_chain_runner)