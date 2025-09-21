import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordBearer

from ataraxai import __version__
from ataraxai.gateway.gateway_task_manager import GatewayTaskManager
from ataraxai.gateway.request_manager import RequestManager
from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
    AtaraxAIOrchestratorFactory,
)
from ataraxai.praxis.katalepsis import Katalepsis
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.routes.chain_runner_route.chain_runner import router_chain_runner
from ataraxai.routes.chat_route.chat import router_chat
from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config import (
    router_llama_cpp,
)
from ataraxai.routes.configs_routes.rag_config_route.rag_config_route import (
    router_rag_config,
)
from ataraxai.routes.configs_routes.user_preferences_route.user_preferences import (
    router_user_preferences,
)
from ataraxai.routes.core_ai_service.core_ai_service import (
    router_core_ai_service_config,
)
from ataraxai.routes.dependency_api import (
    get_orchestrator,
    verify_token,
)
from ataraxai.routes.models_manager_route.models_manager import router_models_manager
from ataraxai.routes.rag_route.rag import router_rag
from ataraxai.routes.status import Status, StatusResponse
from ataraxai.routes.vault_route.vault import router_vault

os.environ.setdefault("ENVIRONMENT", "development")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = AtaraxAILogger("ataraxai.praxis.api").get_logger()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.secret_token = secrets.token_hex(16)
    app.state.orchestrator = await AtaraxAIOrchestratorFactory.create_orchestrator()
    app.state.katalepsis_monitor = Katalepsis()
    app.state.request_manager = RequestManager()
    app.state.gateway_task_manager = GatewayTaskManager()
    await app.state.request_manager.start()
    yield
    logger.info("API is shutting down. Closing orchestrator resources.")
    await app.state.orchestrator.shutdown()
    await app.state.request_manager.stop()
    app.state.secret_token = None


app = FastAPI(
    title="AtaraxAI API",
    description="API for the AtaraxAI Local Assistant Engine",
    version=__version__,
    lifespan=lifespan,
)

if ENVIRONMENT == "development":
    allowed_hosts = ["localhost", "127.0.0.1", "test"]
    allow_origins = ["*"]
else:
    app.docs_url = None
    app.redoc_url = None
    allowed_hosts = ["localhost", "127.0.0.1"]
    allow_origins = ["*"]


app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/v1/status", response_model=StatusResponse, dependencies=[Depends(verify_token)]
)
async def get_state(
    orch: AtaraxAIOrchestrator = Depends(get_orchestrator),
) -> StatusResponse:
    state = await orch.get_state()
    return StatusResponse(
        status=Status.SUCCESS,
        message=f"AtaraxAI is currently in state: {state.name}",
    )


@app.get(
    "/v1/health", response_model=StatusResponse, dependencies=[Depends(verify_token)]
)
async def get_health(
    orch: AtaraxAIOrchestrator = Depends(get_orchestrator),
) -> StatusResponse:
    return StatusResponse(
        status=Status.SUCCESS,
        message="AtaraxAI is healthy.",
    )


all_routers = [
    router_vault,
    router_chat,
    router_rag,
    router_models_manager,
    router_user_preferences,
    router_llama_cpp,
    router_rag_config,
    router_core_ai_service_config,
    router_chain_runner,
]

for router in all_routers:
    app.include_router(router, dependencies=[Depends(verify_token)])


def print_connection_info(port: int, token: str | None):
    connection_info = {"port": port, "token": token, "status": "ready"}
    print(json.dumps(connection_info), flush=True)


def find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def main():
    port = find_free_port()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )

    server = uvicorn.Server(config)

    original_startup = server.startup

    async def custom_startup(**kwargs):
        await original_startup()
        token = app.state.secret_token
        print_connection_info(port, token)

    server.startup = custom_startup

    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
