from fastapi import APIRouter
from fastapi.params import Depends
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
)
from ataraxai.praxis.utils.configs.config_schemas.rag_config_schema import RAGConfig
from ataraxai.praxis.utils.configs.llama_config_manager import LlamaConfigManager
from ataraxai.praxis.utils.configs.rag_config_manager import RAGConfigManager
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager
from ataraxai.praxis.utils.exceptions import ServiceInitializationError
from ataraxai.praxis.utils.services import Services
from ataraxai.routes.configs_routes.rag_config_route.rag_config_api_models import (
    RagConfigAPI,
    RagConfigResponse,
)
from ataraxai.routes.core_ai_service.core_ai_service_api_models import (
    CoreAiServiceInitializationResponse,
)
from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from typing import Dict, Any
from pathlib import Path


logger = AtaraxAILogger("ataraxai.praxis.core_ai_service").get_logger()

router_core_ai_service_config = APIRouter(
    prefix="/api/v1/core_ai_service", tags=["Core AI Service"]
)


def get_config_manager(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> ConfigurationManager:
    return orch.config_manager


def get_llama_config_manager(
    config_manager: ConfigurationManager = Depends(get_config_manager),  # type: ignore
) -> LlamaConfigManager:
    return config_manager.llama_config_manager


def get_core_ai_service_manager(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
    llama_config_manager: LlamaConfigManager = Depends(get_llama_config_manager),  # type: ignore
) -> CoreAIServiceManager:
    llama_cpp_params: LlamaModelParams = llama_config_manager.get_llama_cpp_params()
    if llama_cpp_params.model_info is None:
        raise ServiceInitializationError(
            "Llama CPP model information is not available."
        )
    if llama_cpp_params.model_info.filename is None:
        raise ServiceInitializationError("Llama CPP model filename is not set.")
    model_path = Path(llama_cpp_params.model_info.local_path)
    if not model_path.exists():
        raise ServiceInitializationError(
            f"Llama CPP model file does not exist at {model_path}"
        )
    if not model_path.is_file():
        raise ServiceInitializationError(
            f"Llama CPP model path {model_path} is not a file."
        )
    if not model_path.suffix in [".bin", ".gguf"]:
        raise ServiceInitializationError(
            f"Llama CPP model file {model_path} has an unsupported extension. Supported extensions are .bin and .gguf."
        )
    return orch.core_ai_manager


@router_core_ai_service_config.post(
    "/initialize_core_ai_service", response_model=CoreAiServiceInitializationResponse
)
def initialize_core_ai_service(
    service_manager: CoreAIServiceManager = Depends(get_core_ai_service_manager),  # type: ignore
) -> CoreAiServiceInitializationResponse:
    try:
        service_manager.initialize()
        logger.info("Core AI Service initialized successfully")
        return CoreAiServiceInitializationResponse(
            status=Status.SUCCESS, message="Core AI Service initialized successfully."
        )
        
    except ServiceInitializationError as e:
        logger.error(f"Failed to initialize Core AI Service: {e}")
        return CoreAiServiceInitializationResponse(
            status=Status.ERROR, message=f"Failed to initialize Core AI Service: {e}"
        )


@router_core_ai_service_config.get(
    "/get_core_ai_service_status", response_model=CoreAiServiceInitializationResponse
)
def get_core_ai_service_status(
    service_manager: CoreAIServiceManager = Depends(get_core_ai_service_manager),  # type: ignore
) -> CoreAiServiceInitializationResponse:
    """
    Endpoint to retrieve the current status of the Core AI Service.

    Args:
        service_manager: The Core AI Service manager dependency, injected via FastAPI's Depends.

    Returns:
        CoreAiServiceInitializationResponse: The current status of the Core AI Service.
    """
    configuration_status: Dict[str, Any] = service_manager.get_configuration_status()
    return CoreAiServiceInitializationResponse(
        status=Status.SUCCESS,
        message="Core AI Service status retrieved successfully.",
        # data=configuration_status,
    )
