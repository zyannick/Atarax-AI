
from fastapi import APIRouter
from fastapi.params import Depends


from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from ataraxai.praxis.utils.configs.config_schemas.user_preferences_schema import (
    UserPreferences,
)
from ataraxai.routes.configs_routes.user_preferences_route.user_preferences_api_models import (
    UserPreferencesResponse,
    UserPreferencesAPI,
)

logger = AtaraxAILogger("ataraxai.praxis.user_preferences").get_logger()


router_user_preferences = APIRouter(
    prefix="/api/v1/user_preferences", tags=["User Preferences"]
)


@router_user_preferences.get("/get_preferences", response_model=UserPreferencesResponse)
@handle_api_errors("Get User Preferences", logger=logger)
async def get_user_preferences(orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)) -> UserPreferencesResponse:  # type: ignore
    """
    Endpoint to retrieve user preferences.

    This endpoint fetches the current user preferences from the orchestrator.
    Returns a dictionary containing the user preferences.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        dict: A dictionary containing the user preferences.

    Raises:
        HTTPException: If an error occurs while retrieving user preferences.
    """
    preferences = orch.user_preferences.preferences
    return UserPreferencesResponse(
        status=Status.SUCCESS,
        message="User preferences retrieved successfully.",
        preferences=UserPreferencesAPI(**preferences.model_dump()),
    )


@router_user_preferences.put(
    "/update_preferences", response_model=UserPreferencesResponse
)
@handle_api_errors("Update User Preferences", logger=logger)
async def update_user_preferences(
    preferences: UserPreferencesAPI, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)  # type: ignore
) -> UserPreferencesResponse:
    """
    Endpoint to update user preferences.

    This endpoint updates the user preferences with the provided data.
    Returns a dictionary containing the status and updated preferences.

    Args:
        preferences (dict): A dictionary containing the user preferences to update.
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        dict: A dictionary containing the status and updated user preferences.

    Raises:
        HTTPException: If an error occurs while updating user preferences.
    """

    orch.user_preferences.update_user_preferences(
        UserPreferences(**preferences.model_dump())
    )
    return UserPreferencesResponse(
        status=Status.SUCCESS,
        message="User preferences updated successfully.",
        preferences=UserPreferencesAPI(
            **orch.user_preferences.preferences.model_dump()
        ),
    )
