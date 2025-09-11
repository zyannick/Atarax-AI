# import pytest
# from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
# from ataraxai.routes.status import Status
# from ataraxai.praxis.utils.app_state import AppState





# async def test_orchestrator_status_first_launch(integration_client):
#     response = integration_client.get("/v1/status")
#     assert response.status_code == 200
#     data = response.json()
#     assert data["status"] == Status.SUCCESS
#     assert data["message"].startswith("AtaraxAI is currently in state:")
#     orchestrator : AtaraxAIOrchestrator = integration_client.app.state.orchestrator
#     state = await orchestrator.get_state()
#     assert state == AppState.FIRST_LAUNCH


# def test_orchestrator_health(integration_client):
#     response = integration_client.get("/v1/health")
#     assert response.status_code == 200
#     data = response.json()
#     assert data["status"] == Status.SUCCESS
#     assert data["message"] == "AtaraxAI is healthy."
    
    
