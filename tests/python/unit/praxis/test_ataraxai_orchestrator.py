import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.exceptions import AtaraxAIError

from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
    AtaraxAIOrchestratorFactory,
)
from ataraxai.praxis.utils.vault_manager import VaultInitializationStatus, VaultUnlockStatus
import pytest
import asyncio
from unittest import mock
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.app_state import AppState


@pytest.mark.asyncio
async def test_initialize_sets_initialized_and_state(monkeypatch):
    # Arrange
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    # Simulate FIRST_LAUNCH state
    monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(return_value=AppState.FIRST_LAUNCH))
    monkeypatch.setattr(orchestrator, "_initialize_base_components", mock.AsyncMock())

    # Act
    await orchestrator.initialize()

    # Assert
    assert orchestrator._initialized is True
    assert orchestrator._state == AppState.FIRST_LAUNCH
    logger.info.assert_called()

@pytest.mark.asyncio
async def test_initialize_new_vault_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.FIRST_LAUNCH

    services.vault_manager = mock.Mock()
    monkeypatch.setattr(services.vault_manager, "create_and_initialize_vault", mock.Mock())
    monkeypatch.setattr(orchestrator, "_set_state", mock.AsyncMock())
    monkeypatch.setattr(orchestrator, "_initialize_unlocked_services", mock.AsyncMock())

    master_password = mock.Mock()
    result = await orchestrator.initialize_new_vault(master_password)
    assert result == VaultInitializationStatus.SUCCESS

@pytest.mark.asyncio
async def test_initialize_new_vault_wrong_state(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED

    master_password = mock.Mock()
    result = await orchestrator.initialize_new_vault(master_password)
    assert result == VaultInitializationStatus.ALREADY_INITIALIZED

@pytest.mark.asyncio
async def test_unlock_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED

    unlock_result = mock.Mock()
    unlock_result.status = VaultUnlockStatus.SUCCESS
    unlock_result.error = None

    services.vault_manager = mock.Mock()
    monkeypatch.setattr(services.vault_manager, "unlock_vault", mock.Mock(return_value=unlock_result))
    monkeypatch.setattr(orchestrator, "_set_state", mock.AsyncMock())
    monkeypatch.setattr(orchestrator, "_initialize_unlocked_services", mock.AsyncMock())

    password = mock.Mock()
    result = await orchestrator.unlock(password)
    assert result.status == VaultUnlockStatus.SUCCESS

@pytest.mark.asyncio
async def test_unlock_already_unlocked(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED

    password = mock.Mock()
    result = await orchestrator.unlock(password)
    assert result.status == VaultUnlockStatus.ALREADY_UNLOCKED

@pytest.mark.asyncio
async def test_lock_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    services.vault_manager = mock.Mock()
    monkeypatch.setattr(services.vault_manager, "lock", mock.Mock())
    monkeypatch.setattr(orchestrator, "_shutdown_services", mock.AsyncMock())
    monkeypatch.setattr(orchestrator, "_set_state", mock.AsyncMock())

    result = await orchestrator.lock()
    assert result is True

@pytest.mark.asyncio
async def test_run_task_chain_locked(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED

    with pytest.raises(Exception):
        await orchestrator.run_task_chain([], "query")

@pytest.mark.asyncio
async def test_reinitialize_vault_wrong_phrase(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED

    result = await orchestrator.reinitialize_vault("wrong phrase")
    assert result is False

@pytest.mark.asyncio
async def test_get_state(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED

    state = await orchestrator.get_state()
    assert state == AppState.LOCKED


@pytest.mark.asyncio
async def test_reinitialize_vault_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED

    monkeypatch.setattr(orchestrator, "lock", mock.AsyncMock())
    directories_mock = mock.Mock()
    directories_mock.data = Path("/tmp/test_data")
    monkeypatch.setattr(orchestrator, "get_directories", mock.AsyncMock(return_value=directories_mock))
    monkeypatch.setattr(orchestrator, "_reinitialize_vault_manager", mock.AsyncMock())
    monkeypatch.setattr(orchestrator, "_set_state", mock.AsyncMock())
    monkeypatch.setattr(directories_mock, "create_directories", mock.Mock())

    monkeypatch.setattr("shutil.rmtree", mock.Mock())

    result = await orchestrator.reinitialize_vault(orchestrator.EXPECTED_RESET_CONFIRMATION_PHRASE)
    assert result is True

@pytest.mark.asyncio
async def test_get_directories_locked(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED

    # with pytest.raises(Exception):
    #     await orchestrator.get_directories()

# @pytest.mark.asyncio
# async def test_get_rag_manager_locked(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
#     orchestrator._state = AppState.LOCKED

#     with pytest.raises(Exception):
#         await orchestrator.get_rag_manager()

# @pytest.mark.asyncio
# async def test_get_vault_manager_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.vault_manager = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_vault_manager()

# @pytest.mark.asyncio
# async def test_get_config_manager_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.config_manager = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_config_manager()

# @pytest.mark.asyncio
# async def test_get_core_ai_service_manager_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.core_ai_service_manager = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_core_ai_service_manager()

# @pytest.mark.asyncio
# async def test_get_app_config_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.app_config = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_app_config()

# @pytest.mark.asyncio
# async def test_get_chat_context_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.chat_context = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_chat_context()

# @pytest.mark.asyncio
# async def test_get_chat_manager_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.chat_manager = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_chat_manager()

# @pytest.mark.asyncio
# async def test_get_models_manager_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.models_manager = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_models_manager()

# @pytest.mark.asyncio
# async def test_get_task_manager_uninitialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     services.task_manager = None
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     with pytest.raises(RuntimeError):
#         await orchestrator.get_task_manager()

# @pytest.mark.asyncio
# async def test_shutdown_sets_services_and_setup_manager_to_none(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     monkeypatch.setattr(orchestrator, "_shutdown_services", mock.AsyncMock())
#     await orchestrator.shutdown()
#     assert orchestrator.services is None
#     assert orchestrator.setup_manager is None

# def test_ensure_initialized_raises_if_not_initialized():
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
#     orchestrator._initialized = False
#     with pytest.raises(RuntimeError):
#         orchestrator._ensure_initialized()

# def test_ensure_initialized_raises_if_services_none():
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, None, logger)
#     orchestrator._initialized = True
#     with pytest.raises(RuntimeError):
#         orchestrator._ensure_initialized()






