from unittest import mock

import pytest

from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
    AtaraxAIOrchestratorFactory,
    OrchestratorStateMachine,
    create_orchestrator,
)
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.exceptions import AtaraxAIError, AtaraxAILockError
from ataraxai.praxis.utils.vault_manager import (
    UnlockResult,
    VaultInitializationStatus,
    VaultUnlockStatus,
)


@pytest.mark.asyncio
async def test_orchestrator_initial_state_locked():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    state = await orchestrator.get_state()
    assert state == AppState.LOCKED


@pytest.mark.asyncio
async def test_orchestrator_initialize_sets_initialized(
    monkeypatch: pytest.MonkeyPatch,
):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock())
    monkeypatch.setattr(orchestrator, "_initialize_base_components", mock.AsyncMock())
    await orchestrator.initialize()
    assert orchestrator._initialized is True


@pytest.mark.asyncio
async def test_initialize_new_vault_success(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    vault_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = vault_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._set_state(AppState.FIRST_LAUNCH)
    monkeypatch.setattr(vault_manager, "create_and_initialize_vault", mock.Mock())
    monkeypatch.setattr(orchestrator, "_initialize_unlocked_services", mock.AsyncMock())
    result = await orchestrator.initialize_new_vault(master_password=mock.Mock())
    assert result == VaultInitializationStatus.SUCCESS


@pytest.mark.asyncio
async def test_initialize_new_vault_wrong_state(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    vault_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = vault_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.initialize_new_vault(master_password=mock.Mock())
    assert result == VaultInitializationStatus.ALREADY_INITIALIZED


@pytest.mark.asyncio
async def test_unlock_success(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    vault_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = vault_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    unlock_result = UnlockResult(status=VaultUnlockStatus.SUCCESS, error=None)
    monkeypatch.setattr(
        vault_manager, "unlock_vault", mock.Mock(return_value=unlock_result)
    )
    monkeypatch.setattr(orchestrator, "_initialize_unlocked_services", mock.AsyncMock())
    result = await orchestrator.unlock(password=mock.Mock())
    assert result.status == VaultUnlockStatus.SUCCESS


@pytest.mark.asyncio
async def test_unlock_already_unlocked(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    vault_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = vault_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.unlock(password=mock.Mock())
    assert result.status == VaultUnlockStatus.ALREADY_UNLOCKED


@pytest.mark.asyncio
async def test_lock_success(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    vault_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = vault_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(vault_manager, "lock", mock.Mock())
    monkeypatch.setattr(orchestrator, "_shutdown_services", mock.AsyncMock())
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.lock()
    assert result is True


@pytest.mark.asyncio
async def test_run_task_chain_locked_raises(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(AtaraxAILockError):
        await orchestrator.run_task_chain(
            chain_definition=[], initial_user_query="test"
        )


@pytest.mark.asyncio
async def test_get_rag_manager_locked_raises():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_rag_manager()


@pytest.mark.asyncio
async def test_reinitialize_vault_wrong_phrase(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.reinitialize_vault("wrong phrase")
    assert result is False


@pytest.mark.asyncio
async def test_reinitialize_vault_wrong_state(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.reinitialize_vault(
        AtaraxAIOrchestrator.EXPECTED_RESET_CONFIRMATION_PHRASE
    )
    assert result is False


@pytest.mark.asyncio
async def test_orchestrator_aenter_calls_initialize(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(orchestrator, "initialize", mock.AsyncMock())
    result = await orchestrator.__aenter__()
    orchestrator.initialize.assert_awaited_once()
    assert result is orchestrator


@pytest.mark.asyncio
async def test_orchestrator_aexit_calls_shutdown(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(orchestrator, "shutdown", mock.AsyncMock())
    await orchestrator.__aexit__(None, None, None)
    orchestrator.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test__set_state_invalid_transition(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(ValueError):
        await orchestrator._set_state(AppState.LOCKED)
    await orchestrator._set_state(AppState.ERROR)
    with pytest.raises(ValueError):
        await orchestrator._set_state(AppState.UNLOCKED)


@pytest.mark.asyncio
async def test_get_models_manager_unlocked(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.models_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.get_models_manager()
    assert result == services.models_manager


@pytest.mark.asyncio
async def test_get_models_manager_locked_raises(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.models_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_models_manager()


@pytest.mark.asyncio
async def test_get_chat_context_unlocked(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.chat_context = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.get_chat_context()
    assert result == services.chat_context


@pytest.mark.asyncio
async def test_get_chat_manager_unlocked(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.chat_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.get_chat_manager()
    assert result == services.chat_manager


@pytest.mark.asyncio
async def test_get_chain_task_manager_unlocked(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.task_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.get_chain_task_manager()
    assert result == services.task_manager


@pytest.mark.asyncio
async def test_get_vault_manager_returns(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_vault_manager()
    assert result == services.vault_manager


@pytest.mark.asyncio
async def test_get_config_manager_returns(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.config_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_config_manager()
    assert result == services.config_manager


@pytest.mark.asyncio
async def test_get_core_ai_service_manager_returns(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.core_ai_service_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_core_ai_service_manager()
    assert result == services.core_ai_service_manager


@pytest.mark.asyncio
async def test_get_app_config_returns(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.app_config = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_app_config()
    assert result == services.app_config


@pytest.mark.asyncio
async def test_get_directories_returns(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.directories = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_directories()
    assert result == services.directories


@pytest.mark.asyncio
async def test_get_user_preferences_manager_returns(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    config_manager = mock.Mock()
    config_manager.preferences_manager = mock.Mock()
    services.config_manager = config_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_user_preferences_manager()
    assert result == config_manager.preferences_manager


@pytest.mark.asyncio
async def test_get_user_preferences_returns(monkeypatch: pytest.MonkeyPatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    config_manager = mock.Mock()
    config_manager.get_user_preferences = mock.Mock(return_value="prefs")
    services.config_manager = config_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_user_preferences()
    assert result == "prefs"


@pytest.mark.asyncio
async def test_get_vault_manager_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator.get_vault_manager()


@pytest.mark.asyncio
async def test_get_config_manager_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.config_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator.get_config_manager()


@pytest.mark.asyncio
async def test_get_core_ai_service_manager_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.core_ai_service_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator.get_core_ai_service_manager()


@pytest.mark.asyncio
async def test_get_app_config_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.app_config = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator.get_app_config()


@pytest.mark.asyncio
async def test_get_directories_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.directories = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator.get_directories()


@pytest.mark.asyncio
async def test_get_user_preferences_manager_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.config_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator.get_user_preferences_manager()


@pytest.mark.asyncio
async def test_get_user_preferences_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.config_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator.get_user_preferences()


@pytest.mark.asyncio
async def test_state_machine_initial_state():
    sm = OrchestratorStateMachine()
    state = await sm.get_state()
    assert state == AppState.LOCKED


@pytest.mark.asyncio
async def test_state_machine_valid_transition():
    sm = OrchestratorStateMachine(initial_state=AppState.LOCKED)
    await sm.transition_to(AppState.FIRST_LAUNCH)
    state = await sm.get_state()
    assert state == AppState.FIRST_LAUNCH


@pytest.mark.asyncio
async def test_state_machine_invalid_transition_raises():
    sm = OrchestratorStateMachine(initial_state=AppState.ERROR)
    with pytest.raises(ValueError):
        await sm.transition_to(AppState.UNLOCKED)


@pytest.mark.asyncio
async def test_state_machine_logging(monkeypatch: pytest.MonkeyPatch):
    sm = OrchestratorStateMachine(initial_state=AppState.LOCKED)
    monkeypatch.setattr("logging.info", mock.Mock())
    await sm.transition_to(AppState.FIRST_LAUNCH)


@pytest.mark.asyncio
async def test_orchestrator_factory_creates_and_initializes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.AppConfig", mock.Mock)
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.AtaraxAISettings", mock.Mock
    )
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.AppDirectories.create_default",
        mock.Mock(return_value=mock.Mock()),
    )
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.AtaraxAILogger",
        mock.Mock(
            return_value=mock.Mock(get_logger=mock.Mock(return_value=mock.Mock()))
        ),
    )
    monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.VaultManager", mock.Mock)
    monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.SetupManager", mock.Mock)
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.ConfigurationManager", mock.Mock
    )
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.CoreAIServiceManager", mock.Mock
    )
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.ChatDatabaseManager", mock.Mock
    )
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.ChatContextManager", mock.Mock
    )
    monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.ChatManager", mock.Mock)
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.BackgroundTaskManager", mock.Mock
    )
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.ModelsManager", mock.Mock
    )
    monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.Services", mock.Mock)
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.AtaraxAIOrchestrator", mock.Mock()
    )
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.asyncio.to_thread", mock.AsyncMock()
    )
    orchestrator_mock = mock.AsyncMock()
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.AtaraxAIOrchestratorFactory.create_orchestrator",
        mock.AsyncMock(return_value=orchestrator_mock),
    )
    result = await AtaraxAIOrchestratorFactory.create_orchestrator()
    assert result == orchestrator_mock


@pytest.mark.asyncio
async def test_create_orchestrator_context_manager(monkeypatch: pytest.MonkeyPatch):
    orchestrator_mock = mock.AsyncMock()
    monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.AtaraxAIOrchestratorFactory.create_orchestrator",
        mock.AsyncMock(return_value=orchestrator_mock),
    )
    async with create_orchestrator() as orch:
        assert orch == orchestrator_mock
    orchestrator_mock.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_get_state_returns_current_state():
    orchestrator = AtaraxAIOrchestrator(
        mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    )
    await orchestrator._set_state(AppState.UNLOCKED)
    state = await orchestrator.get_state()
    assert state == AppState.UNLOCKED


@pytest.mark.asyncio
async def test_orchestrator__set_state_valid_transition():
    orchestrator = AtaraxAIOrchestrator(
        mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    )
    await orchestrator._set_state(AppState.FIRST_LAUNCH)
    state = await orchestrator.get_state()
    assert state == AppState.FIRST_LAUNCH


@pytest.mark.asyncio
async def test_orchestrator__set_state_invalid_transition_raises():
    orchestrator = AtaraxAIOrchestrator(
        mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    )
    await orchestrator._set_state(AppState.ERROR)
    with pytest.raises(ValueError):
        await orchestrator._set_state(AppState.UNLOCKED)


@pytest.mark.asyncio
async def test_orchestrator__ensure_initialized_raises_if_not_initialized():
    orchestrator = AtaraxAIOrchestrator(
        mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    )
    orchestrator._initialized = False
    with pytest.raises(RuntimeError):
        orchestrator._ensure_initialized()


@pytest.mark.asyncio
async def test_orchestrator__ensure_initialized_raises_if_services_none():
    orchestrator = AtaraxAIOrchestrator(mock.Mock(), mock.Mock(), None, mock.Mock())
    orchestrator._initialized = True
    with pytest.raises(RuntimeError):
        orchestrator._ensure_initialized()


@pytest.mark.asyncio
async def test_orchestrator_shutdown_sets_services_and_setup_manager_none(
    monkeypatch: pytest.MonkeyPatch,
):
    orchestrator = AtaraxAIOrchestrator(
        mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    )
    monkeypatch.setattr(orchestrator, "_shutdown_services", mock.AsyncMock())
    orchestrator._shutdown = False
    await orchestrator.shutdown()
    assert orchestrator.services is None
    assert orchestrator.setup_manager is None


@pytest.mark.asyncio
async def test_orchestrator_run_task_chain_unlocked(monkeypatch: pytest.MonkeyPatch):
    services = mock.Mock()
    services.run_task_chain = mock.AsyncMock(return_value="result")
    orchestrator = AtaraxAIOrchestrator(mock.Mock(), mock.Mock(), services, mock.Mock())
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.run_task_chain(
        chain_definition=[], initial_user_query="query"
    )
    assert result == "result"


@pytest.mark.asyncio
async def test_orchestrator_run_task_chain_locked_raises():
    orchestrator = AtaraxAIOrchestrator(
        mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    )
    with pytest.raises(AtaraxAILockError):
        await orchestrator.run_task_chain(
            chain_definition=[], initial_user_query="query"
        )


@pytest.mark.asyncio
async def test_orchestrator_get_rag_manager_unlocked():
    services = mock.Mock()
    services.rag_manager = "rag"
    orchestrator = AtaraxAIOrchestrator(mock.Mock(), mock.Mock(), services, mock.Mock())
    await orchestrator._set_state(AppState.UNLOCKED)
    result = await orchestrator.get_rag_manager()
    assert result == "rag"


@pytest.mark.asyncio
async def test_orchestrator_get_rag_manager_locked_raises():
    orchestrator = AtaraxAIOrchestrator(
        mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    )
    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_rag_manager()
