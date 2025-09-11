import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.exceptions import AtaraxAIError, AtaraxAILockError

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
from ataraxai.praxis.utils.exceptions import AtaraxAILockError
from pathlib import Path
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator, AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.vault_manager import VaultInitializationStatus, VaultUnlockStatus, UnlockResult
from ataraxai.praxis.ataraxai_orchestrator import create_orchestrator
from ataraxai.praxis.utils.vault_manager import UnlockResult, VaultUnlockStatus


@pytest.mark.asyncio
async def test_initialize_sets_initialized_and_state(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(return_value=AppState.FIRST_LAUNCH))
    monkeypatch.setattr(orchestrator, "_initialize_base_components", mock.AsyncMock())

    await orchestrator.initialize()

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
async def test_get_rag_manager_locked(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED

    with pytest.raises(Exception):
        await orchestrator.get_rag_manager()

@pytest.mark.asyncio
async def test_get_vault_manager_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(RuntimeError):
        await orchestrator.get_vault_manager()

@pytest.mark.asyncio
async def test_get_config_manager_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.config_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(RuntimeError):
        await orchestrator.get_config_manager()

@pytest.mark.asyncio
async def test_get_core_ai_service_manager_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.core_ai_service_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(RuntimeError):
        await orchestrator.get_core_ai_service_manager()

@pytest.mark.asyncio
async def test_get_app_config_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.app_config = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(RuntimeError):
        await orchestrator.get_app_config()

@pytest.mark.asyncio
async def test_get_chat_context_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.chat_context = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_chat_context()

@pytest.mark.asyncio
async def test_get_chat_manager_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.chat_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_chat_manager()

@pytest.mark.asyncio
async def test_get_models_manager_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.models_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_models_manager()

@pytest.mark.asyncio
async def test_get_task_manager_uninitialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.task_manager = None
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_chain_task_manager()

@pytest.mark.asyncio
async def test_shutdown_sets_services_and_setup_manager_to_none(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    monkeypatch.setattr(orchestrator, "_shutdown_services", mock.AsyncMock())
    await orchestrator.shutdown()
    assert orchestrator.services is None
    assert orchestrator.setup_manager is None

def test_ensure_initialized_raises_if_not_initialized():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._initialized = False
    with pytest.raises(RuntimeError):
        orchestrator._ensure_initialized()

def test_ensure_initialized_raises_if_services_none():
    settings = mock.Mock()
    setup_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, None, logger)
    orchestrator._initialized = True
    with pytest.raises(RuntimeError):
        orchestrator._ensure_initialized()


@pytest.mark.asyncio
async def test_aenter_calls_initialize(monkeypatch):
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
async def test_aexit_calls_shutdown(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(orchestrator, "shutdown", mock.AsyncMock())
    await orchestrator.__aexit__(None, None, None)
    orchestrator.shutdown.assert_awaited_once()

@pytest.mark.asyncio
async def test_set_state_changes_state_and_logs(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED
    await orchestrator._set_state(AppState.UNLOCKED)
    assert orchestrator._state == AppState.UNLOCKED
    logger.info.assert_called()

@pytest.mark.asyncio
async def test_determine_initial_state_vault_exists(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    vault_manager = mock.Mock()
    vault_manager.check_path = "/tmp/test_check"
    services = mock.Mock()
    services.vault_manager = vault_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(Path, "exists", mock.Mock(return_value=True))
    state = await orchestrator._determine_initial_state()
    assert state == AppState.LOCKED

@pytest.mark.asyncio
async def test_determine_initial_state_vault_not_exists(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    vault_manager = mock.Mock()
    vault_manager.check_path = "/tmp/test_check"
    services = mock.Mock()
    services.vault_manager = vault_manager
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(Path, "exists", mock.Mock(return_value=False))
    state = await orchestrator._determine_initial_state()
    assert state == AppState.FIRST_LAUNCH

@pytest.mark.asyncio
async def test_initialize_base_components_first_launch(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.AsyncMock()
    setup_manager.is_first_launch = mock.Mock(return_value=True)
    setup_manager.perform_first_launch_setup = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._initialize_base_components()
    logger.info.assert_called()

@pytest.mark.asyncio
async def test_initialize_base_components_no_setup_manager(monkeypatch):
    settings = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, None, services, logger)
    with pytest.raises(RuntimeError):
        await orchestrator._initialize_base_components()

@pytest.mark.asyncio
async def test_reinitialize_vault_error(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED
    monkeypatch.setattr(orchestrator, "lock", mock.AsyncMock(side_effect=Exception("fail")))
    result = await orchestrator.reinitialize_vault(orchestrator.EXPECTED_RESET_CONFIRMATION_PHRASE)
    assert result is False

@pytest.mark.asyncio
async def test_initialize_unlocked_services_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.initialize = mock.AsyncMock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._initialize_unlocked_services()
    services.initialize.assert_awaited_once()
    logger.info.assert_called()

@pytest.mark.asyncio
async def test_initialize_unlocked_services_error(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.initialize = mock.AsyncMock(side_effect=Exception("fail"))
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    monkeypatch.setattr(orchestrator, "_set_state", mock.AsyncMock())
    monkeypatch.setattr(orchestrator, "lock", mock.AsyncMock())
    with pytest.raises(Exception):
        await orchestrator._initialize_unlocked_services()

@pytest.mark.asyncio
async def test_shutdown_services_calls_shutdown(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.shutdown = mock.AsyncMock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    await orchestrator._shutdown_services()
    services.shutdown.assert_awaited_once()
    logger.info.assert_called()

@pytest.mark.asyncio
async def test_shutdown_services_services_none(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, None, logger)
    await orchestrator._shutdown_services()
    logger.warning.assert_called()

@pytest.mark.asyncio
async def test_shutdown_already_shutdown(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._shutdown = True
    await orchestrator.shutdown()

@pytest.mark.asyncio
async def test_get_task_manager_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.task_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED
    result = await orchestrator.get_chain_task_manager()
    assert result == services.task_manager


@pytest.mark.asyncio
async def test_get_models_manager_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.models_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED
    result = await orchestrator.get_models_manager()
    assert result == services.models_manager

@pytest.mark.asyncio
async def test_get_models_manager_locked(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.models_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED
    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_models_manager()

@pytest.mark.asyncio
async def test_get_chat_context_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.chat_context = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED
    result = await orchestrator.get_chat_context()
    assert result == services.chat_context

@pytest.mark.asyncio
async def test_get_chat_manager_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.chat_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED
    result = await orchestrator.get_chat_manager()
    assert result == services.chat_manager

@pytest.mark.asyncio
async def test_get_rag_manager_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.rag_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED
    result = await orchestrator.get_rag_manager()
    assert result == services.rag_manager

@pytest.mark.asyncio
async def test_get_directories_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.directories = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.UNLOCKED
    result = await orchestrator.get_directories()
    assert result == services.directories

@pytest.mark.asyncio
async def test_get_task_manager_locked(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.task_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED
    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_chain_task_manager()


@pytest.mark.asyncio
async def test_factory_create_orchestrator(monkeypatch, tmp_path):
    mock_orchestrator_instance = mock.AsyncMock()

    with mock.patch(
        "ataraxai.praxis.ataraxai_orchestrator.AtaraxAIOrchestrator",
        return_value=mock_orchestrator_instance
    ) as mock_orchestrator_class:
        mock_app_config = mock.Mock()
        mock_app_config.database_filename = "test.db"
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.AppConfig", mock.Mock(return_value=mock_app_config))
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.AtaraxAISettings", mock.Mock())
        mock_directories = mock.Mock()
        mock_directories.data = tmp_path 
        monkeypatch.setattr(
            "ataraxai.praxis.ataraxai_orchestrator.AppDirectories.create_default", 
            mock.Mock(return_value=mock_directories)
        )
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.AtaraxAILogger", mock.Mock(return_value=mock.Mock(get_logger=mock.Mock(return_value=mock.Mock())))
        )
        monkeypatch.setattr(
        "ataraxai.praxis.ataraxai_orchestrator.SetupManager",
        lambda *args, **kwargs: mock.Mock() 
        )
        monkeypatch.setattr(
            "ataraxai.praxis.ataraxai_orchestrator.VaultManager",
            lambda *args, **kwargs: mock.Mock()
        )
        monkeypatch.setattr(
            "ataraxai.praxis.ataraxai_orchestrator.Services",
            lambda *args, **kwargs: mock.Mock()
        )
        # monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.VaultManager", mock.Mock)
        # monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.SetupManager", mock.Mock)
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.ConfigurationManager", lambda *args, **kwargs: mock.Mock())
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.CoreAIServiceManager", lambda *args, **kwargs: mock.Mock())
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.ChatDatabaseManager", lambda *args, **kwargs: mock.Mock())
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.ChatContextManager", lambda *args, **kwargs: mock.Mock())
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.ChatManager", lambda *args, **kwargs: mock.Mock())
        monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.ModelsManager", lambda *args, **kwargs: mock.Mock())
        # monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.Services", mock.Mock)

        orchestrator = await AtaraxAIOrchestratorFactory.create_orchestrator()

        assert orchestrator is mock_orchestrator_instance

        mock_orchestrator_class.assert_called_once()

        mock_orchestrator_instance.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_orchestrator_context_manager(monkeypatch):
    # Patch factory to return a mock orchestrator
    orchestrator_mock = mock.AsyncMock()
    monkeypatch.setattr("ataraxai.praxis.ataraxai_orchestrator.AtaraxAIOrchestratorFactory.create_orchestrator", mock.AsyncMock(return_value=orchestrator_mock))
    async with create_orchestrator() as orch:
        assert orch == orchestrator_mock
    orchestrator_mock.shutdown.assert_awaited_once()
    # Additional tests for AtaraxAIOrchestrator edge cases and coverage


@pytest.mark.asyncio
async def test_get_vault_manager_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.vault_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_vault_manager()
    assert result == services.vault_manager

@pytest.mark.asyncio
async def test_get_config_manager_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.config_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_config_manager()
    assert result == services.config_manager

@pytest.mark.asyncio
async def test_get_core_ai_service_manager_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.core_ai_service_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_core_ai_service_manager()
    assert result == services.core_ai_service_manager

@pytest.mark.asyncio
async def test_get_app_config_success(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    services.app_config = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    result = await orchestrator.get_app_config()
    assert result == services.app_config

@pytest.mark.asyncio
async def test_get_task_manager_services_none(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, None, logger)
    with pytest.raises(AtaraxAILockError):
        await orchestrator.get_chain_task_manager()

@pytest.mark.asyncio
async def test_unlock_services_none(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, None, logger)
    orchestrator._state = AppState.LOCKED
    password = mock.Mock()
    result = await orchestrator.unlock(password)
    assert result.status == VaultUnlockStatus.ERROR

@pytest.mark.asyncio
async def test_lock_services_none(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, None, logger)
    result = await orchestrator.lock()
    assert result is False


@pytest.mark.asyncio
async def test__set_state_no_change(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._state = AppState.LOCKED
    await orchestrator._set_state(AppState.LOCKED)
    logger.info.assert_not_called()


# @pytest.mark.asyncio
# async def test_initialize_already_initialized(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
#     orchestrator._initialized = True

#     await orchestrator.initialize()
#     logger.warning.assert_called_with("Orchestrator already initialized")

# @pytest.mark.asyncio
# async def test_initialize_first_launch(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(return_value=AppState.FIRST_LAUNCH))
#     monkeypatch.setattr(orchestrator, "_initialize_base_components", mock.AsyncMock())

#     await orchestrator.initialize()
#     assert orchestrator._initialized is True
#     assert orchestrator._state == AppState.FIRST_LAUNCH
#     logger.info.assert_called_with(
#         f"Orchestrator initialized. Current state: {AppState.FIRST_LAUNCH.name}"
#     )

# @pytest.mark.asyncio
# async def test_initialize_not_first_launch(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(return_value=AppState.LOCKED))
#     monkeypatch.setattr(orchestrator, "_initialize_base_components", mock.AsyncMock())

#     await orchestrator.initialize()
#     assert orchestrator._initialized is True
#     assert orchestrator._state == AppState.LOCKED
#     logger.info.assert_called_with(
#         f"Orchestrator initialized. Current state: {AppState.LOCKED.name}"
#     )
#     orchestrator._initialize_base_components.assert_not_awaited()

# @pytest.mark.asyncio
# async def test_initialize_exception(monkeypatch):
#     settings = mock.Mock()
#     setup_manager = mock.Mock()
#     services = mock.Mock()
#     logger = mock.Mock()
#     orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

#     monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(side_effect=Exception("fail")))
#     monkeypatch.setattr(orchestrator, "_set_state", mock.AsyncMock())

#     with pytest.raises(Exception):
#         await orchestrator.initialize()
#     logger.error.assert_called()
#     orchestrator._set_state.assert_awaited_with(AppState.ERROR)


@pytest.mark.asyncio
async def test_initialize_already_initialized(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)
    orchestrator._initialized = True

    await orchestrator.initialize()
    logger.warning.assert_called_with("Orchestrator already initialized")

@pytest.mark.asyncio
async def test_initialize_first_launch(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(return_value=AppState.FIRST_LAUNCH))
    monkeypatch.setattr(orchestrator, "_initialize_base_components", mock.AsyncMock())

    await orchestrator.initialize()
    assert orchestrator._initialized is True
    assert orchestrator._state == AppState.FIRST_LAUNCH
    logger.info.assert_called_with(
        f"Orchestrator initialized. Current state: {AppState.FIRST_LAUNCH.name}"
    )

@pytest.mark.asyncio
async def test_initialize_not_first_launch(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(return_value=AppState.LOCKED))
    monkeypatch.setattr(orchestrator, "_initialize_base_components", mock.AsyncMock())

    await orchestrator.initialize()
    assert orchestrator._initialized is True
    assert orchestrator._state == AppState.LOCKED
    logger.info.assert_called_with(
        f"Orchestrator initialized. Current state: {AppState.LOCKED.name}"
    )
    orchestrator._initialize_base_components.assert_not_awaited()

@pytest.mark.asyncio
async def test_initialize_exception(monkeypatch):
    settings = mock.Mock()
    setup_manager = mock.Mock()
    services = mock.Mock()
    logger = mock.Mock()
    orchestrator = AtaraxAIOrchestrator(settings, setup_manager, services, logger)

    monkeypatch.setattr(orchestrator, "_determine_initial_state", mock.AsyncMock(side_effect=Exception("fail")))
    monkeypatch.setattr(orchestrator, "_set_state", mock.AsyncMock())

    with pytest.raises(Exception):
        await orchestrator.initialize()
    logger.error.assert_called()
    orchestrator._set_state.assert_awaited_with(AppState.ERROR)












