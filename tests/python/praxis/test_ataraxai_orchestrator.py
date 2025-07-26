import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.exceptions import AtaraxAIError

from ataraxai.praxis.ataraxai_orchestrator import (
    AtaraxAIOrchestrator,
    AtaraxAIOrchestratorFactory,
)

@pytest.fixture
def orchestrator():
    # Use the factory to create an orchestrator with mocked dependencies
    with mock.patch("ataraxai.praxis.ataraxai_orchestrator.AppConfig"), \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.AtaraxAILogger") as MockLogger, \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.AtaraxAISettings"), \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.AppDirectories") as MockDirs, \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.VaultManager") as MockVault, \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.SetupManager"), \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.ConfigurationManager"), \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.CoreAIServiceManager"), \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.Services") as MockServices, \
         mock.patch("ataraxai.praxis.ataraxai_orchestrator.UserPreferencesManager"):
        logger = MockLogger().get_logger()
        dirs = MockDirs.create_default.return_value
        dirs.data = Path("/tmp/ataraxai_data")
        dirs.config = Path("/tmp/ataraxai_config")
        vault_manager = MockVault()
        vault_manager.check_path = "/tmp/ataraxai_data/vault.check"
        services = MockServices()
        services.chat_manager = mock.Mock()
        services.rag_manager = mock.Mock()
        services.models_manager = mock.Mock()
        orchestrator = AtaraxAIOrchestratorFactory.create_orchestrator()
        orchestrator.logger = logger
        orchestrator.directories = dirs
        orchestrator.vault_manager = vault_manager
        orchestrator.services = services
        return orchestrator

def test_initial_state_locked(orchestrator):
    orchestrator.vault_manager.check_path = "/tmp/ataraxai_data/vault.check"
    with mock.patch("pathlib.Path.exists", return_value=True):
        state = orchestrator._determine_initial_state()
        assert state == AppState.LOCKED

def test_initial_state_first_launch(orchestrator):
    orchestrator.vault_manager.check_path = "/tmp/ataraxai_data/vault.check"
    with mock.patch("pathlib.Path.exists", return_value=False):
        state = orchestrator._determine_initial_state()
        assert state == AppState.FIRST_LAUNCH

def test_initialize_new_vault_success(orchestrator):
    orchestrator._state = AppState.FIRST_LAUNCH
    orchestrator.vault_manager.create_and_initialize_vault.return_value = None
    orchestrator._initialize_unlocked_services = mock.Mock()
    result = orchestrator.initialize_new_vault(master_password=mock.Mock())
    assert result is True
    assert orchestrator._state == AppState.UNLOCKED
    orchestrator._initialize_unlocked_services.assert_called_once()

def test_initialize_new_vault_wrong_state(orchestrator):
    orchestrator._state = AppState.LOCKED
    result = orchestrator.initialize_new_vault(master_password=mock.Mock())
    assert result is False

def test_initialize_new_vault_exception(orchestrator):
    orchestrator._state = AppState.FIRST_LAUNCH
    orchestrator.vault_manager.create_and_initialize_vault.side_effect = Exception("fail")
    result = orchestrator.initialize_new_vault(master_password=mock.Mock())
    assert result is False
    assert orchestrator._state == AppState.ERROR

def test_reinitialize_vault_wrong_phrase(orchestrator):
    orchestrator._state = AppState.UNLOCKED
    result = orchestrator.reinitialize_vault("wrong phrase")
    assert result is False

def test_reinitialize_vault_wrong_state(orchestrator):
    orchestrator._state = AppState.LOCKED
    result = orchestrator.reinitialize_vault(orchestrator.EXPECTED_RESET_CONFIRMATION_PHRASE)
    assert result is False

def test_reinitialize_vault_success(orchestrator, tmp_path):
    orchestrator._state = AppState.UNLOCKED
    orchestrator.directories.data = tmp_path
    orchestrator.directories.create_directories = mock.Mock()
    orchestrator._init_security_manager = mock.Mock()
    orchestrator.lock = mock.Mock()
    result = orchestrator.reinitialize_vault(orchestrator.EXPECTED_RESET_CONFIRMATION_PHRASE)
    assert result is True
    assert orchestrator._state == AppState.FIRST_LAUNCH
    orchestrator.directories.create_directories.assert_called_once()
    orchestrator._init_security_manager.assert_called_once()

def test_unlock_success(orchestrator):
    orchestrator._state = AppState.LOCKED
    unlock_result = mock.Mock()
    unlock_result.status = orchestrator.vault_manager.unlock_vault.return_value.status = \
        orchestrator.vault_manager.unlock_vault.return_value.status = \
        mock.Mock()
    unlock_result.status = orchestrator.vault_manager.unlock_vault.return_value.status = \
        type("VaultUnlockStatus", (), {"SUCCESS": "SUCCESS"})().SUCCESS
    orchestrator.vault_manager.unlock_vault.return_value.status = "SUCCESS"
    orchestrator._initialize_unlocked_services = mock.Mock()
    result = orchestrator.unlock(password=mock.Mock())
    assert result is True
    assert orchestrator._state == AppState.UNLOCKED
    orchestrator._initialize_unlocked_services.assert_called_once()

def test_unlock_wrong_state(orchestrator):
    orchestrator._state = AppState.UNLOCKED
    result = orchestrator.unlock(password=mock.Mock())
    assert result is True

def test_unlock_fail(orchestrator):
    orchestrator._state = AppState.LOCKED
    orchestrator.vault_manager.unlock_vault.return_value.status = "FAIL"
    result = orchestrator.unlock(password=mock.Mock())
    assert result is False

def test_lock_sets_state_locked(orchestrator):
    orchestrator._state = AppState.UNLOCKED
    orchestrator.vault_manager.lock = mock.Mock()
    orchestrator.shutdown = mock.Mock()
    orchestrator._set_state = mock.Mock()
    orchestrator.lock()
    orchestrator.vault_manager.lock.assert_called_once()
    orchestrator.shutdown.assert_called_once()
    orchestrator._set_state.assert_called_with(AppState.LOCKED)

def test_run_task_chain_delegates(orchestrator):
    orchestrator.services.run_task_chain = mock.Mock(return_value="result")
    chain_def = [{"task": "foo"}]
    result = orchestrator.run_task_chain(chain_def, "query")
    assert result == "result"
    orchestrator.services.run_task_chain.assert_called_once_with(
        chain_definition=chain_def, initial_user_query="query"
    )

def test_chat_property_unlocked(orchestrator):
    orchestrator._state = AppState.UNLOCKED
    orchestrator.services.chat_manager = "chat"
    assert orchestrator.chat == "chat"

def test_chat_property_locked_raises(orchestrator):
    orchestrator._state = AppState.LOCKED
    orchestrator.services = None
    with pytest.raises(AtaraxAIError):
        _ = orchestrator.chat

def test_rag_property_unlocked(orchestrator):
    orchestrator._state = AppState.UNLOCKED
    orchestrator.services.rag_manager = "rag"
    assert orchestrator.rag == "rag"

def test_models_manager_property_unlocked(orchestrator):
    orchestrator._state = AppState.UNLOCKED
    orchestrator.services.models_manager = "models"
    assert orchestrator.models_manager == "models"

def test_user_preferences_property_unlocked(orchestrator):
    orchestrator._state = AppState.UNLOCKED
    orchestrator.services = mock.Mock()
    orchestrator.user_preferences_manager = "prefs"
    assert orchestrator.user_preferences == "prefs"

def test_shutdown_delegates(orchestrator):
    orchestrator.services.shutdown = mock.Mock()
    orchestrator.shutdown()
    orchestrator.services.shutdown.assert_called_once()

def test_context_manager_calls_shutdown(orchestrator):
    orchestrator.shutdown = mock.Mock()
    with orchestrator:
        pass
    orchestrator.shutdown.assert_called_once()


def test_enter_returns_self(orchestrator):
    with orchestrator as ctx:
        assert ctx is orchestrator

def test_exit_calls_shutdown(orchestrator):
    orchestrator.shutdown = mock.Mock()
    orchestrator.__exit__(None, None, None)
    orchestrator.shutdown.assert_called_once()


