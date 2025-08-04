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


class TestAtaraxAIOrchestrator:


    @pytest.fixture
    def orchestrator(self, tmp_path : Path):
        
        mock_app_config = mock.MagicMock()
        mock_settings = mock.MagicMock()
        mock_logger = mock.MagicMock()
        
        mock_directories = mock.MagicMock()
        mock_directories.data = tmp_path / "data"
        mock_directories.config = tmp_path / "config"
        
        mock_vault_manager = mock.MagicMock()
        mock_vault_manager.check_path = mock_directories.data / "vault.check"

        mock_setup_manager = mock.MagicMock()
        mock_config_manager = mock.MagicMock()
        mock_core_ai_manager = mock.MagicMock()
        mock_services = mock.MagicMock()

        
        orchestrator_instance = AtaraxAIOrchestrator(
        app_config=mock_app_config,
        settings=mock_settings,
        logger=mock_logger,
        directories=mock_directories,
        vault_manager=mock_vault_manager,
        setup_manager=mock_setup_manager,
        config_manager=mock_config_manager,
        core_ai_manager=mock_core_ai_manager,
        services=mock_services,
        )
    
        return orchestrator_instance

    def test_initial_state_locked_when_vault_exists(self, orchestrator):
        with mock.patch("pathlib.Path.exists", return_value=True):
            state = orchestrator._determine_initial_state()
            assert state == AppState.LOCKED

    def test_initial_state_first_launch_when_no_vault(self, orchestrator):
        with mock.patch("pathlib.Path.exists", return_value=False):
            state = orchestrator._determine_initial_state()
            assert state == AppState.FIRST_LAUNCH

    def test_initialize_new_vault_success(self, orchestrator):
        orchestrator._state = AppState.FIRST_LAUNCH
        orchestrator.vault_manager.create_and_initialize_vault.return_value = None
        orchestrator._initialize_unlocked_services = mock.Mock()
        
        master_password = mock.Mock()
        result = orchestrator.initialize_new_vault(master_password)
        
        assert result is VaultInitializationStatus.SUCCESS
        assert orchestrator._state == AppState.UNLOCKED
        orchestrator.vault_manager.create_and_initialize_vault.assert_called_once_with(
            master_password
        )
        orchestrator._initialize_unlocked_services.assert_called_once()

    def test_initialize_new_vault_fails_when_wrong_state(self, orchestrator):
        orchestrator._state = AppState.LOCKED
        
        result = orchestrator.initialize_new_vault(mock.Mock())
        
        assert result is VaultInitializationStatus.ALREADY_INITIALIZED
        orchestrator.vault_manager.create_and_initialize_vault.assert_not_called()

    def test_initialize_new_vault_handles_exception(self, orchestrator):
        orchestrator._state = AppState.FIRST_LAUNCH
        orchestrator.vault_manager.create_and_initialize_vault.side_effect = Exception("Vault creation failed")
        
        result = orchestrator.initialize_new_vault(master_password=mock.Mock())
        
        assert result is VaultInitializationStatus.FAILED
        assert orchestrator._state == AppState.ERROR

    def test_reinitialize_vault_fails_with_wrong_phrase(self, orchestrator):
        orchestrator._state = AppState.UNLOCKED
        
        result = orchestrator.reinitialize_vault("wrong phrase")
        
        assert result is False

    def test_reinitialize_vault_fails_when_not_unlocked(self, orchestrator):
        orchestrator._state = AppState.LOCKED
        
        result = orchestrator.reinitialize_vault(orchestrator.EXPECTED_RESET_CONFIRMATION_PHRASE)
        
        assert result is False

    def test_reinitialize_vault_success(self, orchestrator, tmp_path):
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
        orchestrator.lock.assert_called_once()

    def test_unlock_success(self, orchestrator):
        orchestrator._state = AppState.LOCKED
        
        unlock_result = mock.Mock()
        unlock_result.status = VaultUnlockStatus.SUCCESS
        orchestrator.vault_manager.unlock_vault.return_value = unlock_result
        orchestrator._initialize_unlocked_services = mock.Mock()
        
        password = mock.Mock()
        result = orchestrator.unlock(password)

        assert result.status is VaultUnlockStatus.SUCCESS
        assert orchestrator._state == AppState.UNLOCKED
        orchestrator.vault_manager.unlock_vault.assert_called_once_with(password)
        orchestrator._initialize_unlocked_services.assert_called_once()

    def test_unlock_when_already_unlocked(self, orchestrator):
        orchestrator._state = AppState.UNLOCKED
        
        result = orchestrator.unlock(password=mock.Mock())

        assert result.status is VaultUnlockStatus.ALREADY_UNLOCKED
        orchestrator.vault_manager.unlock_vault.assert_not_called()

    def test_unlock_failure(self, orchestrator):
        orchestrator._state = AppState.LOCKED
        
        unlock_result = mock.Mock()
        unlock_result.status = VaultUnlockStatus.INVALID_PASSWORD
        orchestrator.vault_manager.unlock_vault.return_value = unlock_result
        
        result = orchestrator.unlock(password=mock.Mock())
        
        assert result.status is VaultUnlockStatus.INVALID_PASSWORD
        assert orchestrator._state == AppState.LOCKED

    def test_lock_transitions_to_locked_state(self, orchestrator):
        orchestrator._state = AppState.UNLOCKED
        orchestrator.vault_manager.lock = mock.Mock()
        orchestrator.shutdown = mock.Mock()
        orchestrator._set_state = mock.Mock()
        
        orchestrator.lock()
        
        orchestrator.vault_manager.lock.assert_called_once()
        orchestrator.shutdown.assert_called_once()
        orchestrator._set_state.assert_called_once_with(AppState.LOCKED)

    def test_run_task_chain_delegates_to_services(self, orchestrator):
        expected_result = {"output": "task completed"}
        orchestrator.services.run_task_chain.return_value = expected_result
        
        chain_def = [{"task": "analyze", "params": {"input": "test"}}]
        query = "Analyze this data"
        
        result = orchestrator.run_task_chain(chain_def, query)
        
        assert result == expected_result
        orchestrator.services.run_task_chain.assert_called_once_with(
            chain_definition=chain_def, 
            initial_user_query=query
        )

    def test_chat_property_when_unlocked(self, orchestrator):
        orchestrator._state = AppState.UNLOCKED
        expected_chat = mock.Mock()
        orchestrator.services.chat_manager = expected_chat
        
        assert orchestrator.chat == expected_chat


    def test_rag_property_when_unlocked(self, orchestrator):
        orchestrator._state = AppState.UNLOCKED
        expected_rag = mock.Mock()
        orchestrator.services.rag_manager = expected_rag
        
        assert orchestrator.rag == expected_rag

    def test_models_manager_property_when_unlocked(self, orchestrator):
        orchestrator._state = AppState.UNLOCKED
        expected_models = mock.Mock()
        orchestrator.services.models_manager = expected_models
        
        assert orchestrator.models_manager == expected_models


    def test_shutdown_delegates_to_services(self, orchestrator):
        orchestrator.shutdown()
        orchestrator.services.shutdown.assert_called_once()

    def test_shutdown_handles_missing_services(self, orchestrator):
        orchestrator.services = None
        orchestrator.shutdown()

    def test_context_manager_enter_returns_self(self, orchestrator):
        with orchestrator as ctx:
            assert ctx is orchestrator

    def test_context_manager_exit_calls_shutdown(self, orchestrator):
        orchestrator.shutdown = mock.Mock()
        
        with orchestrator:
            pass
        
        orchestrator.shutdown.assert_called_once()

    def test_context_manager_exit_with_exception_still_calls_shutdown(self, orchestrator):
        orchestrator.shutdown = mock.Mock()
        
        with pytest.raises(ValueError):
            with orchestrator:
                raise ValueError("Test exception")
        
        orchestrator.shutdown.assert_called_once()

    def test_explicit_exit_calls_shutdown(self, orchestrator):
        orchestrator.shutdown = mock.Mock()
        
        orchestrator.__exit__(None, None, None)
        
        orchestrator.shutdown.assert_called_once()

