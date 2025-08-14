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
