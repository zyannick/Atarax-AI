import pytest
import tempfile
import shutil
from pathlib import Path
from unittest import mock
from ataraxai.praxis.utils.user_preferences_manager import UserPreferencesManager
from ataraxai.praxis.utils.configs.config_schemas.user_preferences_schema import UserPreferences

@pytest.fixture
def temp_config_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def default_preferences():
    return UserPreferences()

def test_initializes_with_default_preferences(temp_config_dir):
    manager = UserPreferencesManager(config_path=temp_config_dir)
    assert isinstance(manager.preferences, UserPreferences)
    assert manager.config_path.exists()

def test_saves_and_loads_preferences(temp_config_dir):
    original_dir = temp_config_dir
    manager = UserPreferencesManager(config_path=original_dir)
    manager.set("font_size", 16)
    manager._save()
    assert manager.preferences.font_size == 16
    manager2 = UserPreferencesManager(config_path=original_dir)
    assert hasattr(manager2.preferences, "font_size")
    assert getattr(manager2.preferences, "font_size", None) == 16 , f"config dir {original_dir} does not contain expected font_size"

def test_update_user_preferences(temp_config_dir):
    manager = UserPreferencesManager(config_path=temp_config_dir)
    new_prefs = UserPreferences()
    new_prefs.font_size = 17
    manager.update_user_preferences(new_prefs)
    assert manager.preferences.font_size == 17

def test_update_user_preferences_type_error(temp_config_dir):
    manager = UserPreferencesManager(config_path=temp_config_dir)
    with pytest.raises(TypeError):
        manager.update_user_preferences({"not": "UserPreferences"})

def test_get_and_set_preference(temp_config_dir):
    manager = UserPreferencesManager(config_path=temp_config_dir)
    manager.set("font_size", 14)
    assert manager.get("font_size") == 14
    assert manager.get("magic_ataraxai", "default") == "default"

def test_set_invalid_key_raises(temp_config_dir):
    manager = UserPreferencesManager(config_path=temp_config_dir)
    with pytest.raises(AttributeError):
        manager.set("invalid_key", "value")

def test_attributes_error_preferences(temp_config_dir):
    manager = UserPreferencesManager(config_path=temp_config_dir)
    with pytest.raises(AttributeError):
        manager.set("some_key", "before_reload")

def test_load_or_create_handles_corrupt_yaml(temp_config_dir):
    corrupt_file = temp_config_dir / "user_preferences.yaml"
    corrupt_file.write_text("not: valid: yaml: [")
    logger = mock.Mock()
    manager = UserPreferencesManager(config_path=temp_config_dir, logger=logger)
    assert isinstance(manager.preferences, UserPreferences)
    assert logger.error.called

def test_preferences_property_lazy_load(temp_config_dir):
    manager = UserPreferencesManager(config_path=temp_config_dir)
    del manager._preferences
    prefs = manager.preferences
    assert isinstance(prefs, UserPreferences)