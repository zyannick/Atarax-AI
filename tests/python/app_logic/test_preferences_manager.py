import pytest
import tempfile
from pathlib import Path
import yaml
from ataraxai.app_logic.preferences_manager import PreferencesManager, PREFERENCES_FILENAME
from ataraxai.app_logic.utils.config_schemas.user_preferences_schema import UserPreferences

@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def test_preferences_created_if_not_exist(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    config_file = temp_config_dir / PREFERENCES_FILENAME
    assert config_file.exists()
    assert isinstance(pm.preferences, UserPreferences)

def test_preferences_loaded_if_exist(temp_config_dir):
    prefs_data = {"theme": "dark", "language": "fr"}
    config_file = temp_config_dir / PREFERENCES_FILENAME
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(prefs_data, f)
    pm = PreferencesManager(config_path=temp_config_dir)
    assert pm.preferences.theme == "dark"
    assert pm.preferences.language == "fr"

def test_get_and_set_preferences(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    pm.set("theme", "solarized")
    assert pm.get("theme") == "solarized"
    pm2 = PreferencesManager(config_path=temp_config_dir)
    assert pm2.get("theme") == "solarized"

def test_reload_preferences(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    pm.set("theme", "light")
    config_file = temp_config_dir / PREFERENCES_FILENAME
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"theme": "dark"}, f)
    pm.reload()
    assert pm.get("theme") == "dark"

def test_get_returns_default_for_missing_key(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    assert pm.get("nonexistent_key", default="default_value") == "default_value"
    
def test_set_accepts_various_types(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    pm.set("theme", "monokai")
    assert pm.get("theme") == "monokai"
    pm.set("font_size", 14)
    assert pm.get("font_size") == 14
    pm.set("notifications_enabled", True)
    assert pm.get("notifications_enabled") is True
    pm.set("shortcuts", {"save": "Ctrl+S"})
    assert pm.get("shortcuts") == {"save": "Ctrl+S"}

def test_save_and_load_with_custom_values(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    pm.set("theme", "system_default")
    pm.set("language", "es")
    pm.set("font_size", 16)
    pm2 = PreferencesManager(config_path=temp_config_dir)
    assert pm2.get("theme") == "system_default"
    assert pm2.get("language") == "es"
    assert pm2.get("font_size") == 16

def test_load_or_create_handles_invalid_yaml(temp_config_dir, monkeypatch):
    config_file = temp_config_dir / PREFERENCES_FILENAME
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("theme: [unclosed_list\n")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    pm = PreferencesManager(config_path=temp_config_dir)
    assert isinstance(pm.preferences, UserPreferences)
    assert pm.get("theme") == pm.preferences.theme

def test_save_creates_file_if_missing(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    config_file = temp_config_dir / PREFERENCES_FILENAME
    if config_file.exists():
        config_file.unlink()
    pm._save()
    assert config_file.exists()
