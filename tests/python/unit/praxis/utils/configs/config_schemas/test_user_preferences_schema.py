import pytest
from pydantic import ValidationError
from ataraxai.praxis.utils.configs.config_schemas.user_preferences_schema import UserPreferences, AppTheme

def test_default_values():
    prefs = UserPreferences()
    assert prefs.config_version == 1.0
    assert prefs.index_on_startup is True
    assert prefs.realtime_monitoring is False
    assert prefs.font_size == 12
    assert prefs.notifications_enabled is True
    assert prefs.shortcuts == {}
    assert prefs.theme == AppTheme.SYSTEM_DEFAULT
    assert prefs.language == "en"
    assert prefs.is_setup_complete() is True

def test_custom_values():
    prefs = UserPreferences(
        config_version=2.0,
        index_on_startup=False,
        realtime_monitoring=True,
        font_size=16,
        notifications_enabled=False,
        shortcuts={"save": "Ctrl+S"},
        theme=AppTheme.DARK,
        language="fr"
    )
    assert prefs.config_version == 2.0
    assert prefs.index_on_startup is False
    assert prefs.realtime_monitoring is True
    assert prefs.font_size == 16
    assert prefs.notifications_enabled is False
    assert prefs.shortcuts == {"save": "Ctrl+S"}
    assert prefs.theme == AppTheme.DARK
    assert prefs.language == "fr"

@pytest.mark.parametrize("font_size", [7, 33, -1, 100])
def test_font_size_validation(font_size):
    with pytest.raises(ValidationError) as excinfo:
        UserPreferences(font_size=font_size)
    assert "Font size must be between 8 and 32." in str(excinfo.value)

@pytest.mark.parametrize("font_size", [8, 12, 32])
def test_font_size_valid(font_size):
    prefs = UserPreferences(font_size=font_size)
    assert prefs.font_size == font_size

def test_theme_enum():
    prefs = UserPreferences(theme=AppTheme.LIGHT)
    assert prefs.theme == AppTheme.LIGHT

def test_shortcuts_dict():
    prefs = UserPreferences(shortcuts={"open": "Ctrl+O", "close": "Ctrl+W"})
    assert prefs.shortcuts["open"] == "Ctrl+O"
    assert prefs.shortcuts["close"] == "Ctrl+W"