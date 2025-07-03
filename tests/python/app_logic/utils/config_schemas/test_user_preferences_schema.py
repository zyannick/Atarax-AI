import pytest
from ataraxai.app_logic.utils.config_schemas.user_preferences_schema import UserPreferences

def test_default_values():
    prefs = UserPreferences()
    assert prefs.config_version == 1.0
    assert prefs.watched_directories == []
    assert prefs.index_on_startup is True
    assert prefs.realtime_monitoring is False
    assert prefs.theme == "system_default"
    assert prefs.llm_model_path == ""
    assert prefs.whisper_model_path == ""
    assert prefs.language == "en"

def test_custom_values():
    prefs = UserPreferences(
        config_version=2.0,
        watched_directories=["/tmp", "/home/user"],
        index_on_startup=False,
        realtime_monitoring=True,
        theme="dark",
        llm_model_path="/models/llm",
        whisper_model_path="/models/whisper",
        language="fr"
    )
    assert prefs.config_version == 2.0
    assert prefs.watched_directories == ["/tmp", "/home/user"]
    assert prefs.index_on_startup is False
    assert prefs.realtime_monitoring is True
    assert prefs.theme == "dark"
    assert prefs.llm_model_path == "/models/llm"
    assert prefs.whisper_model_path == "/models/whisper"
    assert prefs.language == "fr"

def test_is_setup_complete_true():
    prefs = UserPreferences(llm_model_path="a", whisper_model_path="b")
    assert prefs.is_setup_complete() is True

def test_is_setup_complete_false_llm_missing():
    prefs = UserPreferences(llm_model_path="", whisper_model_path="b")
    assert prefs.is_setup_complete() is False

def test_is_setup_complete_false_whisper_missing():
    prefs = UserPreferences(llm_model_path="a", whisper_model_path="")
    assert prefs.is_setup_complete() is False

def test_is_setup_complete_false_both_missing():
    prefs = UserPreferences(llm_model_path="", whisper_model_path="")
    assert prefs.is_setup_complete() is False